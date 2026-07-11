import sys
import threading
import subprocess
import time
import json
import os

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject


# -----------------------------
# CONFIG
# -----------------------------
VOSK_MODEL_PATH = r"C:\models\vosk-model-en-us-0.22"
SAMPLE_RATE = 16000
COMMANDS_FILE = "commands.json"


# -----------------------------
# Load commands.json
# -----------------------------
def load_commands():
    if not os.path.exists(COMMANDS_FILE):
        print("commands.json not found!")
        return {}
    with open(COMMANDS_FILE, "r") as f:
        return json.load(f)

commands = load_commands()


# -----------------------------
# MODELS
# -----------------------------
vosk_model = Model(VOSK_MODEL_PATH)


# -----------------------------
# STT: Vosk (commands)
# -----------------------------
def transcribe_vosk(audio_data, sample_rate):
    recognizer = KaldiRecognizer(vosk_model, sample_rate)
    recognizer.SetWords(True)

    pcm_data = (audio_data * 32767).astype("int16").tobytes()

    if recognizer.AcceptWaveform(pcm_data):
        result = recognizer.Result()
    else:
        result = recognizer.FinalResult()

    text = json.loads(result).get("text", "")
    print(f"[stt] text='{text}'")
    return text.strip()


# -----------------------------
# Command router (JSON-based)
# -----------------------------
def handle_command(text: str):
    cmd = text.lower().strip()
    print(f"[assistant] understood: {cmd}")

    # fuzzy match: find closest command key
    best_key = None
    for key in commands.keys():
        if key in cmd:
            best_key = key
            break

    if not best_key:
        return "I don't know how to do that yet."

    entry = commands[best_key]
    action = entry.get("action")
    target = entry.get("target")

    if action == "run":
        subprocess.Popen(target, shell=True)
        return f"Running {target}"

    if action == "system":
        if target == "volume_up":
            return "Volume up."
        if target == "close_window":
            return "Closing window."

    return "Command found but no action implemented."


# -----------------------------
# Worker signals
# -----------------------------
class WorkerSignals(QObject):
    status = pyqtSignal(str)


# -----------------------------
# Audio worker (Vosk STT)
# -----------------------------
class AudioWorker(threading.Thread):
    def __init__(self, signals: WorkerSignals, duration: float = 4.0):
        super().__init__(daemon=True)
        self.signals = signals
        self.duration = duration

    def run(self):
        try:
            self.signals.status.emit("Listening...")
            sample_rate = SAMPLE_RATE

            self.signals.status.emit("Recording...")
            audio = sd.rec(
                int(self.duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            self.signals.status.emit("Transcribing...")
            text = transcribe_vosk(audio, sample_rate)
            if not text:
                self.signals.status.emit("I didn't catch that.")
                return

            response = handle_command(text)
            self.signals.status.emit(response)
        except Exception as e:
            self.signals.status.emit(f"Error: {e}")


# -----------------------------
# PyQt6 UI
# -----------------------------
class VoiceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Voice Assistant")
        self.setFixedSize(320, 160)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Press the button and speak")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Talk")
        self.button.clicked.connect(self.on_button_clicked)

        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.signals = WorkerSignals()
        self.signals.status.connect(self.update_status)

        self._busy = False

    def update_status(self, msg: str):
        self.label.setText(msg)

    def on_button_clicked(self):
        if self._busy:
            return
        self._busy = True
        self.label.setText("Starting...")
        worker = AudioWorker(self.signals, duration=4.0)
        worker.start()

        def reset_busy():
            time.sleep(5)
            self._busy = False

        threading.Thread(target=reset_busy, daemon=True).start()


# -----------------------------
# Main
# -----------------------------
def main():
    app = QApplication(sys.argv)
    win = VoiceWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
