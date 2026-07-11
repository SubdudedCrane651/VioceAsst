import sys
import threading
import subprocess
import time
import json
import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject


# -----------------------------
# VOSK MODEL (use the large one)
# -----------------------------
MODEL_PATH = r"C:\models\vosk-model-en-us-0.22"
model = Model(MODEL_PATH)


# -----------------------------
# STT: voice -> text
# -----------------------------
def transcribe_audio(audio_data, sample_rate):
    recognizer = KaldiRecognizer(model, sample_rate)
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
# Command router
# -----------------------------
def handle_command(text: str):
    cmd = text.lower()
    print(f"[assistant] understood: {cmd}")

    if "open" in cmd and "notepad" in cmd:
        subprocess.Popen("notepad.exe")
        return "Opening Notepad."

    if "open" in cmd and "chrome" in cmd:
        subprocess.Popen(
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shell=True
        )
        return "Opening Chrome."

    if "volume up" in cmd:
        return "Volume up (stub)."

    return "I don't know how to do that yet."


# -----------------------------
# Worker signals
# -----------------------------
class WorkerSignals(QObject):
    status = pyqtSignal(str)


# -----------------------------
# Audio worker thread
# -----------------------------
class AudioWorker(threading.Thread):
    def __init__(self, signals: WorkerSignals, duration: float = 4.0):
        super().__init__(daemon=True)
        self.signals = signals
        self.duration = duration

    def run(self):
        try:
            self.signals.status.emit("Listening...")
            sample_rate = 16000

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
            text = transcribe_audio(audio, sample_rate)
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
