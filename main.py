import sys
import json
import os
import threading
import subprocess

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject


# ---- CONFIG ----
VOSK_MODEL_PATH = r"C:\models\vosk-model-en-us-0.22"  # change to your Vosk model path
SAMPLE_RATE = 16000
COMMANDS_FILE = "commands.json"
WAKEWORD = "computer"


def load_commands():
    if not os.path.exists(COMMANDS_FILE):
        return {}
    with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


commands = load_commands()
vosk_model = Model(VOSK_MODEL_PATH)


def handle_command(text: str) -> str:
    cmd = text.lower().strip()
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


class UiSignals(QObject):
    status = pyqtSignal(str)


class KWSStream(threading.Thread):
    """
    Keyword spotting stream:
    - Continuously listens with Vosk
    - If it hears WAKEWORD ("computer"), it arms for the next utterance as a command
    """

    def __init__(self, signals: UiSignals):
        super().__init__(daemon=True)
        self.signals = signals
        self._running = True
        self._armed_for_command = False

    def run(self):
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        recognizer.SetWords(True)

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
        ) as stream:
            while self._running:
                data = bytes(stream.read(8000)[0])
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                else:
                    result = recognizer.PartialResult()

                try:
                    j = json.loads(result)
                except Exception:
                    continue

                text = j.get("text", "") or j.get("partial", "")
                text = text.strip().lower()
                if not text:
                    continue

                # Wake-word detection
                if not self._armed_for_command and WAKEWORD in text:
                    self._armed_for_command = True
                    self.signals.status.emit("Wake-word COMPUTER detected. Say your command.")
                    continue

                # Command capture after wake-word
                if self._armed_for_command and text:
                    self._armed_for_command = False
                    response = handle_command(text)
                    self.signals.status.emit(response)

    def stop(self):
        self._running = False


class OneShotListener(threading.Thread):
    """
    One-shot listener for the Talk button:
    - Records a short chunk
    - Runs Vosk
    - Treats the result as a command directly
    """

    def __init__(self, signals: UiSignals, duration: float = 4.0):
        super().__init__(daemon=True)
        self.signals = signals
        self.duration = duration

    def run(self):
        try:
            self.signals.status.emit("Listening…")
            audio = sd.rec(
                int(self.duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = (audio.flatten() * 32767).astype("int16").tobytes()

            recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
            if recognizer.AcceptWaveform(audio):
                result = recognizer.Result()
            else:
                result = recognizer.FinalResult()

            j = json.loads(result)
            text = j.get("text", "").strip()
            if not text:
                self.signals.status.emit("I didn't catch that.")
                return

            response = handle_command(text)
            self.signals.status.emit(response)
        except Exception as e:
            self.signals.status.emit(f"Error: {e}")


class VoiceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COMPUTER Assistant")
        self.setFixedSize(380, 200)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Say “computer” then your command,\nor press Talk.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Talk")
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.signals = UiSignals()
        self.signals.status.connect(self.update_status)

        self.kws = KWSStream(self.signals)
        self.kws.start()

        self.button.clicked.connect(self.on_button_clicked)

    def update_status(self, msg: str):
        self.label.setText(msg)

    def on_button_clicked(self):
        listener = OneShotListener(self.signals, duration=4.0)
        listener.start()

    def closeEvent(self, event):
        if hasattr(self, "kws"):
            self.kws.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = VoiceWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
