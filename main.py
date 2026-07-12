import sys
import json
import os
import threading
import subprocess
import difflib
import winsound

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject


# ---- CONFIG ----
VOSK_MODEL_PATH = r"C:\models\vosk-model-en-us-0.22"  # change to your model path
SAMPLE_RATE = 16000
COMMANDS_FILE = "commands.json"
WAKEWORD = "computer"
BLOCKSIZE = 4000  # 0.25s per block at 16kHz


# -----------------------------
# LOAD COMMANDS
# -----------------------------
def load_commands():
    if not os.path.exists(COMMANDS_FILE):
        return {}
    with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


commands = load_commands()
vosk_model = Model(VOSK_MODEL_PATH)


# -----------------------------
# SAFE FLOAT32 → INT16 (for Talk button)
# -----------------------------
def float32_to_int16(audio):
    data = audio.flatten()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767).astype(np.int16).tobytes()


# -----------------------------
# NOISE SUPPRESSION (light)
# -----------------------------
def noise_suppress(audio_bytes, threshold=200):
    data = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    mask = np.abs(data) < threshold
    data[mask] = 0
    return data.astype(np.int16).tobytes()


# -----------------------------
# AUTO-GAIN CONTROL (gentle)
# -----------------------------
def apply_agc(audio_bytes):
    data = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    peak = np.max(np.abs(data)) + 1e-6
    gain = min(1.5, 20000.0 / peak)
    data = np.clip(data * gain, -32767, 32767)
    return data.astype(np.int16).tobytes()


# -----------------------------
# FUZZY COMMAND ROUTER
# -----------------------------
def handle_command(text: str) -> str:
    cmd = text.lower().strip()

    keys = list(commands.keys())
    match = difflib.get_close_matches(cmd, keys, n=1, cutoff=0.55)

    if not match:
        return f"I heard: '{cmd}', but I don't know that command."

    best_key = match[0]
    entry = commands[best_key]
    action = entry.get("action")
    target = entry.get("target")

    if action == "run":
        subprocess.Popen(target, shell=True)
        return f"Running {best_key}"

    if action == "system":
        if target == "volume_up":
            return "Volume up."
        if target == "close_window":
            return "Closing window."

    return "Command found but no action implemented."


# -----------------------------
# FUZZY WAKEWORD MATCHING
# -----------------------------
def is_wakeword(text: str) -> bool:
    text = text.lower().strip()

    if WAKEWORD in text:
        return True

    candidates = ["computer", "compter", "commuter", "comp you ter"]
    matches = difflib.get_close_matches(text, candidates, cutoff=0.70)
    return len(matches) > 0


# -----------------------------
# UI SIGNALS
# -----------------------------
class UiSignals(QObject):
    status = pyqtSignal(str)


# -----------------------------
# CONTINUOUS WAKEWORD + 5s COMMAND LISTENING
# -----------------------------
class KWSStream(threading.Thread):
    def __init__(self, signals: UiSignals):
        super().__init__(daemon=True)
        self.signals = signals
        self._running = True

    def run(self):
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        recognizer.SetWords(True)

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
        ) as stream:
            while self._running:
                raw = stream.read(BLOCKSIZE)[0]
                data = bytes(raw)

                data = noise_suppress(data)
                data = apply_agc(data)

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
                if is_wakeword(text):
                    winsound.Beep(1200, 120)
                    self.signals.status.emit("Wake-word COMPUTER detected. Listening for 5 seconds…")
                    self.listen_for_command(stream)
                    # after this returns, loop continues and waits for wake word again
                    continue

    def listen_for_command(self, stream):
        """Use the SAME stream to record ~5 seconds and process as a command."""
        duration = 5.0
        total_samples = int(duration * SAMPLE_RATE)
        collected = []
        samples = 0

        while samples < total_samples and self._running:
            raw = stream.read(BLOCKSIZE)[0]
            collected.append(bytes(raw))
            samples += BLOCKSIZE

        audio_bytes = b"".join(collected)
        audio_bytes = noise_suppress(audio_bytes)
        audio_bytes = apply_agc(audio_bytes)

        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        if recognizer.AcceptWaveform(audio_bytes):
            result = recognizer.Result()
        else:
            result = recognizer.FinalResult()

        try:
            j = json.loads(result)
            text = j.get("text", "").strip()
        except:
            text = ""

        if not text:
            self.signals.status.emit("I didn't catch that.")
            return

        response = handle_command(text)
        self.signals.status.emit(response)

    def stop(self):
        self._running = False


# -----------------------------
# ONE-SHOT LISTENER (Talk button)
# -----------------------------
class OneShotListener(threading.Thread):
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

            audio_bytes = float32_to_int16(audio)
            audio_bytes = noise_suppress(audio_bytes)
            audio_bytes = apply_agc(audio_bytes)

            recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
            if recognizer.AcceptWaveform(audio_bytes):
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


# -----------------------------
# MAIN WINDOW
# -----------------------------
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


# -----------------------------
# MAIN
# -----------------------------
def main():
    app = QApplication(sys.argv)
    win = VoiceWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
