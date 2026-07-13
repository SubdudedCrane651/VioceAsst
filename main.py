import sys
import json
import os
import threading
import subprocess
import difflib
import winsound
import re

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject


# ---- CONFIG ----
VOSK_MODEL_PATH = r"C:\models\vosk-model-en-us-0.22"
SAMPLE_RATE = 16000
BLOCKSIZE = 4000  # 0.25s per block
COMMANDS_FILE = "commands.json"
WAKEWORD = "computer"

# -----------------------------
# LOAD DIRECTORIES FROM directory.json
# -----------------------------
def load_directories():
    try:
        with open("directory.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            dirs = data.get("directories", [])
            return [os.path.expandvars(os.path.expanduser(d)) for d in dirs]
    except Exception:
        return []

SEARCH_DIRS = load_directories()

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
# NOISE SUPPRESSION
# -----------------------------
def noise_suppress(audio_bytes, threshold=200):
    data = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    data[np.abs(data) < threshold] = 0
    return data.astype(np.int16).tobytes()


# -----------------------------
# AUTO-GAIN CONTROL
# -----------------------------
def apply_agc(audio_bytes):
    data = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    peak = np.max(np.abs(data)) + 1e-6
    gain = min(1.5, 20000.0 / peak)
    data = np.clip(data * gain, -32767, 32767)
    return data.astype(np.int16).tobytes()


# -----------------------------
# NATURAL-SPEECH FILE OPENING
# -----------------------------
def extract_keywords(text: str) -> str:
    stopwords = {
        "open", "the", "file", "document", "spreadsheet", "photo", "picture",
        "image", "about", "for", "of", "my", "latest", "newest", "last",
        "that", "this", "a", "an", "pdf", "word", "excel", "powerpoint","shortcut"
    }
    words = re.findall(r"\w+", text.lower())
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered)


def find_best_file(keywords: str) -> str | None:
    if not keywords:
        return None

    best_path = None
    best_score = 0.0

    for base in SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                name = os.path.splitext(f)[0].lower()
                score = difflib.SequenceMatcher(None, keywords, name).ratio()
                if score > best_score:
                    best_score = score
                    best_path = os.path.join(root, f)

    if best_score >= 0.5:
        return best_path
    return None


def handle_natural_file_open(text: str) -> str | None:
    keywords = extract_keywords(text)
    path = find_best_file(keywords)
    if not path:
        return None

    try:
        subprocess.Popen(f'"{path}"', shell=True)
        return f"Opening {os.path.basename(path)}"
    except Exception as e:
        return f"Found a file but could not open it: {e}"


# -----------------------------
# FUZZY COMMAND ROUTER
# -----------------------------
def handle_command(text: str) -> str:
    cmd = text.lower().strip()
    keys = list(commands.keys())
    match = difflib.get_close_matches(cmd, keys, n=1, cutoff=0.55)

    if match:
        best_key = match[0]
        entry = commands[best_key]
        action = entry.get("action")
        target = entry.get("target")

        if action == "run":
            subprocess.Popen(target, shell=True)
            return f"Running {best_key}"

        return "Command found but no action implemented."

    # Fallback: natural-speech file opening
    file_result = handle_natural_file_open(text)
    if file_result:
        return file_result

    return f"I heard: '{cmd}', but I don't know that command."


# -----------------------------
# WAKEWORD MATCHING
# -----------------------------
def is_wakeword(text: str) -> bool:
    text = text.lower().strip()
    if WAKEWORD in text:
        return True
    candidates = ["computer", "compter", "commuter", "comp you ter"]
    return len(difflib.get_close_matches(text, candidates, cutoff=0.70)) > 0


# -----------------------------
# UI SIGNALS
# -----------------------------
class UiSignals(QObject):
    status = pyqtSignal(str)


# -----------------------------
# MAIN LISTENER THREAD
# -----------------------------
class KWSStream(threading.Thread):
    def __init__(self, signals: UiSignals):
        super().__init__(daemon=True)
        self.signals = signals
        self.running = True

    def run(self):
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        recognizer.SetWords(True)

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1,
        ) as stream:

            while self.running:
                raw = stream.read(BLOCKSIZE)[0]
                data = apply_agc(noise_suppress(bytes(raw)))

                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                else:
                    result = recognizer.PartialResult()

                try:
                    j = json.loads(result)
                except:
                    continue

                text = j.get("text", "") or j.get("partial", "")
                text = text.strip().lower()

                if not text:
                    continue

                if is_wakeword(text):
                    winsound.Beep(1200, 120)
                    self.signals.status.emit("Wake-word COMPUTER detected. Listening for 5 seconds…")
                    self.listen_for_command(stream)
                    self.signals.status.emit("Sleeping… Say 'computer' to wake me.")
                    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
                    recognizer.SetWords(True)
                    continue

    def listen_for_command(self, stream):
        duration = 5.0
        total_samples = int(duration * SAMPLE_RATE)
        samples = 0
        chunks = []

        while samples < total_samples and self.running:
            raw = stream.read(BLOCKSIZE)[0]
            chunks.append(bytes(raw))
            samples += BLOCKSIZE

        audio_bytes = apply_agc(noise_suppress(b"".join(chunks)))

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
        self.running = False


# -----------------------------
# ONE-SHOT LISTENER (Talk button)
# -----------------------------
class OneShotListener(threading.Thread):
    def __init__(self, signals: UiSignals):
        super().__init__(daemon=True)
        self.signals = signals

    def run(self):
        self.signals.status.emit("Listening…")
        audio = sd.rec(int(4 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()

        data = audio.flatten()
        data = np.nan_to_num(data)
        data = np.clip(data, -1.0, 1.0)
        audio_bytes = (data * 32767).astype(np.int16).tobytes()

        audio_bytes = apply_agc(noise_suppress(audio_bytes))

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

        self.label = QLabel("Say “computer” then your command.\nSleeping until wake word.")
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
        listener = OneShotListener(self.signals)
        listener.start()

    def closeEvent(self, event):
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
