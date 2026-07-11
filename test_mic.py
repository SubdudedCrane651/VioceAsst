import pyaudio
import numpy as np

DEVICE_INDEXES = [1, 7, 15, 19]  # all your webcam mic entries
RATE = 16000
CHUNK = 4096

pa = pyaudio.PyAudio()

for idx in DEVICE_INDEXES:
    print(f"\nTesting device {idx}...")
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=idx,
            frames_per_buffer=CHUNK
        )

        data = stream.read(CHUNK, exception_on_overflow=False)
        pcm = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(pcm).mean()

        print(f"Volume: {volume}")

        stream.close()

    except Exception as e:
        print(f"Error: {e}")
