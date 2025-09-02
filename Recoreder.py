import sounddevice as sd
import numpy as np
import wave
import time

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

# Silence timeout in seconds
SILENCE_SEC = 2.3
# RMS threshold for speech (tweak as needed)
RMS_THRESHOLD = 500

def bandpass_filter(frame):
    x = frame.astype(np.float32)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/SAMPLE_RATE)
    mask = (freqs >= 300) & (freqs <= 3400)
    X[~mask] = 0
    y = np.fft.irfft(X)
    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y

def rms(frame):
    return np.sqrt(np.mean(frame.astype(np.float64)**2))

def record_until_silence(filename="output.wav"):
    frames = []
    silence_start = None

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
        print("🔊 Calibrating ambient noise...")
        
        print("🎤 Start speaking (will stop after silence)...")
        while True:
            data, _ = stream.read(FRAME_SIZE)
            frame = data[:,0]
            clean = bandpass_filter(frame)
            level = rms(clean)
            now = time.time()

            frames.append(frame.tobytes())

            if level < RMS_THRESHOLD:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= SILENCE_SEC:
                    print(f"🛑 Stopped recording after {SILENCE_SEC} seconds of silence.")
                    break
            else:
                silence_start = None

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    print(f"✅ Saved recording to '{filename}'")

