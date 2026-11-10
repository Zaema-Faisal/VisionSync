import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")

sample_rate = 16000
duration = 5

print("\nSpeak now...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

write("test.wav", sample_rate, audio)

print("Recording finished.")
print("Transcribing...")

segments, info = model.transcribe(
    "test.wav",
    language="en"
)

text = " ".join(segment.text for segment in segments)

print("\nYou said:")
print(text)