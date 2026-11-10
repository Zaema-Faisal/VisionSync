import os
import re
import json
import time
import threading
import winsound
from collections import deque
from typing import Callable, Optional

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer
import pyttsx3

from ai_router import ask_ai, reset_conversation_history


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "vosk-model")
AUDIO_FILE = os.path.join(BASE_DIR, "voice_input.wav")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)

SILENCE_END_SECONDS = 3.0
MAX_RECORDING_SECONDS = 150
WAIT_FOR_SPEECH_SECONDS = 30
MIN_SPEECH_SECONDS = 0.35
PRE_ROLL_SECONDS = 0.5

WAKE_WORD = "vision"

# Models are loaded only when voice is enabled, not when FastAPI imports this file.
vosk_model = None
whisper_model = None

_stop_event = threading.Event()
_service_thread: Optional[threading.Thread] = None
_event_callback: Optional[Callable] = None
_service_lock = threading.Lock()
_voice_enabled = False


def emit_event(event_type: str, **payload):
    """Send voice activity to the API/UI without coupling this module to FastAPI."""
    callback = _event_callback
    if callback:
        try:
            callback({"type": event_type, **payload})
        except Exception as error:
            print("[Voice event callback error]", repr(error))


def get_voice_status():
    return {
        "enabled": _voice_enabled,
        "running": bool(_service_thread and _service_thread.is_alive()),
        "stopping": _stop_event.is_set()
    }


def load_models():
    global vosk_model, whisper_model

    if vosk_model is None:
        print("\nLoading Vosk wake-word model...")
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model folder not found: {VOSK_MODEL_PATH}"
            )
        vosk_model = Model(VOSK_MODEL_PATH)
        print("Vosk loaded.")

    if whisper_model is None:
        print("\nLoading Whisper model...")
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper loaded.")


# ============================================================
# TEXT-TO-SPEECH
# ============================================================

def clean_for_speech(text):
    if not text:
        return ""

    text = re.sub(
        r"[\U0001F000-\U0001FAFF"
        r"\U00002600-\U000027BF"
        r"\uFE0F\u200D\u20E3]+",
        "",
        str(text)
    )
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def speak(text):
    speech_text = clean_for_speech(text)

    if not speech_text:
        print("[No speakable response]")
        return False

    print("\nVision:", speech_text)
    emit_event("status", status="speaking")
    emit_event("assistant_message", text=str(text))

    tts = None
    try:
        tts = pyttsx3.init(driverName="sapi5")

        voices = tts.getProperty("voices") or []
        zira = next(
            (voice for voice in voices if "zira" in voice.name.lower()),
            None
        )

        if zira:
            tts.setProperty("voice", zira.id)

        tts.setProperty("rate", 155)
        tts.setProperty("volume", 1.0)
        tts.say(speech_text)
        tts.runAndWait()

        print("[Speech completed]")
        return True

    except Exception as error:
        print("[TTS error]", repr(error))
        emit_event("error", message=f"Speech playback failed: {error}")
        return False

    finally:
        if tts is not None:
            try:
                tts.stop()
            except Exception:
                pass
            del tts


def activation_sound():
    try:
        winsound.Beep(1000, 130)
        winsound.Beep(1350, 130)
    except Exception:
        pass


# ============================================================
# WAKE WORD
# ============================================================

def wait_for_wake_word():
    print("\n================================")
    print("VISION IS IN STANDBY")
    print("Say: VISION")
    print("================================")

    emit_event("status", status="standby", message='Say "Vision" to wake me.')

    recognizer = KaldiRecognizer(
        vosk_model,
        SAMPLE_RATE,
        json.dumps(["vision", "[unk]"])
    )

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype="int16",
        channels=1
    ) as stream:
        while not _stop_event.is_set():
            audio_data, overflowed = stream.read(CHUNK_SIZE)
            audio_bytes = bytes(audio_data)

            if recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(recognizer.Result())
            else:
                result = json.loads(recognizer.PartialResult())

            detected = result.get(
                "text", result.get("partial", "")
            ).lower()

            words = re.findall(r"[a-z]+", detected)
            if WAKE_WORD in words:
                print("[Wake word detected]")
                emit_event("wake_detected")
                break

    if _stop_event.is_set():
        return False

    activation_sound()
    speak("I'm listening. What would you like to talk about?")
    emit_event("status", status="listening", message="Conversation active.")
    return True


# ============================================================
# RECORDING
# ============================================================

def record_voice():
    print("\nListening. Take your time; I'm waiting for you to speak.")
    emit_event("status", status="listening", message="Waiting for speech.")

    pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS / CHUNK_SECONDS))
    pre_roll = deque(maxlen=pre_roll_chunks)

    frames = []
    noise_samples = []
    speech_started = False
    speech_run = 0.0
    silence_run = 0.0
    elapsed = 0.0
    wait_elapsed = 0.0
    threshold = 0.004

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE
    ) as stream:

        print("[Calibrating microphone...]")
        calibration_chunks = max(3, int(0.7 / CHUNK_SECONDS))

        for _ in range(calibration_chunks):
            if _stop_event.is_set():
                return False
            audio, overflowed = stream.read(CHUNK_SIZE)
            chunk = audio.reshape(-1).copy()
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            noise_samples.append(rms)
            pre_roll.append(chunk)
            elapsed += CHUNK_SECONDS

        noise_floor = float(np.median(noise_samples))
        threshold = max(0.003, min(0.025, noise_floor * 2.8))
        print(f"[Microphone ready; speech threshold={threshold:.4f}]")
        emit_event("status", status="listening", message="Microphone ready.")

        while elapsed < MAX_RECORDING_SECONDS and not _stop_event.is_set():
            audio, overflowed = stream.read(CHUNK_SIZE)
            chunk = audio.reshape(-1).copy()
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            elapsed += CHUNK_SECONDS

            if not speech_started:
                pre_roll.append(chunk)

                if rms >= threshold:
                    speech_run += CHUNK_SECONDS
                    if speech_run >= MIN_SPEECH_SECONDS:
                        speech_started = True
                        frames.extend(list(pre_roll))
                        print("[Speech detected; recording...]")
                        emit_event("status", status="recording", message="Recording your voice.")
                else:
                    speech_run = 0.0
                    wait_elapsed += CHUNK_SECONDS

                    if wait_elapsed >= WAIT_FOR_SPEECH_SECONDS:
                        print("[No speech started within wait period]")
                        return False
                continue

            frames.append(chunk)

            if rms < threshold:
                silence_run += CHUNK_SECONDS
            else:
                silence_run = 0.0

            if silence_run >= SILENCE_END_SECONDS:
                print("[End of speech detected]")
                break

    if _stop_event.is_set() or not speech_started or not frames:
        return False

    audio_data = np.concatenate(frames)
    audio_int16 = np.clip(
        audio_data * 32767, -32768, 32767
    ).astype(np.int16)

    write(AUDIO_FILE, SAMPLE_RATE, audio_int16)
    print(f"[Recording saved: {len(audio_data) / SAMPLE_RATE:.1f} seconds]")
    return True


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def speech_to_text():
    print("\n[Whisper is transcribing...]")
    emit_event("status", status="transcribing", message="Converting speech to text.")

    segments, info = whisper_model.transcribe(
        AUDIO_FILE,
        language="en",
        temperature=0.0,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 700,
            "speech_pad_ms": 600
        },
        condition_on_previous_text=False,
        initial_prompt=(
            "Voice assistant control commands: bye, goodbye, end, end chat, "
            "end the chat, end conversation, that's all, I'm done, see you, "
            "shutdown Vision, shut down Vision. Short spoken commands may "
            "consist of only one or two words."
        )
    )

    segment_list = list(segments)
    text = " ".join(
        segment.text.strip()
        for segment in segment_list
        if segment.text.strip()
    ).strip()

    if not text:
        print("[No speech recognized]")
        return ""

    print("[Whisper raw transcription]", repr(text))

    # CRITICAL: process voice-control commands BEFORE short-phrase and
    # confidence filters. "Bye", "End", and "Goodbye" are valid commands.
    if is_exit_command(text) or is_end_chat(text):
        print("[Recognized control command]", repr(text))
        emit_event("user_message", text=text)
        return text

    words = re.findall(r"[A-Za-z']+", text)

    if len(words) < 2:
        print(f"[Rejected incomplete transcription: {text!r}]")
        return ""

    avg_logprobs = [
        getattr(segment, "avg_logprob", None)
        for segment in segment_list
        if getattr(segment, "avg_logprob", None) is not None
    ]
    no_speech_probs = [
        getattr(segment, "no_speech_prob", None)
        for segment in segment_list
        if getattr(segment, "no_speech_prob", None) is not None
    ]

    # Keep normal conversation quality checks, but do not let these filters
    # block the control commands handled above.
    if avg_logprobs and sum(avg_logprobs) / len(avg_logprobs) < -1.35:
        print(f"[Low-confidence transcription; rejecting: {text!r}]")
        return ""

    if no_speech_probs and sum(no_speech_probs) / len(no_speech_probs) > 0.80:
        print(f"[Rejected likely-noise transcription: {text!r}]")
        return ""

    print("[Recognized speech]", text)
    emit_event("user_message", text=text)
    return text


# ============================================================
# CHAT CONTROL COMMANDS
# ============================================================

def normalize_command(text):
    """Normalize Whisper punctuation/capitalization for reliable command matching."""
    text = str(text or "").lower().strip()
    text = text.replace("’", "'")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_end_chat(text):
    """Detect natural farewells before sending speech to the AI model."""
    message = normalize_command(text)

    # Whisper often transcribes "goodbye" as two words and may repeat it.
    message = re.sub(r"\\bgood\\s+bye\\b", "goodbye", message)
    message = re.sub(r"\\s+", " ", message).strip()

    # Accept a farewell even when preceded by a short polite phrase.
    # Repeated bye/goodbye tokens are intentionally accepted.
    tokens = message.split()
    allowed = {
        "bye", "goodbye", "thank", "thanks", "you", "very", "much",
        "okay", "ok", "alright", "well", "then", "vision", "please",
        "thats", "that", "is", "all", "im", "i", "am", "done",
        "end", "the", "chat", "conversation", "finish", "see",
        "later", "stop", "chatting", "go", "to", "sleep", "now"
    }
    farewell = {"bye", "goodbye", "end", "finish", "done", "sleep", "stop"}
    if not tokens or not any(token in farewell for token in tokens):
        return False

    # Reject mixed substantive requests such as "bye, add milk to my tasks".
    return all(token in allowed for token in tokens)


def is_exit_command(text):
    message = normalize_command(text)
    commands = {
        "shutdown vision", "shut down vision",
        "exit assistant", "close assistant",
        "turn off vision", "quit vision",
        "shutdown vision now", "shut down vision now",
        "please shutdown vision", "please shut down vision",
    }
    return message in commands


# ============================================================
# CONVERSATION SESSION
# ============================================================

def conversation_session():
    print("\n================================")
    print("VISION CONVERSATION ACTIVE")
    print("You do NOT need to say Vision again.")
    print("Say 'End chat' to return to standby.")
    print("Say 'Shutdown Vision' to stop voice service.")
    print("================================")

    emit_event("status", status="listening", message="Conversation active. You do not need to repeat the wake word.")

    while not _stop_event.is_set():
        try:
            recorded = record_voice()

            if _stop_event.is_set():
                break

            if not recorded:
                continue

            user_text = speech_to_text()

            if _stop_event.is_set():
                break

            if not user_text:
                speak("I didn't catch that clearly. Please say it again. Take your time.")
                emit_event("status", status="listening", message="Please repeat that.")
                continue

            if is_exit_command(user_text):
                print("[VOICE COMMAND] Shutdown requested.")
                speak("Goodbye. Voice assistant is turning off.")
                reset_conversation_history()
                return "exit"

            if is_end_chat(user_text):
                print("[VOICE COMMAND] End chat detected. Returning to wake-word standby.")
                speak("Goodbye. I'll return to standby. Say Vision whenever you need me.")
                reset_conversation_history()
                return "standby"

            print("\n[Sending question to AI...]")
            emit_event("status", status="thinking", message="Thinking about your question.")

            try:
                response = ask_ai(user_text)
                if response:
                    speak(response)
                else:
                    speak("I couldn't generate a response. Please try again.")
            except Exception as error:
                print("[AI error]", repr(error))
                emit_event("error", message=f"AI error: {error}")
                speak("I'm having trouble processing that. Please try again.")

            if not _stop_event.is_set():
                emit_event("status", status="listening", message="Chat remains active. Speak your next question.")
                print("\n[Chat remains active. Speak your next question.]")

        except Exception as error:
            print("[Conversation error]", repr(error))
            emit_event("error", message=str(error))
            speak("Something went wrong. Please try again.")
            time.sleep(0.3)

    return "stopped"


# ============================================================
# SERVICE THREAD API (USED BY FASTAPI)
# ============================================================

def _voice_service_main():
    global _voice_enabled

    try:
        load_models()

        while not _stop_event.is_set():
            woke = wait_for_wake_word()

            if _stop_event.is_set():
                break

            if not woke:
                continue

            result = conversation_session()

            if result == "exit":
                break

            if result == "standby":
                emit_event("status", status="standby", message='Say "Vision" to wake me.')

        print("[Voice service stopped.]")

    except Exception as error:
        print("[Voice service error]", repr(error))
        emit_event("error", message=str(error))
        emit_event("status", status="error", message=str(error))

    finally:
        _voice_enabled = False
        emit_event("voice_enabled", enabled=False)
        emit_event("status", status="disabled", message="Voice assistant is off.")


def start_voice_service(callback=None):
    global _event_callback, _service_thread, _voice_enabled

    with _service_lock:
        if _service_thread and _service_thread.is_alive():
            return {"started": False, "message": "Voice service is already running."}

        _event_callback = callback
        _stop_event.clear()
        _voice_enabled = True

        _service_thread = threading.Thread(
            target=_voice_service_main,
            name="VisionSyncVoice",
            daemon=True
        )
        _service_thread.start()

    emit_event("voice_enabled", enabled=True)
    emit_event("status", status="loading", message="Loading voice models.")
    return {"started": True, "message": "Voice service is starting."}


def stop_voice_service():
    global _voice_enabled

    _stop_event.set()
    _voice_enabled = False
    emit_event("voice_enabled", enabled=False)
    emit_event("status", status="stopping", message="Stopping voice assistant.")
    return {"stopped": True, "message": "Voice service stop requested."}


# ============================================================
# STANDALONE MODE
# ============================================================

def main():
    print("\n================================")
    print("       VISION SMART MIRROR")
    print("================================")
    print("Wake word: Vision")
    print("Wake detection: Vosk")
    print("Speech recognition: Whisper")
    print("AI: Gemini / Ollama")
    print("Speech: Windows SAPI5")
    print("\nPress Ctrl+C to exit.")

    def console_event(event):
        if event.get("type") == "status":
            print("[Voice status]", event.get("status"), event.get("message", ""))

    start_voice_service(console_event)

    try:
        while _service_thread and _service_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nVision shutting down...")
        stop_voice_service()


if __name__ == "__main__":
    main()
