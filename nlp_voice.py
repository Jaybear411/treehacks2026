"""
NLP Voice Controller - Push-to-talk with Gemini-powered natural language understanding.

Features:
- Object extraction: "Hey Jarvis, get me the pill bottle" → "pill bottle"
- Conversational mode: free-form chat via Gemini + ElevenLabs TTS response

Flow (object mode - SPACE):
1. User presses space → microphone opens, records until silence
2. Speech is transcribed via Google Speech API
3. Transcription is sent to Gemini to extract the object name
4. Returns the extracted object name for VLM detection

Flow (conversation mode - C):
1. User presses C → microphone opens, records until silence
2. Speech is transcribed via Google Speech API
3. Transcription is sent to Gemini for conversational response
4. Gemini response is spoken aloud via ElevenLabs TTS
"""

import os
import subprocess
import tempfile
import time
from typing import Optional

import speech_recognition as sr

from config import VOICE, GEMINI, ELEVENLABS

# ---------------------------------------------------------------------------
# Load .env once at module level so every key is available before any init
# ---------------------------------------------------------------------------
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=True)
    print(f"Loaded .env from {_ENV_PATH}")
    print(f"  GEMINI_API_KEY: {'set' if os.environ.get('GEMINI_API_KEY') else 'NOT FOUND'}")
    print(f"  ELEVENLABS_API_KEY: {'set' if os.environ.get('ELEVENLABS_API_KEY') else 'NOT FOUND'}")
except ImportError:
    print("WARNING: python-dotenv not installed, .env file will not be loaded")


class NLPVoiceController:
    """
    Push-to-talk voice controller with Gemini NLP for object extraction.

    Usage:
        nlp = NLPVoiceController()
        nlp.calibrate()

        # When user presses space:
        object_name = nlp.listen_and_extract()
        # → "pill bottle"
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Configure recognizer
        self.recognizer.energy_threshold = VOICE.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        # Initialize Gemini (new google.genai SDK)
        self._client = None
        self._chat = None
        self._init_gemini()

        # Initialize ElevenLabs TTS
        self._tts_client = None
        self._init_elevenlabs()

    def _init_gemini(self):
        """Initialize the Gemini client using the new google.genai SDK."""
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in your environment or create a .env file with:\n"
                "  GEMINI_API_KEY=your_key_here"
            )

        self._client = genai.Client(api_key=api_key)

        # Start a persistent chat session for conversation mode
        # System instruction is set via config so Gemini knows its persona
        self._chat = self._client.chats.create(
            model=GEMINI.model_name,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are Jarvis, the friendly AI assistant for the Magic Table — "
                    "a robotic system that retrieves objects from a tabletop for users. "
                    "You are warm, concise, and helpful. Keep your responses to 1-3 "
                    "sentences since they will be spoken aloud. Be conversational and "
                    "natural, like a helpful butler."
                ),
            ),
        )

        print(f"Gemini initialized (model: {GEMINI.model_name})")

    def _init_elevenlabs(self):
        """Initialize the ElevenLabs TTS client."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")

        if not api_key:
            print(
                "WARNING: ELEVENLABS_API_KEY not found. "
                "Conversation mode TTS will be unavailable."
            )
            return

        from elevenlabs.client import ElevenLabs
        self._tts_client = ElevenLabs(api_key=api_key)
        print("ElevenLabs TTS initialized")

    def calibrate(self, duration: float = 1.0):
        """Calibrate microphone for ambient noise."""
        print(f"Calibrating microphone ({duration}s)...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print(f"Ambient noise calibration done (threshold: {self.recognizer.energy_threshold:.0f})")

    def record_and_transcribe(self) -> Optional[str]:
        """
        Record audio until silence and transcribe via Google Speech API.

        Returns transcribed text, or None if nothing was understood.
        """
        with self.microphone as source:
            print("LISTENING... (speak now)")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=VOICE.listen_timeout,
                    phrase_time_limit=VOICE.phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                print("No speech detected (timeout)")
                return None

        print("Transcribing...")
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Heard: '{text}'")
            return text
        except sr.UnknownValueError:
            print("Could not understand speech")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

    # Sentinel value returned when the user says goodbye
    GOODBYE = "GOODBYE"

    # Sentinel value returned when the user requests cleanup mode
    CLEANUP = "CLEANUP"

    def extract_object_name(self, text: str) -> Optional[str]:
        """
        Use Gemini to extract the object name from a natural language command.

        If the user is saying goodbye / ending the session, returns
        NLPVoiceController.GOODBYE instead of an object name.

        If the user says "clean up" (or similar), returns
        NLPVoiceController.CLEANUP instead of an object name.

        Args:
            text: Transcribed voice command

        Returns:
            The extracted object name in lowercase, "GOODBYE", "CLEANUP", or None.
        """
        # Quick check for cleanup intent before hitting Gemini
        text_lower = text.lower()
        if "clean up" in text_lower or "cleanup" in text_lower or "clean it up" in text_lower:
            print(f"Detected cleanup intent in: '{text}'")
            return self.CLEANUP

        prompt = (
            "You are a voice command parser for a robotic object retrieval system. "
            "The user speaks a command to fetch an object from a table.\n\n"
            "RULES:\n"
            "- If the user is asking for an object, return ONLY the object name "
            "in lowercase. No quotes, no punctuation, no explanation.\n"
            "- If the user is saying goodbye, thanks, ending the session, or "
            "anything that is NOT a request for an object, return exactly: GOODBYE\n\n"
            "Examples:\n"
            "  'Hey Jarvis, get me the pill bottle' -> pill bottle\n"
            "  'Hey table, fetch the cardholder' -> cardholder\n"
            "  'bring me my keys please' -> keys\n"
            "  'can you grab the phone' -> phone\n"
            "  'I need my glasses' -> glasses\n"
            "  'Jarvis, where are my AirPods' -> airpods\n"
            "  'get the remote control' -> remote control\n"
            "  'get grandma the red pill bottle' -> red pull bottle\n"
            "  'goodbye' -> GOODBYE\n"
            "  'thanks Jarvis, that will be all' -> GOODBYE\n"
            "  'see you later' -> GOODBYE\n"
            "  'bye bye' -> GOODBYE\n"
            "  'thank you, I'm done' -> GOODBYE\n\n"
            f"User command: '{text}'\n"
            "Response:"
        )

        try:
            start = time.time()
            response = self._client.models.generate_content(
                model=GEMINI.model_name,
                contents=prompt,
            )
            elapsed = time.time() - start

            raw = response.text.strip()
            cleaned = raw.strip("'\".,!?:;").strip()

            # Check for goodbye intent
            if cleaned.upper() == "GOODBYE":
                print(f"Gemini detected goodbye ({elapsed:.2f}s)")
                return self.GOODBYE

            object_name = cleaned.lower()
            print(f"Gemini extracted: '{object_name}' ({elapsed:.2f}s)")
            return object_name if object_name and object_name != "unknown" else None

        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    def scan_table_objects(self, frame) -> list[str]:
        """
        Use Gemini Vision to list all objects visible on the table.

        Sends a camera frame to Gemini and asks it to return a JSON array
        of object names.  The table itself and permanent fixtures are
        excluded.

        Args:
            frame: A BGR numpy array (OpenCV frame) from the overhead camera.

        Returns:
            A list of object name strings (lowercased), e.g.
            ["sunglasses case", "cup", "remote control", "keys"].
            Returns an empty list on failure.
        """
        import json

        import cv2
        from google.genai import types

        # Encode frame as JPEG bytes
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            print("scan_table_objects: failed to encode frame")
            return []
        image_bytes = buffer.tobytes()

        prompt = (
            "You are a vision system for a robotic table. "
            "Look at this overhead image of a tabletop and list ALL distinct, "
            "movable objects you can see on the table surface.\n\n"
            "RULES:\n"
            "- Return ONLY a JSON array of short object names.\n"
            "- Exclude the table itself, the background, and any permanent "
            "fixtures (legs, frame, gantry rails).\n"
            "- Each name should be a simple noun phrase in lowercase "
            '(e.g. "sunglasses case", "cup", "remote control").\n'
            "- If you see no objects, return an empty array: []\n\n"
            'Example response: ["sunglasses case", "cup", "remote control", "keys"]\n'
            "Response:"
        )

        try:
            start = time.time()
            response = self._client.models.generate_content(
                model=GEMINI.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes, mime_type="image/jpeg"
                    ),
                    prompt,
                ],
            )
            elapsed = time.time() - start

            raw = response.text.strip()
            print(f"Gemini Vision scan ({elapsed:.2f}s): {raw}")

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0]

            objects = json.loads(raw)

            if not isinstance(objects, list):
                print(f"scan_table_objects: expected list, got {type(objects)}")
                return []

            cleaned = [
                obj.strip().lower()
                for obj in objects
                if isinstance(obj, str) and obj.strip()
            ]
            print(f"Detected {len(cleaned)} objects: {cleaned}")
            return cleaned

        except json.JSONDecodeError as e:
            print(f"scan_table_objects JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"scan_table_objects error: {e}")
            return []

    def converse(self, text: str) -> Optional[str]:
        """
        Send a message to Gemini in conversation mode and get a spoken response.

        Uses a persistent chat session so Gemini remembers context.

        Args:
            text: The user's transcribed speech.

        Returns:
            Gemini's response text, or None on error.
        """
        try:
            start = time.time()
            response = self._chat.send_message(text)
            elapsed = time.time() - start

            reply = response.text.strip()
            print(f"Gemini replied ({elapsed:.2f}s): '{reply}'")
            return reply

        except Exception as e:
            print(f"Gemini conversation error: {e}")
            return None

    def speak(self, text: str):
        """
        Convert text to speech via ElevenLabs and play it.

        Falls back to printing the text if TTS is unavailable.

        Args:
            text: The text to speak aloud.
        """
        if not self._tts_client:
            print(f"[TTS unavailable] Jarvis says: {text}")
            return

        try:
            start = time.time()
            audio_generator = self._tts_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVENLABS.voice_id,
                model_id=ELEVENLABS.model_id,
                output_format=ELEVENLABS.output_format,
            )

            # Collect audio bytes from the generator
            audio_bytes = b"".join(audio_generator)
            elapsed = time.time() - start
            print(f"ElevenLabs TTS ({elapsed:.2f}s, {len(audio_bytes)} bytes)")

            # Write to temp file and play with afplay
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            subprocess.run(["afplay", tmp_path], check=True)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            print(f"[Fallback] Jarvis says: {text}")

    def listen_and_converse(self) -> Optional[str]:
        """
        Full conversation pipeline: record → transcribe → Gemini chat → ElevenLabs TTS.

        Returns:
            Gemini's response text, or None if any step failed.
        """
        text = self.record_and_transcribe()
        if not text:
            return None

        reply = self.converse(text)
        if reply:
            self.speak(reply)
        return reply

    def listen_and_extract(self) -> Optional[str]:
        """
        Full pipeline: record -> transcribe -> extract object name via Gemini.

        Returns:
            The object name string, or None if any step failed.
        """
        text = self.record_and_transcribe()
        if not text:
            return None

        return self.extract_object_name(text)
