"""
NLP Voice Controller - Push-to-talk with Claude-powered natural language understanding.

Features:
- Object extraction: "Hey Jarvis, get me the pill bottle" → "pill bottle"  (Claude)
- Conversational mode: free-form chat via Claude + ElevenLabs TTS response  (Claude)
- Table scanning: vision-based object listing from camera frame            (OpenAI)

Flow (object mode - SPACE):
1. User presses space → microphone opens, records until silence
2. Speech is transcribed via Google Speech API
3. Transcription is sent to Claude to extract the object name
4. Returns the extracted object name for VLM detection

Flow (conversation mode - C):
1. User presses C → microphone opens, records until silence
2. Speech is transcribed via Google Speech API
3. Transcription is sent to Claude for conversational response
4. Claude response is spoken aloud via ElevenLabs TTS
"""

import os
import subprocess
import tempfile
import time
from typing import Callable, Optional

import speech_recognition as sr

from config import VOICE, OPENAI_VISION, ANTHROPIC, ELEVENLABS

# ---------------------------------------------------------------------------
# Load .env once at module level so every key is available before any init
# ---------------------------------------------------------------------------
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=True)
    print(f"Loaded .env from {_ENV_PATH}")
    print(f"  ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT FOUND'}")
    print(f"  OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'NOT FOUND'}")
    print(f"  ELEVENLABS_API_KEY: {'set' if os.environ.get('ELEVENLABS_API_KEY') else 'NOT FOUND'}")
except ImportError:
    print("WARNING: python-dotenv not installed, .env file will not be loaded")


class NLPVoiceController:
    """
    Push-to-talk voice controller with Claude NLP for object extraction.

    Text tasks (object extraction, conversation) use Anthropic Claude.
    Vision tasks (scan_table_objects) use OpenAI.

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

        # Initialize Anthropic Claude (text-only NLP: object extraction + conversation)
        self._anthropic_client = None
        self._conversation_history: list[dict] = []
        self._init_anthropic()

        # Initialize OpenAI (vision tasks only: scan_table_objects)
        self._openai_client = None
        self._init_openai()

        # Initialize ElevenLabs TTS
        self._tts_client = None
        self._init_elevenlabs()

    def _init_anthropic(self):
        """Initialize the Anthropic (Claude) client for text-only NLP tasks."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Set it in your environment or create a .env file with:\n"
                "  ANTHROPIC_API_KEY=your_key_here"
            )

        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

        # Conversation history is kept as a plain list of dicts so we can
        # pass it to the Messages API for multi-turn chat.
        self._conversation_history = []

        print(f"Anthropic Claude initialized (model: {ANTHROPIC.model_name})")

    def _init_openai(self):
        """Initialize the OpenAI client for vision tasks (scan_table_objects)."""
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print(
                "WARNING: OPENAI_API_KEY not found. "
                "Vision features (scan table / describe table) will be unavailable."
            )
            return

        self._openai_client = OpenAI(api_key=api_key)

        print(f"OpenAI initialized for vision (model: {OPENAI_VISION.model_name})")

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

    def record_and_transcribe(self, hold_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Record audio and transcribe via Google Speech API.

        If hold_check is provided, this runs in push-to-talk mode:
        it records only while hold_check() is True.

        Returns transcribed text, or None if nothing was understood.
        """
        if hold_check is not None:
            return self._record_and_transcribe_hold_to_talk(hold_check)

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

    def _record_and_transcribe_hold_to_talk(
        self,
        hold_check: Callable[[], bool],
        max_seconds: float = 20.0,
    ) -> Optional[str]:
        """
        Hold-to-talk capture: record raw mic audio only while key is held.
        """
        with self.microphone as source:
            print("LISTENING... (hold key to talk)")

            chunks: list[bytes] = []
            start = time.time()

            # Small grace period so key transitions don't instantly cut audio.
            release_grace_s = 0.08
            last_held_ts = time.time() if hold_check() else 0.0

            while True:
                now = time.time()
                held = hold_check()

                if held:
                    last_held_ts = now

                # Stop once key has been released beyond grace period.
                if chunks and (now - last_held_ts) > release_grace_s:
                    break

                # If user released immediately without talking.
                if not chunks and not held and (now - start) > 0.5:
                    print("Push-to-talk released before speech")
                    return None

                if (now - start) > max_seconds:
                    print("Push-to-talk max duration reached")
                    break

                try:
                    # PyAudio stream.read(): only pass frame count (CHUNK).
                    # exception_on_overflow was removed in newer PyAudio versions.
                    chunk = source.stream.read(source.CHUNK)
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    return None

                chunks.append(chunk)

            if not chunks:
                print("No audio captured")
                return None

            audio = sr.AudioData(
                frame_data=b"".join(chunks),
                sample_rate=source.SAMPLE_RATE,
                sample_width=source.SAMPLE_WIDTH,
            )

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

    # Sentinel value returned when the user asks what's on the table
    DESCRIBE_TABLE = "DESCRIBE_TABLE"

    def extract_object_name(self, text: str) -> Optional[str]:
        """
        Use Claude to extract the object name from a natural language command.

        If the user is saying goodbye / ending the session, returns
        NLPVoiceController.GOODBYE instead of an object name.

        If the user says "clean up" (or similar), returns
        NLPVoiceController.CLEANUP instead of an object name.

        If the user asks what's on the table, returns
        NLPVoiceController.DESCRIBE_TABLE instead of an object name.

        Args:
            text: Transcribed voice command

        Returns:
            The extracted object name in lowercase, "GOODBYE", "CLEANUP",
            "DESCRIBE_TABLE", or None.
        """
        text_lower = text.lower()

        # Quick check for cleanup intent before hitting Claude
        if "clean up" in text_lower or "cleanup" in text_lower or "clean it up" in text_lower:
            print(f"Detected cleanup intent in: '{text}'")
            return self.CLEANUP

        # Quick check for describe-table intent
        describe_phrases = [
            "what's on the table",
            "whats on the table",
            "what is on the table",
            "tell me what's on the table",
            "what do you see",
            "what can you see",
            "what objects",
            "what items",
            "describe the table",
            "scan the table",
            "scan the entire table",
            "objects on this table",
            "objects on the table",
            "everything on the table",
            "all the objects",
            "all objects",
        ]
        if any(phrase in text_lower for phrase in describe_phrases):
            print(f"Detected describe-table intent in: '{text}'")
            return self.DESCRIBE_TABLE

        system_prompt = (
            "You are a voice command parser for a robotic object retrieval system. "
            "The user speaks a command about objects on a table.\n\n"
            "RULES:\n"
            "- If the user is asking for an object, return ONLY the object name "
            "in lowercase. No quotes, no punctuation, no explanation.\n"
            "- If the user asks what is on the table, asks to scan/list all objects, "
            "or asks what you can see, return exactly: DESCRIBE_TABLE\n"
            "- If the user asks to clean up/tidy the table, return exactly: CLEANUP\n"
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
            "  'get grandma the red pill bottle' -> red pill bottle\n"
            "  'what are all the objects on this table' -> DESCRIBE_TABLE\n"
            "  'scan the entire table' -> DESCRIBE_TABLE\n"
            "  'what can you see right now' -> DESCRIBE_TABLE\n"
            "  'clean up this table' -> CLEANUP\n"
            "  'goodbye' -> GOODBYE\n"
            "  'thanks Jarvis, that will be all' -> GOODBYE\n"
            "  'see you later' -> GOODBYE\n"
            "  'bye bye' -> GOODBYE\n"
            "  'thank you, I'm done' -> GOODBYE\n"
        )

        try:
            start = time.time()
            response = self._anthropic_client.messages.create(
                model=ANTHROPIC.model_name,
                max_tokens=64,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"User command: '{text}'"},
                ],
            )
            elapsed = time.time() - start

            raw = response.content[0].text.strip()
            cleaned = raw.strip("'\".,!?:;").strip()

            # Check sentinel intents
            if cleaned.upper() == "DESCRIBE_TABLE":
                print(f"Claude detected describe-table intent ({elapsed:.2f}s)")
                return self.DESCRIBE_TABLE
            if cleaned.upper() == "CLEANUP":
                print(f"Claude detected cleanup intent ({elapsed:.2f}s)")
                return self.CLEANUP
            if cleaned.upper() == "GOODBYE":
                print(f"Claude detected goodbye ({elapsed:.2f}s)")
                return self.GOODBYE

            object_name = cleaned.lower()
            print(f"Claude extracted: '{object_name}' ({elapsed:.2f}s)")
            return object_name if object_name and object_name != "unknown" else None

        except Exception as e:
            print(f"Claude error: {e}")
            return None

    def scan_table_objects(self, frame) -> list[str]:
        """
        Use OpenAI Vision to list all objects visible on the table.

        Sends a camera frame (as base64 JPEG) to the OpenAI Chat Completions
        API and asks it to return a JSON array of object names.  The table
        itself and permanent fixtures are excluded.

        Args:
            frame: A BGR numpy array (OpenCV frame) from the overhead camera.

        Returns:
            A list of object name strings (lowercased), e.g.
            ["sunglasses case", "cup", "remote control", "keys"].
            Returns an empty list on failure.
        """
        import base64
        import json

        import cv2

        if not self._openai_client:
            print("scan_table_objects: OpenAI client not available (no OPENAI_API_KEY)")
            return []

        # Encode frame as JPEG bytes, then base64
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            print("scan_table_objects: failed to encode frame")
            return []
        base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

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
            response = self._openai_client.chat.completions.create(
                model=OPENAI_VISION.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }],
                max_tokens=512,
            )
            elapsed = time.time() - start

            raw = response.choices[0].message.content.strip()
            print(f"OpenAI Vision scan ({elapsed:.2f}s): {raw}")

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

    @staticmethod
    def objects_to_sentence(objects: list[str]) -> str:
        """
        Turn a list of object names into a natural English sentence.

        Examples:
            [] -> "I don't see any objects on the table."
            ["cup"] -> "I can see a cup on the table."
            ["cup", "keys"] -> "I can see a cup and keys on the table."
            ["cup", "keys", "phone"] -> "I can see a cup, keys, and a phone on the table."
        """
        if not objects:
            return "I don't see any objects on the table."

        if len(objects) == 1:
            return f"I can see a {objects[0]} on the table."

        if len(objects) == 2:
            return f"I can see a {objects[0]} and a {objects[1]} on the table."

        # Oxford comma for 3+
        listed = ", a ".join(objects[:-1])
        return f"I can see a {listed}, and a {objects[-1]} on the table."

    def converse(self, text: str) -> Optional[str]:
        """
        Send a message to Claude in conversation mode and get a spoken response.

        Uses a growing message history so Claude remembers context.

        Args:
            text: The user's transcribed speech.

        Returns:
            Claude's response text, or None on error.
        """
        try:
            # Append user turn to conversation history
            self._conversation_history.append({"role": "user", "content": text})

            start = time.time()
            response = self._anthropic_client.messages.create(
                model=ANTHROPIC.model_name,
                max_tokens=256,
                system=(
                    "You are Jarvis, the friendly AI assistant for the Magic Table — "
                    "a robotic system that retrieves objects from a tabletop for users. "
                    "You are warm, concise, and helpful. Keep your responses to 1-3 "
                    "sentences since they will be spoken aloud. Be conversational and "
                    "natural, like a helpful butler."
                ),
                messages=self._conversation_history,
            )
            elapsed = time.time() - start

            reply = response.content[0].text.strip()

            # Append assistant turn so future calls have full context
            self._conversation_history.append({"role": "assistant", "content": reply})

            print(f"Claude replied ({elapsed:.2f}s): '{reply}'")
            return reply

        except Exception as e:
            print(f"Claude conversation error: {e}")
            # Remove the failed user turn so history stays consistent
            if self._conversation_history and self._conversation_history[-1]["role"] == "user":
                self._conversation_history.pop()
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

    def listen_and_converse(self, hold_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Full conversation pipeline: record → transcribe → Claude chat → ElevenLabs TTS.

        Returns:
            Claude's response text, or None if any step failed.
        """
        text = self.record_and_transcribe(hold_check=hold_check)
        if not text:
            return None

        reply = self.converse(text)
        if reply:
            self.speak(reply)
        return reply

    def listen_and_extract(self, hold_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Full pipeline: record -> transcribe -> extract object name via Claude.

        Returns:
            The object name string, or None if any step failed.
        """
        text = self.record_and_transcribe(hold_check=hold_check)
        if not text:
            return None

        return self.extract_object_name(text)
