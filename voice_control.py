"""
Voice Control - Speech recognition for object retrieval commands.

This module handles:
- Continuous voice listening
- Command parsing ("bring me the keys", "get my phone", etc.)
- Object name extraction and matching

Supported command patterns:
- "bring me the [object]"
- "get me the [object]"
- "fetch the [object]"
- "grab the [object]"
- "I need the [object]"
- "where is the [object]" (finds and highlights, doesn't move)
- Just saying the object name also works
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import speech_recognition as sr

from config import VOICE, DETECTION


class CommandType(Enum):
    """Types of voice commands."""
    FETCH = "fetch"      # Bring object to pickup zone
    LOCATE = "locate"    # Just highlight/identify where object is
    STOP = "stop"        # Emergency stop
    HOME = "home"        # Return to home position
    UNKNOWN = "unknown"


@dataclass
class VoiceCommand:
    """Parsed voice command."""
    command_type: CommandType
    object_name: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0


class VoiceController:
    """
    Voice command recognition and parsing.
    
    Usage:
        voice = VoiceController(known_objects=["keys", "phone", "wallet"])
        voice.start_listening()
        
        while True:
            cmd = voice.get_command(timeout=0.1)
            if cmd:
                print(f"Command: {cmd.command_type}, Object: {cmd.object_name}")
    """
    
    # Command trigger phrases
    FETCH_TRIGGERS = [
        "bring me", "bring", "get me", "get", "fetch", "grab",
        "i need", "need", "give me", "pass me", "hand me",
        "can you get", "could you get", "please get",
        "retrieve", "pull"
    ]
    
    LOCATE_TRIGGERS = [
        "where is", "where's", "find", "locate", "show me",
        "point to", "highlight"
    ]
    
    STOP_TRIGGERS = [
        "stop", "halt", "freeze", "emergency", "abort", "cancel"
    ]
    
    HOME_TRIGGERS = [
        "go home", "return home", "home position", "reset position"
    ]
    
    def __init__(self, known_objects: List[str] = None):
        """
        Initialize voice controller.
        
        Args:
            known_objects: List of object names to recognize
        """
        self.known_objects = known_objects or DETECTION.prompts
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Command queue
        self._command_queue: queue.Queue[VoiceCommand] = queue.Queue()
        
        # Threading
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_command: Optional[Callable[[VoiceCommand], None]] = None
        self.on_listening_start: Optional[Callable[[], None]] = None
        self.on_listening_stop: Optional[Callable[[], None]] = None
        self.on_speech_detected: Optional[Callable[[str], None]] = None
        
        # Configure recognizer
        self.recognizer.energy_threshold = VOICE.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def calibrate_for_ambient_noise(self, duration: float = 2.0):
        """Calibrate microphone for ambient noise."""
        print(f"Calibrating for ambient noise ({duration}s)...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print(f"Energy threshold set to: {self.recognizer.energy_threshold}")
    
    def _parse_command(self, text: str) -> VoiceCommand:
        """Parse raw text into a structured command."""
        text_lower = text.lower().strip()
        
        # Check for stop command first (highest priority)
        for trigger in self.STOP_TRIGGERS:
            if trigger in text_lower:
                return VoiceCommand(
                    command_type=CommandType.STOP,
                    raw_text=text,
                    confidence=1.0
                )
        
        # Check for home command
        for trigger in self.HOME_TRIGGERS:
            if trigger in text_lower:
                return VoiceCommand(
                    command_type=CommandType.HOME,
                    raw_text=text,
                    confidence=1.0
                )
        
        # Check for locate command
        for trigger in self.LOCATE_TRIGGERS:
            if trigger in text_lower:
                obj = self._extract_object(text_lower, trigger)
                if obj:
                    return VoiceCommand(
                        command_type=CommandType.LOCATE,
                        object_name=obj,
                        raw_text=text,
                        confidence=0.9
                    )
        
        # Check for fetch command
        for trigger in self.FETCH_TRIGGERS:
            if trigger in text_lower:
                obj = self._extract_object(text_lower, trigger)
                if obj:
                    return VoiceCommand(
                        command_type=CommandType.FETCH,
                        object_name=obj,
                        raw_text=text,
                        confidence=0.9
                    )
        
        # Check if just the object name was said
        obj = self._find_object_in_text(text_lower)
        if obj:
            return VoiceCommand(
                command_type=CommandType.FETCH,
                object_name=obj,
                raw_text=text,
                confidence=0.7
            )
        
        return VoiceCommand(
            command_type=CommandType.UNKNOWN,
            raw_text=text,
            confidence=0.0
        )
    
    def _extract_object(self, text: str, trigger: str) -> Optional[str]:
        """Extract object name from text after removing trigger phrase."""
        # Remove trigger phrase
        idx = text.find(trigger)
        if idx >= 0:
            remaining = text[idx + len(trigger):].strip()
            # Remove common filler words
            for filler in ["the", "my", "a", "an", "some"]:
                if remaining.startswith(filler + " "):
                    remaining = remaining[len(filler) + 1:]
            return self._find_object_in_text(remaining)
        return None
    
    def _find_object_in_text(self, text: str) -> Optional[str]:
        """Find a known object name in text."""
        text_lower = text.lower()
        
        # Direct match
        for obj in self.known_objects:
            if obj.lower() in text_lower:
                return obj
        
        # Fuzzy matching for common misheard words
        fuzzy_map = {
            "key": "keys",
            "airpod": "airpods",
            "air pods": "airpods",
            "air pod": "airpods",
            "pills": "pill bottle",
            "medication": "pill bottle",
            "medicine": "pill bottle",
            "spectacles": "glasses",
            "eyeglasses": "glasses",
            "sunglasses": "glasses",
            "cell phone": "phone",
            "cellphone": "phone",
            "mobile": "phone",
            "mobile phone": "phone",
            "billfold": "wallet",
        }
        
        for misheard, correct in fuzzy_map.items():
            if misheard in text_lower and correct in self.known_objects:
                return correct
        
        return None
    
    def _listen_loop(self):
        """Background listening loop."""
        with self.microphone as source:
            while self._listening:
                try:
                    if self.on_listening_start:
                        self.on_listening_start()
                    
                    # Listen for audio
                    audio = self.recognizer.listen(
                        source,
                        timeout=VOICE.listen_timeout,
                        phrase_time_limit=VOICE.phrase_time_limit
                    )
                    
                    if self.on_listening_stop:
                        self.on_listening_stop()
                    
                    # Recognize speech
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"Heard: '{text}'")
                        
                        if self.on_speech_detected:
                            self.on_speech_detected(text)
                        
                        # Parse and queue command
                        command = self._parse_command(text)
                        if command.command_type != CommandType.UNKNOWN:
                            self._command_queue.put(command)
                            if self.on_command:
                                self.on_command(command)
                        else:
                            print(f"Could not understand command: '{text}'")
                            
                    except sr.UnknownValueError:
                        # Speech was not understood
                        pass
                    except sr.RequestError as e:
                        print(f"Speech recognition service error: {e}")
                        
                except sr.WaitTimeoutError:
                    # No speech detected within timeout, continue listening
                    pass
                except Exception as e:
                    print(f"Listening error: {e}")
                    time.sleep(0.5)
    
    def start_listening(self):
        """Start background voice listening."""
        if self._listening:
            return
        
        print("Starting voice recognition...")
        self.calibrate_for_ambient_noise()
        
        self._listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        print(f"Listening for commands. Known objects: {self.known_objects}")
        print("Say things like:")
        print('  - "bring me the keys"')
        print('  - "get my phone"')
        print('  - "where is the wallet"')
        print('  - "stop"')
    
    def stop_listening(self):
        """Stop background voice listening."""
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        print("Voice recognition stopped")
    
    def get_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """
        Get the next command from the queue.
        
        Args:
            timeout: How long to wait for a command (None = non-blocking)
            
        Returns:
            VoiceCommand if available, None otherwise
        """
        try:
            return self._command_queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None
    
    def listen_once(self) -> Optional[VoiceCommand]:
        """
        Listen for a single command (blocking).
        
        Returns:
            VoiceCommand if understood, None otherwise
        """
        with self.microphone as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=VOICE.listen_timeout,
                    phrase_time_limit=VOICE.phrase_time_limit
                )
                
                text = self.recognizer.recognize_google(audio)
                print(f"Heard: '{text}'")
                return self._parse_command(text)
                
            except sr.WaitTimeoutError:
                print("No speech detected")
            except sr.UnknownValueError:
                print("Could not understand speech")
            except sr.RequestError as e:
                print(f"Speech service error: {e}")
        
        return None
    
    def __enter__(self):
        self.start_listening()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_listening()


def test_voice():
    """Interactive test of voice recognition."""
    voice = VoiceController()
    
    print("\n" + "="*50)
    print("Voice Recognition Test")
    print("="*50)
    print(f"Known objects: {voice.known_objects}")
    print("Say a command to test, or press Ctrl+C to exit\n")
    
    voice.calibrate_for_ambient_noise()
    
    while True:
        try:
            cmd = voice.listen_once()
            if cmd:
                print(f"\n  Command Type: {cmd.command_type.value}")
                print(f"  Object: {cmd.object_name}")
                print(f"  Raw: '{cmd.raw_text}'")
                print(f"  Confidence: {cmd.confidence}")
                print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def test_parsing():
    """Test command parsing without microphone."""
    voice = VoiceController()
    
    test_phrases = [
        "bring me the keys",
        "get my phone",
        "fetch the wallet",
        "where is the pill bottle",
        "find my glasses",
        "stop",
        "go home",
        "keys",
        "airpods please",
        "I need my medication",
        "can you get the remote",
    ]
    
    print("\nCommand Parsing Test:")
    print("=" * 50)
    for phrase in test_phrases:
        cmd = voice._parse_command(phrase)
        print(f"'{phrase}'")
        print(f"  -> {cmd.command_type.value}: {cmd.object_name}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "parse":
        test_parsing()
    else:
        test_voice()
