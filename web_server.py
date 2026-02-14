"""
Web Server - REST API for remote command control of the Magic Table.

Runs alongside the main application and accepts text commands
from the Next.js web interface (or any HTTP client).

Endpoints:
  POST /api/command  - Process a text command (same as voice pipeline)
  POST /api/chat     - Conversational chat with Jarvis
  GET  /api/status   - System status

Usage:
  # Standalone (no hardware, NLP only):
  python web_server.py

  # Integrated with main.py:
  python main.py --web
"""

import os
import sys
import threading
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Load .env so API keys are available
# ---------------------------------------------------------------------------
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=True)
except ImportError:
    pass

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# References to live system components (set via create_app or start_web_server)
# ---------------------------------------------------------------------------
_magic_table = None          # MagicTable instance (full system)
_nlp_controller = None       # NLPVoiceController (Gemini NLP)

# Chat history for web sessions (simple in-memory list)
_chat_history: list[dict] = []


def create_app(magic_table=None):
    """
    Configure the Flask app with a reference to the running MagicTable.

    If magic_table is None the server starts in "standalone" mode —
    it can still parse commands via Gemini but won't drive hardware.
    """
    global _magic_table, _nlp_controller

    _magic_table = magic_table

    if magic_table and magic_table.nlp_voice:
        _nlp_controller = magic_table.nlp_voice
    elif _nlp_controller is None:
        # Standalone mode — spin up an NLPVoiceController just for Gemini
        try:
            from nlp_voice import NLPVoiceController
            _nlp_controller = NLPVoiceController()
            print("NLP controller initialized (standalone mode)")
        except Exception as e:
            print(f"WARNING: Could not initialise NLP controller: {e}")

    return app


# ── Health / Status ──────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def get_status():
    """Return current system status."""
    status = {
        "online": True,
        "nlp_available": _nlp_controller is not None,
        "motor_connected": bool(
            _magic_table
            and getattr(_magic_table, "motor", None)
            and _magic_table.motor.connected
        ),
        "tracker_active": bool(
            _magic_table and getattr(_magic_table, "tracker", None)
        ),
    }
    if _magic_table:
        status["status_message"] = getattr(_magic_table, "_status_message", "")
    return jsonify(status)


# ── Command processing (mirrors voice_detect_pipeline) ──────────────────────

@app.route("/api/command", methods=["POST"])
def handle_command():
    """
    Process a text command — the same pipeline as pressing SPACE + speaking.

    Request:  {"text": "fetch the red bottle"}
    Response: {"status": "...", "object": "...", ...}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty command"}), 400

    # Store user message
    _chat_history.append({"role": "user", "text": text})

    if not _nlp_controller:
        msg = "NLP controller not initialised — check GEMINI_API_KEY"
        _chat_history.append({"role": "assistant", "text": msg})
        return jsonify({"error": msg}), 503

    # ── 1. Extract object name via Gemini ────────────────────────────────
    result = _nlp_controller.extract_object_name(text)

    # ── 2. Handle goodbye ────────────────────────────────────────────────
    from nlp_voice import NLPVoiceController

    if result == NLPVoiceController.GOODBYE:
        reply = "Goodbye! Happy to help anytime."
        _chat_history.append({"role": "assistant", "text": reply})
        return jsonify({"status": "goodbye", "reply": reply})

    # ── 3. No object extracted → fall back to conversation ───────────────
    if not result:
        reply = _nlp_controller.converse(text)
        if reply:
            _chat_history.append({"role": "assistant", "text": reply})
            return jsonify({"status": "conversation", "reply": reply})
        msg = "I didn't quite understand that. Could you rephrase?"
        _chat_history.append({"role": "assistant", "text": msg})
        return jsonify({"status": "error", "reply": msg})

    # ── 4. Object recognised — run detection + fetch ─────────────────────
    object_name = result
    response_data: dict = {
        "status": "success",
        "object": object_name,
        "action": "fetch",
    }

    # If the full system is running, do VLM detection + gantry move
    if _magic_table and _magic_table.tracker:
        _magic_table._status_message = f"[WEB] Scanning for '{object_name}'..."

        detected = _magic_table.tracker.run_detection(target_label=object_name)

        if detected:
            obj = detected[0]
            px, py = obj.center_x, obj.center_y
            phys_x, phys_y = _magic_table._pixel_to_physical(px, py)

            response_data.update({
                "found": True,
                "pixel": {"x": round(px, 1), "y": round(py, 1)},
                "physical_mm": {"x": round(phys_x, 1), "y": round(phys_y, 1)},
                "confidence": round(obj.confidence, 2),
            })

            # Drive the gantry if motor is connected
            if _magic_table.motor and _magic_table.motor.connected:
                from config import GANTRY

                success = _magic_table.motor.drag_object_to(
                    from_x=phys_x,
                    from_y=phys_y,
                    to_x=GANTRY.pickup_x_mm,
                    to_y=GANTRY.pickup_y_mm,
                )
                response_data["delivered"] = success

            reply = (
                f"Found '{object_name}' at ({phys_x:.0f}, {phys_y:.0f}) mm "
                f"with {obj.confidence:.0%} confidence."
            )
            if response_data.get("delivered"):
                reply += " Delivered to pickup zone!"
        else:
            response_data["found"] = False
            reply = f"I looked for '{object_name}' but couldn't find it on the table."
    else:
        # No tracker — just acknowledge the parsed command
        reply = (
            f"Got it — you want '{object_name}'. "
            "The hardware isn't connected right now, but the command was understood."
        )
        response_data["found"] = None

    response_data["reply"] = reply
    _chat_history.append({"role": "assistant", "text": reply})
    return jsonify(response_data)


# ── Conversation mode ────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def handle_chat():
    """
    Pure conversation — chat with Jarvis (no object detection).

    Request:  {"text": "How are you?"}
    Response: {"reply": "I'm doing great! ..."}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400

    if not _nlp_controller:
        return jsonify({"error": "NLP controller not initialised"}), 503

    _chat_history.append({"role": "user", "text": text})

    reply = _nlp_controller.converse(text)
    if reply:
        _chat_history.append({"role": "assistant", "text": reply})
        return jsonify({"status": "success", "reply": reply})

    msg = "Sorry, I couldn't come up with a response."
    _chat_history.append({"role": "assistant", "text": msg})
    return jsonify({"error": msg}), 500


# ── Chat history ─────────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the chat history for the current session."""
    return jsonify({"history": _chat_history})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear chat history."""
    _chat_history.clear()
    return jsonify({"status": "cleared"})


# ── Server launcher ─────────────────────────────────────────────────────────

def start_web_server(magic_table=None, host="0.0.0.0", port=5050):
    """
    Start the web server in a daemon thread so it runs alongside main.py.

    Returns the thread object.
    """
    create_app(magic_table)

    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="web-server")
    thread.start()

    print(f"\n{'=' * 50}")
    print(f"  Web API server running on http://{host}:{port}")
    print(f"    POST /api/command  — send text commands")
    print(f"    POST /api/chat     — chat with Jarvis")
    print(f"    GET  /api/status   — system status")
    print(f"{'=' * 50}\n")

    return thread


# ── Standalone entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Magic Table Web API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    create_app()  # standalone — no MagicTable
    print("Starting web server in standalone mode (no hardware)...")
    app.run(host=args.host, port=args.port, debug=True)
