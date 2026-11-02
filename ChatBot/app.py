import json
import logging
import os
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 8_000
API_URL = "https://router.huggingface.co/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ProviderError(RuntimeError):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def parse_messages(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("Send a messages array.")

    messages = payload["messages"]
    if not messages or len(messages) > MAX_MESSAGES:
        raise ValueError(f"Send between 1 and {MAX_MESSAGES} messages.")

    validated = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        text = message.get("content")
        if role not in ("user", "assistant") or not isinstance(text, str):
            raise ValueError("Messages need a user or assistant role and text content.")
        if index and role == validated[-1]["role"]:
            raise ValueError("Conversation roles must alternate between user and assistant.")
        text = text.strip()
        if not text or len(text) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Messages must contain 1 to {MAX_MESSAGE_LENGTH} characters.")
        validated.append({"role": role, "content": text})

    if validated[-1]["role"] != "user":
        raise ValueError("The last message must be from the user.")
    return validated


def is_local_development_origin(origin):
    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and hostname in ("localhost", "127.0.0.1", "::1")
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


def generate_reply(messages, api_key):
    model = MODEL.strip()
    if not model or len(model) > 200 or any(character.isspace() for character in model):
        raise RuntimeError("The configured Hugging Face model name is invalid.")

    body = {"model": model, "messages": messages, "max_tokens": 1024}
    request = Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        logging.warning("Hugging Face request failed with HTTP %s.", error.code)
        if error.code == 402:
            raise ProviderError(
                "Hugging Face returned HTTP 402 (payment required). Your token reached the service, "
                "but this account has no available Inference Providers credits. Check "
                "https://huggingface.co/settings/billing and "
                "https://huggingface.co/settings/inference-providers/overview. Add credits, use eligible "
                "monthly credits, or configure a provider key in Hugging Face settings.",
                status_code=402,
            ) from error
        if error.code == 429:
            raise RuntimeError("Hugging Face is rate limited right now. Please try again shortly.") from error
        if error.code == 401:
            raise RuntimeError(
                "Hugging Face rejected the token (HTTP 401). Create a token with Inference Providers "
                "permission, set HF_TOKEN, and restart the server."
            ) from error
        if error.code == 403:
            raise RuntimeError(
                "Hugging Face denied access (HTTP 403). Check the token's Inference Providers permission, "
                "provider availability, and account billing or credits."
            ) from error
        if error.code == 404:
            raise RuntimeError(
                f"Hugging Face could not find model '{model}' (HTTP 404). Set HF_MODEL to a model "
                "available through Inference Providers and restart the server."
            ) from error
        if error.code == 405:
            raise RuntimeError(
                "Hugging Face rejected the chat-completions method (HTTP 405). Check that the configured "
                "model is available for chat completion through Inference Providers."
            ) from error
        raise RuntimeError("Hugging Face could not complete that request. Check the model and token configuration.") from error
    except (URLError, TimeoutError) as error:
        logging.warning("Hugging Face request could not reach the service: %s", error)
        raise RuntimeError("Could not reach Hugging Face. Check your connection and try again.") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        logging.error("Hugging Face returned an unreadable response.")
        raise RuntimeError("Hugging Face returned an unreadable response. Please try again.") from error
    except (HTTPException, OSError) as error:
        logging.warning("Hugging Face returned an incomplete response: %s", error)
        raise RuntimeError("Hugging Face returned an incomplete response. Please try again.") from error

    try:
        reply = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        logging.warning("Hugging Face response did not contain a chat completion.")
        raise RuntimeError("Hugging Face did not return a chat response. Please try again.") from error
    if isinstance(reply, list):
        reply = "".join(
            part["text"] for part in reply
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(reply, str):
        reply = ""
    reply = reply.strip()
    if not reply:
        raise RuntimeError("Hugging Face returned an empty response. Please try again.")
    return reply


class ChatHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self, preflight=False):
        origin = self.headers.get("Origin", "")
        self.send_header("Vary", "Origin")
        if not is_local_development_origin(origin):
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        if preflight:
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")

    def do_OPTIONS(self):
        if self.path not in ("/api/chat", "/api/health"):
            self.send_error(404)
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_cors_headers(preflight=True)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            configured = bool(os.environ.get("HF_TOKEN"))
            self.send_json(200, {"ok": True, "configured": configured})
            return
        if self.path not in ("/", "/Index.html"):
            self.send_error(404)
            return
        try:
            page = (ROOT / "Index.html").read_bytes()
        except OSError:
            logging.exception("Could not read the chat interface.")
            self.send_json(500, {"error": "The chat interface could not be loaded."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = self.headers.get("Content-Length")
        if not length:
            self.send_json(400, {"error": "Request body is required."})
            return
        try:
            body_length = int(length)
        except ValueError:
            self.send_json(400, {"error": "Invalid request length."})
            return
        if body_length < 1 or body_length > MAX_BODY_BYTES:
            self.send_json(413, {"error": "The request is too large."})
            return

        try:
            payload = json.loads(self.rfile.read(body_length).decode("utf-8"))
            messages = parse_messages(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.send_json(400, {"error": str(error)})
            return

        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            self.send_json(503, {"error": "The server is missing HF_TOKEN. Configure it and restart the server."})
            return
        try:
            reply = generate_reply(messages, api_key)
        except ProviderError as error:
            self.send_json(error.status_code, {"error": str(error)})
            return
        except RuntimeError as error:
            self.send_json(502, {"error": str(error)})
            return
        except Exception:
            logging.exception("Unexpected error while generating a Hugging Face response.")
            self.send_json(502, {"error": "The chatbot server encountered an unexpected error. Check its terminal output and try again."})
            return
        self.send_json(200, {"reply": reply})

    def send_json(self, status, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_string, *args):
        logging.info("%s - %s", self.address_string(), format_string % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    logging.info("Orbit is serving at http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down Orbit.")
    finally:
        server.server_close()
