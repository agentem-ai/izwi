#!/usr/bin/env python3
"""
Persistent Qwen3-ASR Daemon for speech-to-text transcription.
Supports Qwen3-ASR-0.6B and Qwen3-ASR-1.7B models via Unix socket.
"""

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import io
import json
import signal
import socket
import struct
import threading
import time
import traceback
import base64
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import OrderedDict
import urllib.request

import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

DEFAULT_SOCKET_PATH = "/tmp/izwi_qwen3_asr_daemon.sock"
DEFAULT_MODEL_06B = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_MODEL_17B = "Qwen/Qwen3-ASR-1.7B"


class ASRModelCache:
    """Cache for loaded Qwen3-ASR models."""

    def __init__(self, max_size: int = 1):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, model_id: str):
        """Get model from cache."""
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return self.cache[model_id]
            return None

    def put(self, model_id: str, model_data: dict):
        """Add model to cache."""
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return

            while len(self.cache) >= self.max_size:
                evicted_id, _ = self.cache.popitem(last=False)
                print(f"[ASR Daemon] Evicting model: {evicted_id}", file=sys.stderr)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

            self.cache[model_id] = model_data

    def remove(self, model_id: str) -> bool:
        """Remove model from cache."""
        with self.lock:
            if model_id in self.cache:
                del self.cache[model_id]
                return True
            return False

    def clear(self):
        """Clear all cached models."""
        with self.lock:
            self.cache.clear()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    def list_models(self) -> list:
        """List cached model IDs."""
        with self.lock:
            return list(self.cache.keys())


class Qwen3ASRDaemon:
    """Qwen3-ASR Daemon that handles requests via Unix socket."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.model_cache = ASRModelCache()
        self.running = False
        self.server_socket = None
        self.device = None
        self.dtype = None
        self._init_device()

    def _init_device(self):
        """Initialize device settings."""
        try:
            import torch

            if torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float32
            elif torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            print(f"[ASR Daemon] Using device: {self.device}", file=sys.stderr)
        except ImportError:
            self.device = "cpu"
            self.dtype = None

    def _get_local_model_path(self, model_id: str) -> Optional[Path]:
        """Get local model path if downloaded."""
        model_name = model_id.split("/")[-1]
        possible_paths = [
            Path.home()
            / "Library"
            / "Application Support"
            / "izwi"
            / "models"
            / model_name,
            Path.home() / ".cache" / "izwi" / "models" / model_name,
            Path.home() / ".local" / "share" / "izwi" / "models" / model_name,
        ]

        for path in possible_paths:
            if path.exists():
                if (path / "config.json").exists() and (
                    (path / "model.safetensors").exists()
                    or (path / "model-00001-of-00002.safetensors").exists()
                ):
                    # Ensure chat_template.json exists (required for transcription)
                    self._ensure_chat_template(path, model_id)
                    return path
        return None

    def _ensure_chat_template(self, local_path: Path, model_id: str):
        """Ensure chat_template.json exists in local model path."""
        chat_template_path = local_path / "chat_template.json"
        if chat_template_path.exists():
            return

        # Download chat_template.json from HuggingFace
        url = f"https://huggingface.co/{model_id}/raw/main/chat_template.json"
        print(
            f"[ASR Daemon] Downloading chat_template.json from {url}", file=sys.stderr
        )
        try:
            urllib.request.urlretrieve(url, chat_template_path)
            print(
                f"[ASR Daemon] Downloaded chat_template.json to {chat_template_path}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[ASR Daemon] Warning: Failed to download chat_template.json: {e}",
                file=sys.stderr,
            )

    def _load_model(self, model_id: str = DEFAULT_MODEL_06B) -> dict:
        """Load Qwen3-ASR model."""
        cached = self.model_cache.get(model_id)
        if cached is not None:
            print(f"[ASR Daemon] Using cached model: {model_id}", file=sys.stderr)
            return cached

        local_path = self._get_local_model_path(model_id)
        load_path = str(local_path) if local_path else model_id

        print(f"[ASR Daemon] Loading model: {load_path}", file=sys.stderr)
        start_time = time.time()

        import torch
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            load_path,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

        model_data = {
            "model": model,
            "model_id": model_id,
        }

        load_time = time.time() - start_time
        print(f"[ASR Daemon] Model loaded in {load_time:.2f}s", file=sys.stderr)

        self.model_cache.put(model_id, model_data)
        return model_data

    def _handle_check(self, request: dict) -> dict:
        """Handle dependency check request."""
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            return {"status": "ok", "device": self.device}
        except ImportError as e:
            return {"error": f"Missing dependency: {str(e)}"}

    def _handle_status(self, request: dict) -> dict:
        """Handle status request."""
        return {
            "status": "ok",
            "device": self.device,
            "cached_models": self.model_cache.list_models(),
            "pid": os.getpid(),
        }

    def _handle_preload(self, request: dict) -> dict:
        """Handle model preload request."""
        model_id = request.get("model_id", DEFAULT_MODEL_06B)

        try:
            self._load_model(model_id)
            return {"status": "ok", "model_id": model_id}
        except Exception as e:
            return {"error": f"Failed to preload model: {str(e)}"}

    def _handle_unload(self, request: dict) -> dict:
        """Handle model unload request."""
        model_id = request.get("model_id", "")
        if model_id:
            if self.model_cache.remove(model_id):
                return {"status": "ok", "unloaded": model_id}
            return {"error": f"Model not loaded: {model_id}"}
        else:
            self.model_cache.clear()
            return {"status": "ok", "unloaded": "all"}

    def _handle_transcribe(self, request: dict) -> dict:
        """Handle transcription request."""
        import torch

        audio_b64 = request.get("audio_base64", "")
        model_id = request.get("model_id", DEFAULT_MODEL_06B)
        language = request.get("language", None)

        if not audio_b64:
            return {"error": "No audio provided"}

        try:
            model_data = self._load_model(model_id)
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

        model = model_data["model"]

        audio_path = self._decode_audio_to_file(audio_b64)
        if audio_path is None:
            return {"error": "Could not decode audio"}

        try:
            results = model.transcribe(
                audio=audio_path,
                language=language,
            )

            if results and len(results) > 0:
                result = results[0]
                response = {
                    "transcription": result.text,
                    "language": (
                        result.language if hasattr(result, "language") else None
                    ),
                }
            else:
                response = {"transcription": "", "language": None}

            return response

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": f"Transcription failed: {str(e)}"}
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    def _decode_audio_to_file(self, audio_b64: str) -> Optional[str]:
        """Decode audio from base64 and save to temp file."""
        audio_bytes = base64.b64decode(audio_b64)

        for ext in [".wav", ".webm", ".mp3", ".ogg", ".m4a", ".flac"]:
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                import soundfile as sf

                try:
                    data, sr = sf.read(temp_path)
                    return temp_path
                except Exception:
                    pass

                try:
                    import torchaudio

                    wav, sr = torchaudio.load(temp_path)
                    return temp_path
                except Exception:
                    pass

                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                return f.name
        except Exception:
            return None

    def handle_request(self, request: dict) -> dict:
        """Route request to appropriate handler."""
        command = request.get("command", "transcribe")

        handlers = {
            "check": self._handle_check,
            "status": self._handle_status,
            "preload": self._handle_preload,
            "unload": self._handle_unload,
            "transcribe": self._handle_transcribe,
            "shutdown": lambda r: {"status": "shutdown"},
        }

        handler = handlers.get(command)
        if handler:
            return handler(request)
        return {"error": f"Unknown command: {command}"}

    def _recv_message(self, conn: socket.socket) -> Optional[dict]:
        """Receive length-prefixed JSON message."""
        try:
            length_data = conn.recv(4)
            if not length_data:
                return None
            length = struct.unpack(">I", length_data)[0]

            data = b""
            while len(data) < length:
                chunk = conn.recv(min(length - len(data), 65536))
                if not chunk:
                    return None
                data += chunk

            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"[ASR Daemon] Error receiving message: {e}", file=sys.stderr)
            return None

    def _send_message(self, conn: socket.socket, message: dict):
        """Send length-prefixed JSON message."""
        try:
            data = json.dumps(message).encode("utf-8")
            length = struct.pack(">I", len(data))
            conn.sendall(length + data)
        except Exception as e:
            print(f"[ASR Daemon] Error sending message: {e}", file=sys.stderr)

    def _handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection."""
        try:
            while self.running:
                request = self._recv_message(conn)
                if request is None:
                    break

                if request.get("command") == "shutdown":
                    self._send_message(conn, {"status": "shutdown"})
                    self.running = False
                    break

                try:
                    response = self.handle_request(request)
                except Exception as e:
                    response = {"error": f"Internal error: {str(e)}"}
                    traceback.print_exc(file=sys.stderr)

                self._send_message(conn, response)
        finally:
            conn.close()

    def start(self):
        """Start the daemon server."""
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)

        os.chmod(self.socket_path, 0o600)

        self.running = True
        print(
            f"[ASR Daemon] Started on {self.socket_path} (PID: {os.getpid()})",
            file=sys.stderr,
        )

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                thread = threading.Thread(target=self._handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[ASR Daemon] Accept error: {e}", file=sys.stderr)

        self.stop()

    def stop(self):
        """Stop the daemon server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.model_cache.clear()
        print("[ASR Daemon] Stopped", file=sys.stderr)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(
            f"[ASR Daemon] Received signal {signum}, shutting down...", file=sys.stderr
        )
        self.running = False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-ASR Daemon")
    parser.add_argument(
        "--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path"
    )
    parser.add_argument(
        "--preload", action="store_true", help="Preload model on startup"
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_06B,
        help="HuggingFace model ID to use (Qwen/Qwen3-ASR-0.6B or Qwen/Qwen3-ASR-1.7B)",
    )
    args = parser.parse_args()

    daemon = Qwen3ASRDaemon(socket_path=args.socket)

    if args.preload:
        print(f"[ASR Daemon] Preloading model: {args.model_id}", file=sys.stderr)
        try:
            daemon._load_model(args.model_id)
        except Exception as e:
            print(f"[ASR Daemon] Preload failed: {e}", file=sys.stderr)

    daemon.start()


if __name__ == "__main__":
    main()
