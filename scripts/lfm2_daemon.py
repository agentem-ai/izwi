#!/usr/bin/env python3
"""
Persistent LFM2-Audio Daemon for Liquid AI's LFM2-Audio model.
Supports TTS, ASR, and Audio-to-Audio chat via Unix socket.
"""

import sys
import os
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

import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

DEFAULT_SOCKET_PATH = "/tmp/izwi_lfm2_daemon.sock"
DEFAULT_HF_REPO = "LiquidAI/LFM2-Audio-1.5B"

LFM2_VOICES = {
    "us_male": "Perform TTS. Use the US male voice.",
    "us_female": "Perform TTS. Use the US female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
}

LFM2_SYSTEM_PROMPTS = {
    "tts": LFM2_VOICES,
    "asr": "Perform ASR.",
    "chat": "Respond with interleaved text and audio.",
}


class LFM2ModelCache:
    """Cache for loaded LFM2-Audio models."""

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
                print(f"[LFM2 Daemon] Evicting model: {evicted_id}", file=sys.stderr)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
            except Exception:
                pass

    def list_models(self) -> list:
        """List cached model IDs."""
        with self.lock:
            return list(self.cache.keys())


class LFM2Daemon:
    """LFM2-Audio Daemon that handles requests via Unix socket."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.model_cache = LFM2ModelCache()
        self.running = False
        self.server_socket = None
        self.device = None
        self.dtype = None
        self._init_device()

    def _init_device(self):
        """Initialize device settings."""
        try:
            import torch

            # On macOS, never use CUDA even if PyTorch was compiled with CUDA support
            # Check for MPS first (Apple Silicon), then CPU
            if torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float32
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            print(f"[LFM2 Daemon] Using device: {self.device}", file=sys.stderr)
        except ImportError:
            self.device = "cpu"
            self.dtype = None

    def _get_local_model_path(self) -> Optional[Path]:
        """Get local model path if downloaded."""
        # Check common locations for downloaded models
        possible_paths = [
            Path.home()
            / "Library"
            / "Application Support"
            / "izwi"
            / "models"
            / "LFM2-Audio-1.5B",
            Path.home() / ".cache" / "izwi" / "models" / "LFM2-Audio-1.5B",
            Path.home() / ".local" / "share" / "izwi" / "models" / "LFM2-Audio-1.5B",
        ]

        for path in possible_paths:
            if path.exists():
                # Check for essential files
                if (path / "model.safetensors").exists() and (
                    path / "config.json"
                ).exists():
                    return path
        return None

    def _setup_hf_cache_symlink(self, local_path: Path) -> bool:
        """
        Create a symlink in the HuggingFace cache to point to our local model.
        This allows liquid_audio's from_pretrained to find the model.
        """
        # HuggingFace cache structure: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dir = hf_cache_dir / "models--LiquidAI--LFM2-Audio-1.5B"
        snapshots_dir = model_cache_dir / "snapshots"

        # Create a fake snapshot directory
        snapshot_hash = "local"
        snapshot_dir = snapshots_dir / snapshot_hash

        try:
            # Create parent directories
            snapshots_dir.mkdir(parents=True, exist_ok=True)

            # Create refs/main to point to our snapshot
            refs_dir = model_cache_dir / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            refs_main = refs_dir / "main"

            # Write the snapshot hash to refs/main
            with open(refs_main, "w") as f:
                f.write(snapshot_hash)

            # Create symlink or copy files to snapshot directory
            if snapshot_dir.exists():
                if snapshot_dir.is_symlink():
                    snapshot_dir.unlink()
                elif snapshot_dir.is_dir():
                    import shutil

                    shutil.rmtree(snapshot_dir)

            # Create symlink to local model path
            snapshot_dir.symlink_to(local_path)

            print(
                f"[LFM2 Daemon] Created HF cache symlink: {snapshot_dir} -> {local_path}",
                file=sys.stderr,
            )
            return True
        except Exception as e:
            print(
                f"[LFM2 Daemon] Failed to create HF cache symlink: {e}", file=sys.stderr
            )
            return False

    def _load_model(self, model_id: str = DEFAULT_HF_REPO) -> dict:
        """Load LFM2-Audio model and processor."""
        cached = self.model_cache.get(model_id)
        if cached is not None:
            print(f"[LFM2 Daemon] Using cached model: {model_id}", file=sys.stderr)
            return cached

        # Check for local model path first
        local_path = self._get_local_model_path()

        if local_path:
            # Set up HF cache symlink so liquid_audio can find the model
            self._setup_hf_cache_symlink(local_path)
            print(f"[LFM2 Daemon] Using local model at: {local_path}", file=sys.stderr)

        # Always use the HF repo ID - the symlink will redirect to local files
        print(f"[LFM2 Daemon] Loading model: {model_id}", file=sys.stderr)
        start_time = time.time()

        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        # Load model following the official documentation
        print(f"[LFM2 Daemon] Loading processor...", file=sys.stderr)
        try:
            processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
            print(f"[LFM2 Daemon] Loading model...", file=sys.stderr)
            model = LFM2AudioModel.from_pretrained(model_id).eval()
        except Exception as e:
            if "Torch not compiled with CUDA enabled" not in str(e):
                raise
            print(
                "[LFM2 Daemon] CUDA not available; retrying model load on CPU",
                file=sys.stderr,
            )
            self.device = "cpu"
            self.dtype = torch.float32
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                torch.set_default_device("cpu")
            except Exception:
                pass
            processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
            print(f"[LFM2 Daemon] Loading model...", file=sys.stderr)
            model = LFM2AudioModel.from_pretrained(model_id).eval()

        # Move to device if not CPU
        if self.device != "cpu":
            print(f"[LFM2 Daemon] Moving model to {self.device}...", file=sys.stderr)
            model = model.to(self.device)

        model_data = {
            "model": model,
            "processor": processor,
        }

        load_time = time.time() - start_time
        print(f"[LFM2 Daemon] Model loaded in {load_time:.2f}s", file=sys.stderr)

        self.model_cache.put(model_id, model_data)
        return model_data

    def _handle_check(self, request: dict) -> dict:
        """Handle dependency check request."""
        try:
            import torch
            import torchaudio
            from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

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
            "voices": list(LFM2_VOICES.keys()),
        }

    def _handle_preload(self, request: dict) -> dict:
        """Handle model preload request."""
        model_id = request.get("model_id", DEFAULT_HF_REPO)

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

    def _handle_tts(self, request: dict) -> dict:
        """Handle TTS generation request."""
        import torch
        import torchaudio
        from liquid_audio import ChatState

        text = request.get("text", "")
        voice = request.get("voice", "us_female")
        model_id = request.get("model_id", DEFAULT_HF_REPO)
        max_new_tokens = request.get("max_new_tokens", 1024)
        audio_temperature = request.get("audio_temperature", 0.8)
        audio_top_k = request.get("audio_top_k", 64)

        if not text:
            return {"error": "No text provided"}

        if voice not in LFM2_VOICES:
            return {"error": f"Invalid voice. Choose from: {list(LFM2_VOICES.keys())}"}

        try:
            model_data = self._load_model(model_id)
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

        model = model_data["model"]
        processor = model_data["processor"]

        try:
            chat = ChatState(processor)
            chat.new_turn("system")
            chat.add_text(LFM2_VOICES[voice])
            chat.end_turn()

            chat.new_turn("user")
            chat.add_text(text)
            chat.end_turn()

            chat.new_turn("assistant")

            audio_out: List[torch.Tensor] = []
            for t in model.generate_sequential(
                **chat,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
            ):
                if t.numel() > 1:
                    audio_out.append(t)

            if len(audio_out) <= 1:
                return {"error": "No audio generated"}

            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            waveform = processor.decode(audio_codes)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                torchaudio.save(temp_path, waveform.cpu(), 24000)
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            finally:
                os.unlink(temp_path)

            return {
                "audio_base64": audio_b64,
                "sample_rate": 24000,
                "format": "wav",
            }

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": f"TTS generation failed: {str(e)}"}

    def _handle_asr(self, request: dict) -> dict:
        """Handle ASR (speech-to-text) request."""
        import torch
        import torchaudio
        from liquid_audio import ChatState

        audio_b64 = request.get("audio_base64", "")
        model_id = request.get("model_id", DEFAULT_HF_REPO)
        max_new_tokens = request.get("max_new_tokens", 512)

        if not audio_b64:
            return {"error": "No audio provided"}

        try:
            model_data = self._load_model(model_id)
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

        model = model_data["model"]
        processor = model_data["processor"]

        wav, sampling_rate = self._decode_audio(audio_b64)
        if wav is None:
            return {"error": "Could not decode audio"}

        try:
            chat = ChatState(processor)
            chat.new_turn("system")
            chat.add_text(LFM2_SYSTEM_PROMPTS["asr"])
            chat.end_turn()

            chat.new_turn("user")
            chat.add_audio(wav, sampling_rate)
            chat.end_turn()

            chat.new_turn("assistant")

            text_out = []
            for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
                if t.numel() == 1:
                    text_out.append(processor.text.decode(t))

            transcription = "".join(text_out)

            return {
                "transcription": transcription,
            }

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": f"ASR failed: {str(e)}"}

    def _handle_audio_chat(self, request: dict) -> dict:
        """Handle audio-to-audio chat request."""
        import torch
        import torchaudio
        from liquid_audio import ChatState, LFMModality

        audio_b64 = request.get("audio_base64", "")
        text_input = request.get("text", "")
        model_id = request.get("model_id", DEFAULT_HF_REPO)
        max_new_tokens = request.get("max_new_tokens", 512)
        audio_temperature = request.get("audio_temperature", 1.0)
        audio_top_k = request.get("audio_top_k", 4)

        if not audio_b64 and not text_input:
            return {"error": "No audio or text input provided"}

        try:
            model_data = self._load_model(model_id)
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

        model = model_data["model"]
        processor = model_data["processor"]

        try:
            chat = ChatState(processor)
            chat.new_turn("system")
            chat.add_text(LFM2_SYSTEM_PROMPTS["chat"])
            chat.end_turn()

            chat.new_turn("user")
            if audio_b64:
                wav, sampling_rate = self._decode_audio(audio_b64)
                if wav is None:
                    return {"error": "Could not decode audio"}
                chat.add_audio(wav, sampling_rate)
            else:
                chat.add_text(text_input)
            chat.end_turn()

            chat.new_turn("assistant")

            text_out: List[torch.Tensor] = []
            audio_out: List[torch.Tensor] = []
            modality_out: List[LFMModality] = []

            for t in model.generate_interleaved(
                **chat,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
            ):
                if t.numel() == 1:
                    text_out.append(t)
                    modality_out.append(LFMModality.TEXT)
                else:
                    audio_out.append(t)
                    modality_out.append(LFMModality.AUDIO_OUT)

            response_text = "".join([processor.text.decode(t) for t in text_out])

            audio_b64_out = None
            if len(audio_out) > 1:
                audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
                waveform = processor.decode(audio_codes)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                try:
                    torchaudio.save(temp_path, waveform.cpu(), 24000)
                    with open(temp_path, "rb") as f:
                        audio_bytes = f.read()
                    audio_b64_out = base64.b64encode(audio_bytes).decode("utf-8")
                finally:
                    os.unlink(temp_path)

            return {
                "text": response_text,
                "audio_base64": audio_b64_out,
                "sample_rate": 24000,
                "format": "wav",
            }

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": f"Audio chat failed: {str(e)}"}

    def _decode_audio(self, audio_b64: str):
        """Decode audio from base64."""
        import numpy as np
        import soundfile as sf
        import torchaudio
        import torch

        audio_bytes = base64.b64decode(audio_b64)

        try:
            wav, sr = sf.read(io.BytesIO(audio_bytes))
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            return torch.from_numpy(wav).float().unsqueeze(0), sr
        except Exception:
            pass

        try:
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            sr = audio_segment.frame_rate
            samples = audio_segment.get_array_of_samples()
            wav = np.array(samples, dtype=np.float32)
            wav = wav / (2 ** (audio_segment.sample_width * 8 - 1))
            return torch.from_numpy(wav).float().unsqueeze(0), sr
        except Exception:
            pass

        for ext in [".webm", ".mp3", ".ogg", ".wav", ".m4a"]:
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    temp_path = f.name
                    f.write(audio_bytes)
                wav, sr = torchaudio.load(temp_path)
                os.unlink(temp_path)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                return wav, sr
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return None, None

    def handle_request(self, request: dict) -> dict:
        """Route request to appropriate handler."""
        command = request.get("command", "tts")

        handlers = {
            "check": self._handle_check,
            "status": self._handle_status,
            "preload": self._handle_preload,
            "unload": self._handle_unload,
            "tts": self._handle_tts,
            "asr": self._handle_asr,
            "audio_chat": self._handle_audio_chat,
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
            print(f"[LFM2 Daemon] Error receiving message: {e}", file=sys.stderr)
            return None

    def _send_message(self, conn: socket.socket, message: dict):
        """Send length-prefixed JSON message."""
        try:
            data = json.dumps(message).encode("utf-8")
            length = struct.pack(">I", len(data))
            conn.sendall(length + data)
        except Exception as e:
            print(f"[LFM2 Daemon] Error sending message: {e}", file=sys.stderr)

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
            f"[LFM2 Daemon] Started on {self.socket_path} (PID: {os.getpid()})",
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
                    print(f"[LFM2 Daemon] Accept error: {e}", file=sys.stderr)

        self.stop()

    def stop(self):
        """Stop the daemon server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.model_cache.clear()
        print("[LFM2 Daemon] Stopped", file=sys.stderr)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(
            f"[LFM2 Daemon] Received signal {signum}, shutting down...", file=sys.stderr
        )
        self.running = False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LFM2-Audio Daemon")
    parser.add_argument(
        "--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path"
    )
    parser.add_argument(
        "--preload", action="store_true", help="Preload model on startup"
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_HF_REPO, help="HuggingFace model ID to use"
    )
    args = parser.parse_args()

    daemon = LFM2Daemon(socket_path=args.socket)

    if args.preload:
        print(f"[LFM2 Daemon] Preloading model: {args.model_id}", file=sys.stderr)
        try:
            daemon._load_model(args.model_id)
        except Exception as e:
            print(f"[LFM2 Daemon] Preload failed: {e}", file=sys.stderr)

    daemon.start()


if __name__ == "__main__":
    main()
