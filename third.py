#!/usr/bin/env python3
import json
import os
import threading
import time
import argparse
from typing import Dict, List, Generator, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
import sys
import termios
import tty
import select


class LLMClient:
    """Simple OpenAI-compatible client for streaming chat completions."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def stream_chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Stream chat completion tokens. If stop_event is set, stop early."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    if stop_event is not None and stop_event.is_set():
                        break

                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue

                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = event.get("choices") or []
                    if not choices:
                        continue

                    delta = (choices[0] or {}).get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"HTTP error: {exc}")


class InterruptibleStreamer:
    """Manages an interruptible streaming session and restarts with new system prompts."""

    def __init__(
        self,
        client: LLMClient,
        temperature: float,
        max_tokens: int,
        max_system_chars: int,
        print_prefix: str = "🤖 AI: ",
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_system_chars = max_system_chars
        self.print_prefix = print_prefix

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[str] = []
        self._active_messages: List[Dict] = []
        self._running = False

    def start(self, initial_system_prompt: str) -> None:
        with self._lock:
            if self._running:
                return
            self._buffer = []
            self._active_messages = [{"role": "system", "content": initial_system_prompt}]
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_stream, daemon=True)
            self._running = True
            print("🚀 Starting stream with system prompt...")
            print(f"📝 System: {initial_system_prompt}\n")
            print(self.print_prefix, end="", flush=True)
            self._thread.start()

    def interrupt_and_restart(self, user_message: str) -> None:
        # Always compose with whatever is in the buffer, even if already stopped
        streamed_text = self.get_buffer_text()

        # Stop current stream if running
        with self._lock:
            if self._running:
                self._stop_event.set()
                thread = self._thread
            else:
                thread = None

        if thread is not None:
            thread.join(timeout=2.0)

        # Build new system prompt from user message + buffered stream
        new_system = self._compose_new_system(user_message, streamed_text)

        # Start new session
        with self._lock:
            self._buffer = []
            self._active_messages = [{"role": "system", "content": new_system}]
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_stream, daemon=True)
            self._running = True
            print("\n🔁 Restarting stream with new system prompt...")
            print(f"📝 System: {new_system}\n")
            print(self.print_prefix, end="", flush=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_buffer_text(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def _run_stream(self) -> None:
        try:
            for chunk in self.client.stream_chat(
                self._active_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break
                # Print and buffer
                print(chunk, end="", flush=True)
                with self._lock:
                    self._buffer.append(chunk)
            print()  # newline after stream ends
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        finally:
            with self._lock:
                self._running = False

    def _compose_new_system(self, user_message: str, prior_stream_text: str) -> str:
        # Compose new system prompt using a simple template
        # Respect max_system_chars by trimming prior_stream_text from the end
        header_user = "User instruction:\n"
        header_ctx = "\n\nContext from previous stream (truncated):\n"
        base = f"{header_user}{user_message}"
        remaining = max(self.max_system_chars - len(base) - len(header_ctx), 0)
        if remaining <= 0 or not prior_stream_text:
            return base
        # Take the tail end which is likely most recent
        context_snippet = prior_stream_text[-remaining:]
        return f"{base}{header_ctx}{context_snippet}"


class KeypressWatcher:
    """Watches for any key to request an interrupt and buffers typed characters."""

    def __init__(self):
        self._event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._orig_attrs = None
        self._buffered_chars: List[str] = []

    def start(self) -> None:
        if self._running:
            return
        self._event.clear()
        self._buffered_chars = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore_terminal()

    def wait_for_interrupt(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout=timeout)

    def clear(self) -> None:
        self._event.clear()

    def consume_buffer(self) -> str:
        data = "".join(self._buffered_chars)
        self._buffered_chars = []
        return data

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        try:
            self._orig_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while self._running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not self._running:
                    break
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._buffered_chars.append(ch)
                        if not self._event.is_set():
                            self._event.set()
        except Exception:
            pass
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._orig_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_attrs)
            except Exception:
                pass
            self._orig_attrs = None


def main():
    parser = argparse.ArgumentParser(description="Interruptible Continuous LLM Streamer")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen/qwen3-4b-2507"))
    parser.add_argument("--system", default=os.environ.get("OPENAI_SYSTEM_PROMPT", "You are a continuous thinking agent. Generate helpful, insightful content continuously."))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", 0.7)))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("OPENAI_MAX_TOKENS", -1)))
    parser.add_argument("--max-system-chars", type=int, default=int(os.environ.get("OPENAI_MAX_SYSTEM_CHARS", 4000)))

    args = parser.parse_args()

    print("🚀 Interruptible Continuous LLM Streamer")
    print(f"🔗 Endpoint: {args.base_url}")
    print(f"🤖 Model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"🧱 Max system chars: {args.max_system_chars}")
    print("⌨️  Start typing at any time to interrupt and enter your message. Press Enter to submit.")
    print()

    try:
        client = LLMClient(base_url=args.base_url, model=args.model)
        streamer = InterruptibleStreamer(
            client=client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_system_chars=args.max_system_chars,
        )

        streamer.start(args.system)

        watcher = KeypressWatcher()
        watcher.start()

        while True:
            # Wait for any key to interrupt
            watcher.wait_for_interrupt()

            # Pause stream and collect user input cleanly
            streamer.stop()
            watcher.stop()

            # Gather any already-typed characters
            prefill = watcher.consume_buffer()

            try:
                # Print prompt with prefilled characters and read the rest of the line
                sys.stdout.write("👤 You: ")
                sys.stdout.write(prefill)
                sys.stdout.flush()
                rest = sys.stdin.readline()
            except KeyboardInterrupt:
                print("\n👋 Stopping...")
                break

            full_input = (prefill + rest).rstrip("\n").strip()

            if full_input.lower() in ["quit", "exit", "stop"]:
                break

            if not full_input:
                # No message provided; resume prior system prompt
                last_system = args.system
                if streamer._active_messages:
                    last_system = streamer._active_messages[0].get("content", args.system)
                streamer.start(last_system)
                watcher.start()
                continue

            # Restart streaming with new system prompt composed from user message + prior output
            streamer.interrupt_and_restart(full_input)
            watcher.start()

        watcher.stop()
        streamer.stop()
        print("👋 Goodbye")

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print(f"💡 Make sure your LLM server is running at {args.base_url}")


if __name__ == "__main__":
    main() 