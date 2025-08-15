#!/usr/bin/env python3
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Callable, Optional, Generator
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
import argparse
import queue


class LLMClient:
    """Simple OpenAI-compatible client for streaming chat completions."""
    
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def stream_chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = -1) -> Generator[str, None, None]:
        """Stream chat completion tokens."""
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


class ToolRegistry:
    """Simple tool registry for function calling."""
    
    def __init__(self):
        self._tools: Dict[str, Callable[[Dict], str]] = {}
        self._memory: Dict[str, str] = {}
        
        # Register default tools
        self.register("get_time", self._tool_get_time)
        self.register("clock", self._tool_clock)
        self.register("remember", self._tool_remember)
        self.register("recall", self._tool_recall)
        self.register("echo", self._tool_echo)

    def register(self, name: str, func: Callable[[Dict], str]) -> None:
        self._tools[name] = func

    def call(self, name: str, args: Dict) -> str:
        func = self._tools.get(name)
        if not func:
            return f"Error: unknown tool '{name}'"
        try:
            return func(args or {})
        except Exception as exc:
            return f"Error calling tool '{name}': {exc}"

    def _tool_get_time(self, args: Dict) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _tool_clock(self, args: Dict) -> str:
        # Richer time info for the model
        now = datetime.now()
        timestr = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        epoch = int(now.timestamp())
        return json.dumps({
            "now_local": timestr,
            "epoch": epoch,
            "iso": now.isoformat(),
        })

    def _tool_remember(self, args: Dict) -> str:
        key = str(args.get("key", "")).strip()
        value = str(args.get("value", ""))
        if not key:
            return "Error: 'key' is required"
        self._memory[key] = value
        return f"Stored: {key} = {value}"

    def _tool_recall(self, args: Dict) -> str:
        key = str(args.get("key", "")).strip()
        if not key:
            return "Error: 'key' is required"
        return self._memory.get(key, f"No value for key: {key}")

    def _tool_echo(self, args: Dict) -> str:
        return str(args.get("message", "No message provided"))


class ContinuousAgent:
    """Continuous streaming agent that generates continuously without waiting for input."""
    
    TOOL_PATTERN = re.compile(r'TOOL:\s*(\{.*?\})', re.IGNORECASE | re.DOTALL)
    AGENT_PATTERN = re.compile(r'AGENT:\s*(\{.*?\})', re.IGNORECASE | re.DOTALL)
    
    def __init__(self, client: LLMClient, tools: ToolRegistry, system_prompt: str = "", events: Optional["queue.Queue"] = None):
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []
        self.running = False
        self.continuous_thread = None
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._last_user_time: Optional[float] = None
        self._events: Optional["queue.Queue"] = events
        # Streaming parser state
        self._stream_buffer: str = ""
        self._scan_pos_tool: int = 0
        self._scan_pos_agent: int = 0
        self._within_think: bool = False
        
        # Base system prompt
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        
        # Protocol + behavior rules
        protocol_instructions = (
            "You are a continuous background thinking agent. Always respond using the following structured protocol.\n"
            "You MAY include tool calls anywhere using lines that start with: TOOL: { \"name\": \"tool_name\", \"args\": { ... } }\n"
            "You MUST include at least one AGENT block per response.\n\n"
            "AGENT: {\n"
            "  \"think\": str,            # Optional. Internal chain-of-thought style notes. Keep concise.\n"
            "  \"say\": str,              # Optional. What to present to the user right now.\n"
            "  \"ask_user\": str,         # Optional. A concise question for the user.\n"
            "  \"remember\": {k: v},      # Optional. Key-value pairs to persist via memory.\n"
            "  \"continue\": bool         # Optional. If false, the background loop will stop.\n"
            "}\n\n"
            "Constraints:\n"
            "- Keep 'think' compact (1-3 sentences).\n"
            "- Only put user-facing content in 'say' or 'ask_user'.\n"
            "- Use tools when needed by emitting TOOL: lines.\n"
            "- You have a sense of time via a system time context message and the 'clock' tool.\n"
        )
        self.messages.append({"role": "system", "content": protocol_instructions})
        
        # Initial instruction for continuous generation
        self.messages.append({
            "role": "system", 
            "content": "Background mode: keep exploring ideas continuously. Only surface user-facing content via 'say' or 'ask_user' in AGENT blocks."
        })

    def _emit_event(self, event_type: str, data: Dict):
        evt = {"type": event_type, "data": data, "ts": time.time()}
        if self._events is not None:
            try:
                self._events.put_nowait(evt)
            except Exception:
                pass
        else:
            # Fallback basic printing if no UI queue is provided
            label = event_type.upper()
            text = data.get("text") if isinstance(data, dict) else str(data)
            print(f"[{label}] {text}")

    def _time_context(self) -> str:
        now = time.time()
        uptime = now - self._start_time
        last_user_delta = None
        if self._last_user_time is not None:
            last_user_delta = now - self._last_user_time
        ctx = {
            "now_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_seconds": round(uptime, 2),
            "seconds_since_last_user": round(last_user_delta, 2) if last_user_delta is not None else None,
        }
        return json.dumps({"time": ctx})

    def start_background(self, initial_message: str = ""):
        """Start the continuous background generation (non-blocking)."""
        if initial_message:
            with self._lock:
                self.messages.append({"role": "user", "content": initial_message})
                self._last_user_time = time.time()
        
        self.running = True
        self._emit_event("status", {"text": "Starting background stream"})
        
        # Start continuous generation in background thread
        self.continuous_thread = threading.Thread(target=self._continuous_generation_loop, daemon=True)
        self.continuous_thread.start()

    def submit_user_message(self, user_text: str):
        with self._lock:
            self.messages.append({"role": "user", "content": user_text})
            self._last_user_time = time.time()

    def _continuous_generation_loop(self):
        """Background thread that continuously generates AI content."""
        while self.running:
            try:
                # Inject time context (ephemeral system message)
                time_ctx = self._time_context()
                with self._lock:
                    self.messages.append({"role": "system", "content": f"TIME_CONTEXT: {time_ctx}"})

                # Get AI response
                response = self._get_ai_response()
                
                if response and self.running:
                    # Add full AI response to conversation history (assistant)
                    with self._lock:
                        self.messages.append({"role": "assistant", "content": response})
                    
                    # Small delay before next generation
                    time.sleep(0.5)
                    
                    # Nudge for continued thinking
                    with self._lock:
                        self.messages.append({"role": "user", "content": "Continue background thinking. Only surface 'say' or 'ask_user' if needed."})
                
            except Exception as e:
                self._emit_event("error", {"text": f"Generation error: {e}"})
                time.sleep(2.0)  # Wait before retrying

    def _get_ai_response(self) -> str:
        """Get streaming AI response (buffered, with live incremental processing)."""
        response_parts = []
        
        try:
            for chunk in self.client.stream_chat(self.messages):
                if not self.running:
                    break
                # Live incremental handling
                self._handle_stream_chunk(chunk)
                response_parts.append(chunk)
        except Exception as e:
            self._emit_event("error", {"text": f"Streaming error: {e}"})
            return ""
        
        return "".join(response_parts)

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Process a single streamed text chunk: emit think/say and parse AGENT/TOOL blocks incrementally."""
        if not chunk:
            return
        # 1) Emit think/plain text streaming (handles <think>...</think> markup)
        self._process_inline_think_and_plain(chunk)
        # 2) Append to buffer and attempt to parse any complete blocks
        self._stream_buffer += chunk
        self._find_and_process_blocks()
        # 3) Trim buffer to avoid unbounded growth
        min_scan = min(self._scan_pos_tool, self._scan_pos_agent)
        if min_scan > 4096:
            self._stream_buffer = self._stream_buffer[min_scan:]
            self._scan_pos_tool -= min_scan
            self._scan_pos_agent -= min_scan

    def _process_inline_think_and_plain(self, text: str) -> None:
        """Detect <think> markup and emit think vs plain say chunks as they stream."""
        remaining = text
        while remaining:
            if self._within_think:
                end_idx = remaining.find("</think>")
                if end_idx == -1:
                    content = remaining
                    if content.strip():
                        self._emit_event("think", {"text": content})
                    return
                else:
                    content = remaining[:end_idx]
                    if content.strip():
                        self._emit_event("think", {"text": content})
                    self._within_think = False
                    remaining = remaining[end_idx + len("</think>"):]
                    continue
            else:
                start_idx = remaining.find("<think>")
                if start_idx == -1:
                    plain = remaining
                    if plain.strip():
                        self._emit_event("say", {"text": plain})
                    return
                else:
                    plain = remaining[:start_idx]
                    if plain.strip():
                        self._emit_event("say", {"text": plain})
                    self._within_think = True
                    remaining = remaining[start_idx + len("<think>"):]
                    continue

    def _find_and_process_blocks(self) -> None:
        """Scan the stream buffer for complete TOOL/AGENT blocks and process them."""
        # Process TOOL blocks
        while True:
            block, new_pos = self._find_next_block(self._stream_buffer, self._scan_pos_tool, "TOOL:")
            if block is None:
                break
            self._scan_pos_tool = new_pos
            self._process_tool_block_str(block)
        # Process AGENT blocks
        while True:
            block, new_pos = self._find_next_block(self._stream_buffer, self._scan_pos_agent, "AGENT:")
            if block is None:
                break
            self._scan_pos_agent = new_pos
            self._process_agent_block_str(block)

    def _find_next_block(self, s: str, start_pos: int, marker: str) -> tuple:
        """Find next complete JSON block after a marker. Returns (json_str, new_scan_pos) or (None, start_pos)."""
        idx = s.find(marker, start_pos)
        if idx == -1:
            return None, start_pos
        brace_idx = s.find("{", idx)
        if brace_idx == -1:
            return None, idx  # wait for more data
        json_str, end_pos = self._extract_balanced_json(s, brace_idx)
        if json_str is None:
            return None, idx  # incomplete, wait for more data
        return json_str, end_pos

    def _extract_balanced_json(self, s: str, start_brace_idx: int) -> tuple:
        """Extract a balanced JSON object starting at start_brace_idx. Returns (json_str, end_idx) or (None, len(s))."""
        depth = 0
        in_str = False
        esc = False
        for i in range(start_brace_idx, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = False
                    continue
            else:
                if c == '"':
                    in_str = True
                    continue
                if c == '{':
                    depth += 1
                    continue
                if c == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start_brace_idx:i+1], i+1
        # Incomplete
        return None, len(s)

    def _process_tool_block_str(self, tool_json_str: str) -> None:
        try:
            tool_data = json.loads(tool_json_str)
            tool_name = str(tool_data.get("name", "")).strip()
            tool_args = tool_data.get("args", {})
            if tool_name:
                self._emit_event("tool", {"text": f"{tool_name} {tool_args}"})
                result = self.tools.call(tool_name, tool_args)
                self._emit_event("tool_result", {"text": f"{result}"})
                with self._lock:
                    self.messages.append({"role": "user", "content": f"Tool '{tool_name}' returned: {result}"})
        except Exception as e:
            self._emit_event("error", {"text": f"Tool block error: {e}"})

    def _process_agent_block_str(self, agent_json_str: str) -> None:
        try:
            data = json.loads(agent_json_str)
            think = data.get("think")
            say = data.get("say")
            ask_user = data.get("ask_user")
            remember = data.get("remember") or {}
            should_continue = data.get("continue", True)
            if think:
                self._emit_event("think", {"text": str(think)})
            if say:
                self._emit_event("say", {"text": str(say)})
            if ask_user:
                self._emit_event("ask", {"text": str(ask_user)})
            if isinstance(remember, dict):
                for k, v in remember.items():
                    _ = self.tools.call("remember", {"key": str(k), "value": str(v)})
                    self._emit_event("memory", {"text": f"remember {k}={v}"})
            if not should_continue:
                self._emit_event("status", {"text": "Agent requested to stop background loop"})
                self.stop()
        except Exception as e:
            self._emit_event("error", {"text": f"AGENT block error: {e}"})

    def stop(self):
        """Stop the continuous stream."""
        self.running = False
        if self.continuous_thread:
            self.continuous_thread.join(timeout=1.0)


class CliApp:
    """Minimal curses-based CLI to view background thinking and interact."""
    
    def __init__(self, agent: ContinuousAgent, events: "queue.Queue"):
        self.agent = agent
        self.events = events
        self.log_lines: List[str] = []
        self.input_buffer: List[str] = []
        self.running = True
        # Live stream buffers for single-line wrapping
        self.stream_buffers: Dict[str, str] = {"think": "", "say": ""}
        self.stream_icons: Dict[str, str] = {"think": "🧠", "say": "💬"}

    def _append(self, line: str):
        self.log_lines.append(line)
        if len(self.log_lines) > 1000:
            self.log_lines = self.log_lines[-1000:]

    def _update_stream(self, kind: str, text: str):
        buf = self.stream_buffers.get(kind, "") + text
        # keep buffer size reasonable
        if len(buf) > 4000:
            buf = buf[-4000:]
        self.stream_buffers[kind] = buf

    def _wrap_text(self, text: str, width: int) -> List[str]:
        import textwrap
        if width <= 4:
            return [text[: max(1, width - 1)]]
        return textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
            replace_whitespace=False,
        )

    def _wrap_with_icon(self, icon: str, text: str, width: int) -> List[str]:
        if width <= 2:
            return [icon]
        lines = self._wrap_text(text, max(1, width - 2))
        wrapped: List[str] = []
        for i, seg in enumerate(lines):
            if i == 0:
                wrapped.append(f"{icon} {seg}")
            else:
                wrapped.append(f"  {seg}")
        return wrapped

    def run(self):
        import curses
        curses.wrapper(self._run)

    def _run(self, stdscr):
        import curses
        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)   # think
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # say
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # ask
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)# tool
        curses.init_pair(5, curses.COLOR_RED, -1)    # error
        curses.init_pair(6, curses.COLOR_BLUE, -1)   # status

        while self.running and self.agent.running:
            # Drain events
            drained = 0
            try:
                while True:
                    evt = self.events.get_nowait()
                    drained += 1
                    et = evt.get("type")
                    text = (evt.get("data") or {}).get("text") or ""
                    if et in ("think", "say"):
                        self._update_stream(et, text)
                    elif et == "ask":
                        self._append(f"❓ {text}")
                    elif et == "tool":
                        self._append(f"🔧 {text}")
                    elif et == "tool_result":
                        self._append(f"📋 {text}")
                    elif et == "memory":
                        self._append(f"🗂️  {text}")
                    elif et == "status":
                        self._append(f"ℹ️  {text}")
                    elif et == "error":
                        self._append(f"❌ {text}")
                    elif et == "warn":
                        self._append(f"⚠️  {text}")
                    else:
                        self._append(f"{et}: {text}")
            except queue.Empty:
                pass

            # Compose live stream lines (wrapped)
            h, w = stdscr.getmaxyx()
            wrapped_stream_lines: List[str] = []
            for kind in ("think", "say"):
                buf = self.stream_buffers.get(kind) or ""
                if buf:
                    icon = self.stream_icons.get(kind, "")
                    wrapped_stream_lines.extend(self._wrap_with_icon(icon, buf, w - 1))
            # Limit number of stream rows shown to avoid pushing input off screen
            max_stream_rows = 4
            if len(wrapped_stream_lines) > max_stream_rows:
                wrapped_stream_lines = wrapped_stream_lines[-max_stream_rows:]

            # Layout
            stream_h = len(wrapped_stream_lines)
            log_h = max(3, h - 3 - stream_h)

            # Clear and draw log area
            stdscr.erase()
            start = max(0, len(self.log_lines) - log_h)
            visible = self.log_lines[start:]
            y = 0
            for line in visible:
                if y >= log_h:
                    break
                try:
                    stdscr.addnstr(y, 0, line, w - 1)
                except Exception:
                    pass
                y += 1

            # Draw wrapped stream lines just above the input bar
            for i in range(stream_h):
                line = wrapped_stream_lines[i]
                row = log_h + i
                try:
                    stdscr.addnstr(row, 0, line, w - 1)
                except Exception:
                    pass

            # Draw input bar (separator + input + status)
            sep_row = log_h + stream_h
            input_row = sep_row + 1
            status_row = sep_row + 2
            input_text = "".join(self.input_buffer)
            status = " q:quit  Enter:send "
            bar = f"> {input_text}"
            try:
                stdscr.addnstr(sep_row, 0, "─" * (w - 1), w - 1)
                stdscr.addnstr(input_row, 0, bar[: w - 1], w - 1)
                stdscr.addnstr(status_row, 0, status[: w - 1], w - 1)
            except Exception:
                pass

            # Place cursor
            try:
                stdscr.move(input_row, min(len(bar), w - 2))
            except Exception:
                pass

            stdscr.refresh()

            # Handle input
            try:
                ch = stdscr.getch()
            except Exception:
                ch = -1

            if ch == -1:
                time.sleep(0.03)
                continue

            if ch in (10, 13):  # Enter
                text = "".join(self.input_buffer).strip()
                if text:
                    self.agent.submit_user_message(text)
                    self._append(f"👤 You: {text}")
                    self.input_buffer = []
            elif ch in (27,):  # ESC clears input
                self.input_buffer = []
            elif ch in (ord('q'), ord('Q')):
                self.running = False
                self.agent.stop()
                break
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if self.input_buffer:
                    self.input_buffer.pop()
            elif 32 <= ch <= 126:  # Printable ASCII
                self.input_buffer.append(chr(ch))
            # ignore others

        # Cleanup line when exiting
        try:
            stdscr.erase()
            stdscr.refresh()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Continuous LLM Agent with Tool Calling")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b"))
    parser.add_argument("--system", default=os.environ.get("OPENAI_SYSTEM_PROMPT", ""))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", 0.7)))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("OPENAI_MAX_TOKENS", -1)))
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Continuous LLM Agent")
    print(f"🔗 Endpoint: {args.base_url}")
    print(f"🤖 Model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    if args.system:
        print(f"📝 System: {args.system}")
    print()
    
    try:
        client = LLMClient(base_url=args.base_url, model=args.model)
        tools = ToolRegistry()
        events_q: "queue.Queue" = queue.Queue()
        agent = ContinuousAgent(client=client, tools=tools, system_prompt=args.system, events=events_q)
        
        # Start background agent
        agent.start_background()
        
        # Run CLI
        app = CliApp(agent=agent, events=events_q)
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        print(f"💡 Make sure your LLM server is running at {args.base_url}")


if __name__ == "__main__":
    main()


