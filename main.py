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
    
    def __init__(self, client: LLMClient, tools: ToolRegistry, system_prompt: str = ""):
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []
        self.running = False
        self.continuous_thread = None
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        
        # Add initial instruction for continuous generation
        self.messages.append({
            "role": "system", 
            "content": "You are a continuous thinking agent. Keep generating thoughts, ideas, and responses continuously. You can call tools when needed using TOOL: format. Think out loud and explore ideas."
        })

    def start_continuous_stream(self, initial_message: str = ""):
        """Start the continuous streaming generation."""
        if initial_message:
            self.messages.append({"role": "user", "content": initial_message})
        
        self.running = True
        print(f"🚀 Starting continuous AI stream...")
        print(f"💡 The AI will generate continuously and can call tools using: TOOL: {{'name': 'tool_name', 'args': {{'key': 'value'}}}}")
        print(f"👤 Type messages to interact, 'quit' to stop, or just watch the stream.\n")
        
        # Start continuous generation in background thread
        self.continuous_thread = threading.Thread(target=self._continuous_generation_loop, daemon=True)
        self.continuous_thread.start()
        
        # Start user interaction loop in main thread
        self._user_interaction_loop()

    def _continuous_generation_loop(self):
        """Background thread that continuously generates AI content."""
        while self.running:
            try:
                # Get AI response
                print("🤖 AI: ", end="", flush=True)
                response = self._get_ai_response()
                
                if response and self.running:
                    # Check for tool calls
                    self._process_tool_calls(response)
                    
                    # Add AI response to conversation
                    self.messages.append({"role": "assistant", "content": response})
                    
                    # Small delay before next generation
                    time.sleep(0.5)
                    
                    # Ask AI to continue thinking
                    self.messages.append({"role": "user", "content": "Continue thinking and exploring ideas."})
                
            except Exception as e:
                print(f"\n❌ Generation error: {e}")
                time.sleep(2.0)  # Wait before retrying

    def _user_interaction_loop(self):
        """Main thread for user interaction."""
        while self.running:
            try:
                # Non-blocking input check (simplified approach)
                user_input = input("👤 You (or press Enter to continue watching): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    print("👋 Stopping continuous stream...")
                    self.running = False
                    break
                
                if user_input:
                    # Add user message and let AI respond
                    self.messages.append({"role": "user", "content": user_input})
                    print("🤖 AI: ", end="", flush=True)
                    response = self._get_ai_response()
                    
                    if response:
                        self._process_tool_calls(response)
                        self.messages.append({"role": "assistant", "content": response})
                    
                    print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n👋 Stopping...")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _get_ai_response(self) -> str:
        """Get streaming AI response."""
        response_parts = []
        
        try:
            for chunk in self.client.stream_chat(self.messages):
                if not self.running:
                    break
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            return ""
        
        return "".join(response_parts)

    def _process_tool_calls(self, response: str):
        """Process any tool calls in the response."""
        tool_matches = self.TOOL_PATTERN.findall(response)
        
        for tool_match in tool_matches:
            try:
                tool_data = json.loads(tool_match)
                tool_name = tool_data.get("name", "").strip()
                tool_args = tool_data.get("args", {})
                
                if tool_name:
                    print(f"\n🔧 Calling tool: {tool_name}")
                    result = self.tools.call(tool_name, tool_args)
                    print(f"📋 Result: {result}")
                    
                    # Add tool result to conversation
                    self.messages.append({
                        "role": "user", 
                        "content": f"Tool '{tool_name}' returned: {result}"
                    })
                    
            except json.JSONDecodeError:
                print(f"\n⚠️  Invalid tool call format: {tool_match}")
            except Exception as e:
                print(f"\n❌ Tool call error: {e}")

    def stop(self):
        """Stop the continuous stream."""
        self.running = False
        if self.continuous_thread:
            self.continuous_thread.join(timeout=1.0)


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
        agent = ContinuousAgent(client=client, tools=tools, system_prompt=args.system)
        
        agent.start_continuous_stream()
        
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        print(f"💡 Make sure your LLM server is running at {args.base_url}")


if __name__ == "__main__":
    main()


