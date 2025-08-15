#!/usr/bin/env python3
import json
import os
import threading
import time
import argparse
from typing import Dict, List, Generator, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
import sys
import termios
import tty
import select
import re
from datetime import datetime


class ConversationStorage:
    """Stores conversation history in a growing file with searchable index."""
    
    def __init__(self, storage_file: str = "conversation_history.jsonl"):
        self.storage_file = storage_file
        self.conversation_index = []  # In-memory index for fast access
        self._load_existing_history()
    
    def _load_existing_history(self):
        """Load existing conversation history from file."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            self.conversation_index.append(entry)
                        except json.JSONDecodeError:
                            continue
    
    def add_exchange(self, user_message: str, ai_response: str):
        """Add a new conversation exchange to storage."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": ai_response,
            "topics": self._extract_topics(user_message, ai_response),
            "length": len(ai_response)
        }
        
        # Add to in-memory index
        self.conversation_index.append(entry)
        
        # Append to file
        with open(self.storage_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _extract_topics(self, user_msg: str, ai_msg: str) -> List[str]:
        """Extract key topics from messages for searchability."""
        # Simple keyword extraction - could be enhanced with NLP
        combined = (user_msg + " " + ai_msg).lower()
        topics = []
        
        # Extract common topics
        topic_patterns = [
            r'\b(snake|game|tetris|history|how|what|when|where|why)\b',
            r'\b(programming|code|python|algorithm|data|structure)\b',
            r'\b(ai|machine learning|neural network|llm|gpt)\b',
            r'\b(conversation|chat|discussion|topic|subject)\b'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, combined)
            topics.extend(matches)
        
        return list(set(topics))
    
    def search_relevant_context(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for relevant conversation context based on query."""
        query_lower = query.lower()
        scored_results = []
        
        for entry in self.conversation_index:
            score = 0
            
            # Topic matching
            for topic in entry.get("topics", []):
                if topic in query_lower:
                    score += 3
            
            # Direct keyword matching
            if any(word in entry["user"].lower() for word in query_lower.split()):
                score += 2
            
            if any(word in entry["assistant"].lower() for word in query_lower.split()):
                score += 1
            
            # Recency bonus
            if entry in self.conversation_index[-3:]:  # Last 3 exchanges
                score += 1
            
            if score > 0:
                scored_results.append((score, entry))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_results[:max_results]]


class ContextInjectingLLMClient:
    """LLM client that can inject context during streaming."""
    
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
    
    def stream_with_context_injection(
        self,
        messages: List[Dict],
        conversation_storage: ConversationStorage,
        user_query: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        stop_event: Optional[threading.Event] = None,
        context_injection_interval: int = 50,  # Inject context every N tokens
    ) -> Generator[str, None, None]:
        """Stream chat completion with progressive context injection."""
        url = f"{self.base_url}/chat/completions"
        
        # Start with minimal context
        initial_context = conversation_storage.search_relevant_context(user_query, max_results=2)
        initial_system = self._build_initial_system(user_query, initial_context)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": initial_system}] + messages,
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
        
        token_count = 0
        
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                print(f"🔗 HTTP response received, status: {resp.status}")
                line_count = 0
                
                for raw_line in resp:
                    if stop_event is not None and stop_event.is_set():
                        break
                    
                    line_count += 1
                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                    except Exception as e:
                        print(f"⚠️  Decode error on line {line_count}: {e}")
                        continue
                    
                    if not line:
                        continue
                        
                    if not line.startswith("data: "):
                        print(f"⚠️  Non-data line {line_count}: {line[:50]}...")
                        continue
                    
                    data = line[6:].strip()
                    if data == "[DONE]":
                        print("🏁 Stream finished - restarting for continuous flow")
                        # For continuous thinking, we should restart the stream
                        # instead of breaking
                        break
                    
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error on line {line_count}: {e}")
                        continue
                    
                    choices = event.get("choices") or []
                    if not choices:
                        print(f"⚠️  No choices in event on line {line_count}")
                        continue
                    
                    delta = (choices[0] or {}).get("delta") or {}
                    content = delta.get("content")
                    
                    if content:
                        token_count += 1
                        print(f"📝 Token {token_count}: '{content}'")
                        
                        # Yield the actual token content first
                        yield content
                        
                        # After yielding, check if we should inject context
                        # This way context injection doesn't interrupt the token flow
                        if token_count % context_injection_interval == 0:
                            additional_context = conversation_storage.search_relevant_context(
                                user_query, max_results=3
                            )
                            if additional_context:
                                # Inject context marker that we can process
                                context_marker = f"\n\n[CONTEXT: {len(additional_context)} relevant exchanges found]\n"
                                yield context_marker
                    else:
                        print(f"⚠️  No content in delta on line {line_count}")
                        
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"HTTP error: {exc}")
    
    def stream_continuous_thoughts(
        self,
        conversation_storage: ConversationStorage,
        user_query: str,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
        context_injection_interval: int = 50,
    ) -> Generator[str, None, None]:
        """Stream continuous thoughts without natural completion."""
        # For continuous thinking, we need to keep restarting the stream
        # when it naturally ends
        while not stop_event or not stop_event.is_set():
            try:
                # Build system prompt for continuous thinking
                initial_context = conversation_storage.search_relevant_context(user_query, max_results=2)
                system_prompt = self._build_continuous_system(user_query, initial_context)
                
                # Create messages that encourage continuous output
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": "I'll continue thinking about this topic..."}
                ]
                
                # Stream with higher max_tokens to encourage longer responses
                for chunk in self._stream_single_completion(messages, temperature, 1000, stop_event):
                    if stop_event and stop_event.is_set():
                        return
                    yield chunk
                
                # If we get here, the stream ended naturally
                # Inject a continuation prompt
                continuation = "\n\n[Continuing thoughts on this topic...]\n"
                yield continuation
                
                # Small delay before restarting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️  Stream error, restarting: {e}")
                time.sleep(1)
                continue
    
    def _stream_single_completion(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Stream a single completion without context injection."""
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
                    if stop_event and stop_event.is_set():
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
    
    def _build_continuous_system(self, user_query: str, initial_context: List[Dict]) -> str:
        """Build system prompt that encourages continuous thinking."""
        base_prompt = f"""You are a continuous thinking agent. The user asked: {user_query}

Your task is to generate continuous, flowing thoughts on this topic. Don't stop until interrupted. 
Think deeply, explore different angles, make connections, and continue developing your thoughts.

"""
        
        if initial_context:
            base_prompt += "Relevant context from our conversation:\n"
            for entry in initial_context:
                base_prompt += f"- User: {entry['user'][:100]}...\n"
                base_prompt += f"- Assistant: {entry['assistant'][:150]}...\n\n"
        
        base_prompt += "\nNow continue thinking about this topic continuously..."
        
        return base_prompt
    
    def _build_initial_system(self, user_query: str, initial_context: List[Dict]) -> str:
        """Build initial system prompt with minimal relevant context."""
        base_prompt = f"You are a helpful AI assistant. The user asked: {user_query}\n\n"
        
        if initial_context:
            base_prompt += "Relevant context from our conversation:\n"
            for entry in initial_context:
                base_prompt += f"- User: {entry['user'][:100]}...\n"
                base_prompt += f"- Assistant: {entry['assistant'][:150]}...\n\n"
        
        return base_prompt


class StreamingContextManager:
    """Manages streaming with progressive context injection."""
    
    def __init__(
        self,
        client: ContextInjectingLLMClient,
        conversation_storage: ConversationStorage,
        temperature: float,
        max_tokens: int,
        print_prefix: str = "🤖 AI: ",
    ):
        self.client = client
        self.conversation_storage = conversation_storage
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.print_prefix = print_prefix
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[str] = []
        self._running = False
    
    def start_stream(self, user_message: str) -> None:
        """Start streaming with context injection."""
        with self._lock:
            if self._running:
                return
            
            self._buffer = []
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_context_injected_stream, 
                args=(user_message,),
                daemon=True
            )
            self._running = True
            print(f"🚀 Starting context-injected stream for: {user_message}")
            print(self.print_prefix, end="", flush=True)
            self._thread.start()
    
    def _run_context_injected_stream(self, user_message: str) -> None:
        """Run the stream with progressive context injection."""
        try:
            print(f"🔄 Starting continuous thoughts stream for: {user_message}")
            chunk_count = 0
            
            # Use the continuous thoughts method instead of one-shot completion
            for chunk in self.client.stream_continuous_thoughts(
                conversation_storage=self.conversation_storage,
                user_query=user_message,
                temperature=self.temperature,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break
                
                chunk_count += 1
                
                # Process continuation markers
                if chunk.startswith("[Continuing"):
                    print(f"\n🔄 {chunk}")
                    continue
                
                # Print and buffer the actual token content
                print(chunk, end="", flush=True)
                with self._lock:
                    self._buffer.append(chunk)
            
            print()  # newline after stream ends
            print(f"✅ Continuous stream completed with {chunk_count} chunks")
            
            # Store the complete exchange
            complete_response = self.get_buffer_text()
            self.conversation_storage.add_exchange(user_message, complete_response)
            
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self._running = False
    
    def stop(self) -> None:
        """Stop the current stream."""
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


class KeypressWatcher:
    """Watches for keypresses to interrupt and collect user input."""
    
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
    parser = argparse.ArgumentParser(description="Streaming Context Injection LLM Streamer")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", 0.7)))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("OPENAI_MAX_TOKENS", -1)))
    parser.add_argument("--storage-file", default="conversation_history.jsonl")
    parser.add_argument("--context-interval", type=int, default=50, help="Inject context every N tokens")
    
    args = parser.parse_args()
    
    print("🚀 Streaming Context Injection LLM Streamer")
    print(f"🔗 Endpoint: {args.base_url}")
    print(f"🤖 Model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"💾 Storage: {args.storage_file}")
    print(f"🔍 Context injection every {args.context_interval} tokens")
    print("⌨️  Start typing at any time to interrupt and enter your message. Press Enter to submit.")
    print()
    
    try:
        # Initialize conversation storage
        conversation_storage = ConversationStorage(args.storage_file)
        
        # Initialize client and streamer
        client = ContextInjectingLLMClient(base_url=args.base_url, model=args.model)
        streamer = StreamingContextManager(
            client=client,
            conversation_storage=conversation_storage,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        
        watcher = KeypressWatcher()
        watcher.start()
        
        while True:
            # Wait for any key to interrupt
            watcher.wait_for_interrupt()
            
            # Stop current stream if running
            if streamer.is_running():
                streamer.stop()
            watcher.stop()
            
            # Collect user input
            prefill = watcher.consume_buffer()
            
            try:
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
                print("💡 Please provide a message to continue the conversation.")
                watcher.start()
                continue
            
            # Start new stream with context injection
            streamer.start_stream(full_input)
            watcher.start()
        
        watcher.stop()
        streamer.stop()
        print("👋 Goodbye")
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print(f"💡 Make sure your LLM server is running at {args.base_url}")


if __name__ == "__main__":
    main()
