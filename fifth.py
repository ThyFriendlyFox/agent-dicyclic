#!/usr/bin/env python3
"""
Continuous output generator for a hosted OpenAI-compatible chat model.
Continuously streams output using the provided system prompt and
accumulated assistant context.
"""

import requests
import json
import time
import sys
import argparse
from typing import List, Dict, Any

class ContinuousStreamGenerator:
    def __init__(
        self,
        *,
        system_prompt: str,
        base_url: str = "http://localhost:1234",
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.7,
        max_tokens: int = -1,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = system_prompt
        
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages including system prompt and conversation history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages
        
    def stream_response(self, user_input: str) -> str:
        """Stream a response from the model and return the full response."""
        self.add_to_history("user", user_input)
        
        payload = {
            "model": self.model,
            "messages": self.get_messages(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return ""
                
            full_response = ""
            print("Assistant: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
            
            print()  # New line after response
            self.add_to_history("assistant", full_response)
            return full_response
            
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return ""
    
    def continuous_stream(self, *, seed_user_message: str = "continue", interval_seconds: float = 0.5):
        """Run continuous stream of interactions using a minimal user trigger.

        The previous assistant outputs are kept in history, so each new call
        generates more output under the given system prompt.
        """
        current_trigger = seed_user_message
        
        while True:
            try:
                response = self.stream_response(current_trigger)
                
                if not response:
                    print("Failed to get response. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nStream ended by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Retrying in 3 seconds...")
                time.sleep(3)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous output streamer for OpenAI-compatible chat API")
    parser.add_argument("--system", required=True, help="System prompt to drive continuous generation")
    parser.add_argument("--base-url", default="http://localhost:1234", help="Base URL of the API server")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=-1, help="Max tokens per completion (-1 for server default)")
    parser.add_argument("--seed-user", default="continue", help="Minimal user trigger message for each loop")
    parser.add_argument("--interval", type=float, default=0.5, help="Delay between loops in seconds")
    return parser.parse_args()

def main():
    args = parse_args()
    generator = ContinuousStreamGenerator(
        system_prompt=args.system,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    try:
        generator.continuous_stream(
            seed_user_message=args.seed_user,
            interval_seconds=args.interval,
        )
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
