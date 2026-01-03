"""LLM Client wrapper for Google Gemini."""

import json
from typing import Any

from google import genai
from google.genai import types

from src.config import config


class GeminiClient:
    """Wrapper for Google Gemini API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.gemini_api_key
        self.model = model or config.gemini_model
        self.client = genai.Client(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = None,
        max_tokens: int = 2048,
        response_format: str = "text"  # "text" or "json"
    ) -> str:
        """Generate a response from the model."""
        
        generation_config = types.GenerateContentConfig(
            temperature=temperature or config.temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )
        
        if response_format == "json":
            generation_config.response_mime_type = "application/json"
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config,
        )
        
        return response.text
    
    def generate_with_history(
        self,
        messages: list[dict],
        system_instruction: str = None,
        temperature: float = None,
    ) -> str:
        """Generate a response with conversation history."""
        
        # Build the full prompt with history
        history_text = ""
        for msg in messages[:-1]:  # All but last message
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"
        
        current_message = messages[-1]["content"] if messages else ""
        
        full_prompt = f"""Conversation History:
{history_text}

Current User Message: {current_message}

Please respond to the current user message, taking into account the conversation history."""
        
        return self.generate(
            prompt=full_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
    
    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_instruction: str = None,
    ) -> dict:
        """Generate a structured JSON response matching the schema."""
        
        schema_str = json.dumps(schema, indent=2)
        structured_prompt = f"""{prompt}

Please respond with a JSON object matching this schema:
{schema_str}

Respond ONLY with valid JSON, no additional text."""
        
        response_text = self.generate(
            prompt=structured_prompt,
            system_instruction=system_instruction,
            response_format="json",
            temperature=0.3,  # Lower temperature for structured output
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response_text}")


# Singleton instance
_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    """Get or create the Gemini client instance."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
