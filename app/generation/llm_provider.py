import requests
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from app.core.config import settings

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class MockLLMProvider(BaseLLMProvider):
    """A mock LLM for testing without API keys."""
    def generate(self, prompt: str) -> str:
        if "information is not available" in prompt:
            # We are testing the guardrail prompt probably
            # Let's just return a generic answer or the guardrail response
            # if we see a specific keyword. For simplicity:
            pass
            
        return "This is a mocked answer. Set LLM_PROVIDER to 'openai' or 'ollama' for real answers."

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API Provider."""
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        
    def generate(self, prompt: str) -> str:
        if not settings.openai_api_key:
            return "Error: OPENAI_API_KEY is not set."
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 # Low temperature for RAG
        )
        return response.choices[0].message.content

class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama Provider."""
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"

def get_llm() -> BaseLLMProvider:
    """Factory function to get the configured LLM provider."""
    provider_name = settings.llm_provider.lower()
    
    if provider_name == "openai":
        return OpenAILLMProvider()
    elif provider_name == "ollama":
        return OllamaLLMProvider()
    else:
        return MockLLMProvider()
