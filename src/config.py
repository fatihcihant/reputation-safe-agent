"""Configuration settings for the reputation-safe agent."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv



@dataclass
class Config:
    """Application configuration."""
    load_dotenv()
    # Gemini API
    gemini_api_key = os.getenv("gemini_api_key")
    gemini_model= os.getenv("gemini_model")
    
    # Qdrant Cloud
    qdrant_url= os.getenv("qdrant_url") 
    qdrant_api_key = os.getenv("qdrant_api_key") 
    qdrant_collection = os.getenv("qdrant_collection") 
    
    # Tavily API for web search
    tavily_api_key= os.getenv("avily_api_key") 

    # Agent settings
    max_retries: int = 3
    temperature: float = 0.7
    
    # Guardrail settings
    blocked_terms: list[str] = None
    required_disclaimers: dict[str, str] = None
     
    def __post_init__(self):
        if self.blocked_terms is None:
            self.blocked_terms = [
                "competitor_name",
                "internal_only",
                "confidential",
            ]
        
        if self.required_disclaimers is None:
            self.required_disclaimers = {
                "refund": "Refund policies are subject to terms and conditions.",
                "warranty": "Warranty coverage varies by product. Check product documentation for details.",
                "price": "Prices may vary and are subject to change.",
            }


config = Config()
