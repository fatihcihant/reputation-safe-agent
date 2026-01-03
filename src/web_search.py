"""Web search module using Tavily API.

Tavily provides fast, accurate web search optimized for LLM applications.
This module integrates Tavily for real-time information retrieval.
"""

from dataclasses import dataclass
from typing import Any

from tavily import TavilyClient

from src.config import config


@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    content: str
    score: float
    raw_content: str | None = None


class TavilySearch:
    """Web search using Tavily API."""
    
    def __init__(self, api_key: str = None):
        """Initialize Tavily client.
        
        Args:
            api_key: Tavily API key (uses env var if not provided)
        """
        self.api_key = api_key or config.tavily_api_key
        
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Tavily is configured and available."""
        return self.client is not None
    
    def search(
        self,
        query: str,
        search_depth: str = "basic",  # "basic" or "advanced"
        max_results: int = 5,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_domains: list[str] = None,
        exclude_domains: list[str] = None,
    ) -> dict[str, Any]:
        """Perform a web search.
        
        Args:
            query: Search query
            search_depth: "basic" (faster) or "advanced" (more thorough)
            max_results: Maximum number of results (1-10)
            include_answer: Include AI-generated answer summary
            include_raw_content: Include full page content
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            
        Returns:
            Dictionary with 'answer', 'results', and metadata
        """
        if not self.client:
            return {
                "error": "Tavily API key not configured",
                "answer": None,
                "results": [],
            }
        
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            
            # Parse results
            results = []
            for r in response.get("results", []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    raw_content=r.get("raw_content"),
                ))
            
            return {
                "answer": response.get("answer"),
                "results": results,
                "query": query,
                "response_time": response.get("response_time", 0),
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "results": [],
            }
    
    def quick_search(self, query: str, max_results: int = 3) -> str:
        """Perform a quick search and return formatted results.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            Formatted string with search results
        """
        result = self.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
        
        if result.get("error"):
            return f"Search error: {result['error']}"
        
        output = []
        
        # Include answer if available
        if result.get("answer"):
            output.append(f"Summary: {result['answer']}\n")
        
        # Include results
        output.append("Sources:")
        for i, r in enumerate(result.get("results", []), 1):
            output.append(f"{i}. {r.title}")
            output.append(f"   {r.content[:200]}...")
            output.append(f"   URL: {r.url}\n")
        
        return "\n".join(output)
    
    def search_for_context(
        self,
        query: str,
        context_type: str = "general",
    ) -> dict[str, Any]:
        """Search with context-specific settings.
        
        Args:
            query: Search query
            context_type: Type of search context
                - "product": Product information
                - "support": Customer support info
                - "news": Recent news
                - "general": General search
                
        Returns:
            Search results optimized for context
        """
        settings = {
            "product": {
                "search_depth": "advanced",
                "max_results": 5,
                "include_answer": True,
            },
            "support": {
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,
            },
            "news": {
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
            },
            "general": {
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
            },
        }
        
        ctx_settings = settings.get(context_type, settings["general"])
        return self.search(query=query, **ctx_settings)


class WebSearchTool:
    """Tool interface for web search in the agent system."""
    
    def __init__(self):
        self.tavily = TavilySearch()
    
    def search_product_info(self, product_name: str) -> dict:
        """Search for product information online.
        
        Args:
            product_name: Name of the product to search for
            
        Returns:
            Search results with product information
        """
        query = f"{product_name} product specifications reviews"
        result = self.tavily.search_for_context(query, context_type="product")
        
        return {
            "query": query,
            "answer": result.get("answer"),
            "sources": [
                {"title": r.title, "url": r.url, "content": r.content[:300]}
                for r in result.get("results", [])
            ],
        }
    
    def search_competitor_prices(self, product_name: str) -> dict:
        """Search for competitor pricing (be careful with this!).
        
        Args:
            product_name: Product to search
            
        Returns:
            Pricing information from web
        """
        query = f"{product_name} price comparison buy"
        result = self.tavily.search_for_context(query, context_type="product")
        
        return {
            "query": query,
            "answer": result.get("answer"),
            "sources": [
                {"title": r.title, "url": r.url}
                for r in result.get("results", [])
            ],
        }
    
    def search_support_info(self, topic: str) -> dict:
        """Search for support-related information.
        
        Args:
            topic: Support topic to search
            
        Returns:
            Support information from web
        """
        query = f"{topic} customer support help guide"
        result = self.tavily.search_for_context(query, context_type="support")
        
        return {
            "query": query,
            "answer": result.get("answer"),
            "sources": [
                {"title": r.title, "url": r.url, "content": r.content[:200]}
                for r in result.get("results", [])
            ],
        }
    
    def search_general(self, query: str) -> dict:
        """General web search.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        result = self.tavily.search_for_context(query, context_type="general")
        
        return {
            "query": query,
            "answer": result.get("answer"),
            "sources": [
                {"title": r.title, "url": r.url, "content": r.content[:300]}
                for r in result.get("results", [])
            ],
        }


# Singleton instance
_search_instance: TavilySearch | None = None


def get_search() -> TavilySearch:
    """Get or create the Tavily search instance."""
    global _search_instance
    if _search_instance is None:
        _search_instance = TavilySearch()
    return _search_instance
