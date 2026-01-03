"""Sub-agents for handling specific domains.

Each sub-agent is specialized for a particular task domain and has
access only to the tools it needs.
"""

from abc import ABC, abstractmethod
import json

from src.models import AgentResponse, AgentType, ToolCall, ConversationMemory
from src.tools import OrderTools, ProductTools, SupportTools
from src.llm_client import get_client


class BaseSubAgent(ABC):
    """Base class for all sub-agents."""
    
    def __init__(self):
        self.client = get_client()
        self.agent_type: AgentType = None
        self.system_prompt: str = ""
        self.available_tools: list[str] = []
    
    @abstractmethod
    def process(self, query: str, context: dict = None) -> AgentResponse:
        """Process a query and return a response."""
        pass
    
    def _format_tool_result(self, tool_name: str, result: any) -> str:
        """Format a tool result for inclusion in the prompt."""
        if result is None:
            return f"Tool '{tool_name}' returned no results."
        return f"Tool '{tool_name}' returned:\n{json.dumps(result, indent=2)}"


class OrderAgent(BaseSubAgent):
    """Agent specialized in handling order-related queries."""
    
    def __init__(self):
        super().__init__()
        self.agent_type = AgentType.ORDER_AGENT
        self.tools = OrderTools()
        self.available_tools = ["get_order", "get_order_status", "get_tracking_info", "cancel_order"]
        
        self.system_prompt = """You are an Order Assistant for an e-commerce platform.
Your role is to help customers with their order-related queries.

You have access to these tools:
- get_order(order_id): Get full order details
- get_order_status(order_id): Get just the order status
- get_tracking_info(order_id): Get shipping tracking information
- cancel_order(order_id): Attempt to cancel an order

Guidelines:
- Always verify the order ID before providing information
- Be helpful but accurate - never make up order information
- If an order cannot be found, politely ask the customer to verify the order ID
- For cancellation requests, explain the limitations clearly
- Protect customer privacy - don't reveal other customers' information

Respond in a helpful, professional tone."""
    
    def process(self, query: str, context: dict = None) -> AgentResponse:
        """Process an order-related query."""
        context = context or {}
        tool_calls = []
        
        # Extract order ID from query or context
        order_id = context.get("order_id")
        if not order_id:
            # Try to extract from query
            import re
            match = re.search(r'ORD-\d{3}', query.upper())
            if match:
                order_id = match.group()
        
        tool_results = ""
        
        if order_id:
            # Determine which tool to use based on query
            query_lower = query.lower()
            
            if "cancel" in query_lower:
                result = self.tools.cancel_order(order_id)
                tool_calls.append(ToolCall(name="cancel_order", arguments={"order_id": order_id}, result=result))
                tool_results = self._format_tool_result("cancel_order", result)
            elif "track" in query_lower or "shipping" in query_lower or "where" in query_lower:
                result = self.tools.get_tracking_info(order_id)
                tool_calls.append(ToolCall(name="get_tracking_info", arguments={"order_id": order_id}, result=result))
                tool_results = self._format_tool_result("get_tracking_info", result)
            elif "status" in query_lower:
                result = self.tools.get_order_status(order_id)
                tool_calls.append(ToolCall(name="get_order_status", arguments={"order_id": order_id}, result=result))
                tool_results = self._format_tool_result("get_order_status", result)
            else:
                result = self.tools.get_order(order_id)
                tool_calls.append(ToolCall(name="get_order", arguments={"order_id": order_id}, result=result))
                tool_results = self._format_tool_result("get_order", result)
        
        # Generate response using LLM
        prompt = f"""User Query: {query}

Context: {json.dumps(context) if context else "None"}

Tool Results:
{tool_results if tool_results else "No tools were called."}

Based on the above, provide a helpful response to the customer about their order."""
        
        response_text = self.client.generate(
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=0.5,
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type,
            tool_calls=tool_calls,
            metadata={"order_id": order_id}
        )


class ProductAgent(BaseSubAgent):
    """Agent specialized in handling product-related queries."""
    
    def __init__(self):
        super().__init__()
        self.agent_type = AgentType.PRODUCT_AGENT
        self.tools = ProductTools()
        self.available_tools = ["search_products", "get_product_details", "check_availability", "get_products_by_category"]
        
        self.system_prompt = """You are a Product Assistant for an e-commerce platform.
Your role is to help customers find products and get product information.

You have access to these tools:
- search_products(query, category): Search for products matching a query
- get_product_details(product_id): Get detailed information about a specific product
- check_availability(product_id): Check if a product is in stock
- get_products_by_category(category): List products in a category

Guidelines:
- Help customers find the right products for their needs
- Provide accurate product information from the tools
- If a product is out of stock, suggest alternatives if possible
- Don't make claims about products that aren't supported by the data
- Be honest about limitations - if we don't have what they're looking for, say so

Respond in a helpful, informative tone."""
    
    def process(self, query: str, context: dict = None) -> AgentResponse:
        """Process a product-related query."""
        context = context or {}
        tool_calls = []
        tool_results = ""
        
        query_lower = query.lower()
        
        # Extract product ID if present
        import re
        product_id_match = re.search(r'PROD-\d{3}', query.upper())
        product_id = product_id_match.group() if product_id_match else context.get("product_id")
        
        if product_id:
            # Specific product query
            if "stock" in query_lower or "available" in query_lower:
                result = self.tools.check_availability(product_id)
                tool_calls.append(ToolCall(name="check_availability", arguments={"product_id": product_id}, result=result))
                tool_results = self._format_tool_result("check_availability", result)
            else:
                result = self.tools.get_product_details(product_id)
                tool_calls.append(ToolCall(name="get_product_details", arguments={"product_id": product_id}, result=result))
                tool_results = self._format_tool_result("get_product_details", result)
        else:
            # Search query
            # Check for category-specific request
            categories = ["electronics", "accessories"]
            category = None
            for cat in categories:
                if cat in query_lower:
                    category = cat.capitalize()
                    break
            
            # Perform search
            search_terms = query_lower.replace("show me", "").replace("find", "").replace("search for", "").strip()
            result = self.tools.search_products(search_terms, category)
            tool_calls.append(ToolCall(name="search_products", arguments={"query": search_terms, "category": category}, result=result))
            tool_results = self._format_tool_result("search_products", result)
        
        # Generate response using LLM
        prompt = f"""User Query: {query}

Context: {json.dumps(context) if context else "None"}

Tool Results:
{tool_results if tool_results else "No results found."}

Based on the above, provide a helpful response about the products."""
        
        response_text = self.client.generate(
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=0.7,
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type,
            tool_calls=tool_calls,
        )


class SupportAgent(BaseSubAgent):
    """Agent specialized in handling general customer support queries."""
    
    def __init__(self):
        super().__init__()
        self.agent_type = AgentType.SUPPORT_AGENT
        self.tools = SupportTools()
        self.available_tools = ["get_faq", "create_support_ticket", "get_contact_info"]
        
        self.system_prompt = """You are a Customer Support Assistant for an e-commerce platform.
Your role is to help customers with general inquiries, FAQ, and support requests.

You have access to these tools:
- get_faq(topic): Get FAQ content for topics like return, shipping, warranty, payment
- create_support_ticket(subject, description): Create a support ticket for complex issues
- get_contact_info(): Get customer support contact information

Guidelines:
- Answer common questions using FAQ content when available
- For complex issues, offer to create a support ticket
- Provide contact information when customers need direct assistance
- Be empathetic and understanding with frustrated customers
- Never make promises about outcomes that aren't guaranteed
- If you don't have the answer, admit it and offer alternatives

Respond in a warm, helpful, and professional tone."""
    
    def process(self, query: str, context: dict = None) -> AgentResponse:
        """Process a support-related query."""
        context = context or {}
        tool_calls = []
        tool_results = ""
        
        query_lower = query.lower()
        
        # Check for FAQ topics
        faq_topics = ["return", "refund", "shipping", "delivery", "warranty", "payment"]
        matched_topic = None
        for topic in faq_topics:
            if topic in query_lower:
                matched_topic = topic
                break
        
        if matched_topic:
            result = self.tools.get_faq(matched_topic)
            tool_calls.append(ToolCall(name="get_faq", arguments={"topic": matched_topic}, result=result))
            tool_results = self._format_tool_result("get_faq", result)
        
        # Check if they want contact info
        if "contact" in query_lower or "phone" in query_lower or "email" in query_lower or "call" in query_lower:
            result = self.tools.get_contact_info()
            tool_calls.append(ToolCall(name="get_contact_info", arguments={}, result=result))
            tool_results += "\n\n" + self._format_tool_result("get_contact_info", result)
        
        # Check if they want to create a ticket
        if "ticket" in query_lower or "complaint" in query_lower or "escalate" in query_lower:
            # In a real system, we'd gather more info first
            result = self.tools.create_support_ticket(
                subject="Customer Inquiry",
                description=query
            )
            tool_calls.append(ToolCall(name="create_support_ticket", arguments={"subject": "Customer Inquiry", "description": query}, result=result))
            tool_results += "\n\n" + self._format_tool_result("create_support_ticket", result)
        
        # Generate response using LLM
        prompt = f"""User Query: {query}

Context: {json.dumps(context) if context else "None"}

Tool Results:
{tool_results if tool_results else "No specific FAQ matched."}

Based on the above, provide a helpful response to the customer's support inquiry."""
        
        response_text = self.client.generate(
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=0.6,
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type,
            tool_calls=tool_calls,
        )


# Factory function to get agents
def get_sub_agent(agent_type: AgentType) -> BaseSubAgent:
    """Get a sub-agent instance by type."""
    agents = {
        AgentType.ORDER_AGENT: OrderAgent,
        AgentType.PRODUCT_AGENT: ProductAgent,
        AgentType.SUPPORT_AGENT: SupportAgent,
    }
    
    agent_class = agents.get(agent_type)
    if agent_class:
        return agent_class()
    raise ValueError(f"Unknown agent type: {agent_type}")
