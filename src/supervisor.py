"""Supervisor Agent - Orchestrates sub-agents and owns the conversation.

The supervisor is responsible for:
1. Understanding user intent
2. Routing to appropriate sub-agents
3. Managing conversation context
4. Composing final responses from sub-agent outputs
"""

import json
from typing import Any

from src.models import AgentResponse, AgentType, ConversationMemory
from src.sub_agents import get_sub_agent, BaseSubAgent
from src.llm_client import get_client


class SupervisorAgent:
    """Orchestrates sub-agents and manages the conversation."""
    
    def __init__(self):
        self.client = get_client()
        self.memory = ConversationMemory()
        
        # Available sub-agents
        self.sub_agents: dict[AgentType, BaseSubAgent] = {
            AgentType.ORDER_AGENT: get_sub_agent(AgentType.ORDER_AGENT),
            AgentType.PRODUCT_AGENT: get_sub_agent(AgentType.PRODUCT_AGENT),
            AgentType.SUPPORT_AGENT: get_sub_agent(AgentType.SUPPORT_AGENT),
        }
        
        self.system_prompt = """You are a Customer Service Supervisor for TechStore, an e-commerce platform.
Your role is to:
1. Understand what the customer needs
2. Route their request to the appropriate specialist
3. Compose helpful, on-brand responses

Available specialists:
- ORDER_AGENT: For order status, tracking, cancellations, and order-related questions
- PRODUCT_AGENT: For product search, details, availability, and recommendations
- SUPPORT_AGENT: For general support, FAQ, returns policy, shipping info, and support tickets

Response Guidelines:
- Be warm and professional
- Use simple, clear language
- Don't make promises you can't keep
- If unsure, acknowledge uncertainty
- Stay focused on helping the customer

Brand Voice:
- Friendly but professional
- Helpful and solution-oriented
- Honest about limitations
- Never defensive or dismissive"""

        self.routing_prompt = """Based on the user message, determine which specialist should handle this request.

User Message: {message}

Previous Context: {context}

Respond with a JSON object:
{{
    "intent": "brief description of what the user wants",
    "route_to": "ORDER_AGENT" | "PRODUCT_AGENT" | "SUPPORT_AGENT" | "NONE",
    "requires_multiple": true/false,
    "additional_routes": ["list of additional agents if multiple needed"],
    "extracted_entities": {{
        "order_id": "if mentioned",
        "product_id": "if mentioned",
        "topic": "main topic"
    }},
    "is_greeting_or_smalltalk": true/false
}}

Only respond with the JSON object, no additional text."""
    
    def _route_query(self, user_message: str) -> dict:
        """Determine which sub-agent(s) should handle the query."""
        context_summary = {
            "recent_messages": len(self.memory.messages),
            "last_topic": self.memory.get_context("last_topic"),
            "last_order_id": self.memory.get_context("order_id"),
            "last_product_id": self.memory.get_context("product_id"),
        }
        
        prompt = self.routing_prompt.format(
            message=user_message,
            context=json.dumps(context_summary)
        )
        
        response = self.client.generate(
            prompt=prompt,
            system_instruction="You are a routing classifier. Output only valid JSON.",
            temperature=0.1,
            response_format="json"
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback routing
            return self._fallback_routing(user_message)
    
    def _fallback_routing(self, message: str) -> dict:
        """Simple keyword-based fallback routing."""
        message_lower = message.lower()
        
        # Order keywords
        order_keywords = ["order", "tracking", "shipped", "delivery", "cancel", "status", "ord-"]
        if any(kw in message_lower for kw in order_keywords):
            return {"route_to": "ORDER_AGENT", "intent": "order inquiry"}
        
        # Product keywords
        product_keywords = ["product", "price", "stock", "available", "buy", "search", "find", "show me", "prod-"]
        if any(kw in message_lower for kw in product_keywords):
            return {"route_to": "PRODUCT_AGENT", "intent": "product inquiry"}
        
        # Support/FAQ keywords
        support_keywords = ["return", "refund", "warranty", "shipping", "payment", "help", "support", "contact"]
        if any(kw in message_lower for kw in support_keywords):
            return {"route_to": "SUPPORT_AGENT", "intent": "support inquiry"}
        
        # Default to support for general queries
        return {"route_to": "SUPPORT_AGENT", "intent": "general inquiry", "is_greeting_or_smalltalk": True}
    
    def _get_agent_type(self, route_str: str) -> AgentType | None:
        """Convert route string to AgentType."""
        mapping = {
            "ORDER_AGENT": AgentType.ORDER_AGENT,
            "PRODUCT_AGENT": AgentType.PRODUCT_AGENT,
            "SUPPORT_AGENT": AgentType.SUPPORT_AGENT,
        }
        return mapping.get(route_str)
    
    def _compose_response(
        self,
        user_message: str,
        routing_result: dict,
        sub_agent_responses: list[AgentResponse]
    ) -> str:
        """Compose the final response from sub-agent outputs."""
        
        # Handle greetings/smalltalk directly
        if routing_result.get("is_greeting_or_smalltalk") and not sub_agent_responses:
            return self._handle_greeting(user_message)
        
        # If no sub-agent responses, handle directly
        if not sub_agent_responses:
            return self._handle_direct(user_message)
        
        # Compose from sub-agent responses
        sub_agent_content = "\n\n".join([
            f"[{r.agent_type.value}]: {r.content}"
            for r in sub_agent_responses
        ])
        
        composition_prompt = f"""User Message: {user_message}

Routing Decision: {routing_result.get('intent', 'general inquiry')}

Sub-Agent Responses:
{sub_agent_content}

Instructions:
- Synthesize the sub-agent response(s) into a natural, helpful reply
- Maintain our brand voice (friendly, professional, solution-oriented)
- If multiple agents responded, combine their information coherently
- Don't repeat yourself or include redundant information
- Keep the response focused and concise
- Add a brief follow-up offer if appropriate (e.g., "Is there anything else I can help you with?")

Compose the final response to send to the customer:"""
        
        response = self.client.generate(
            prompt=composition_prompt,
            system_instruction=self.system_prompt,
            temperature=0.6,
        )
        
        return response
    
    def _handle_greeting(self, message: str) -> str:
        """Handle greetings and smalltalk."""
        prompt = f"""The customer said: "{message}"

Respond with a warm, brief greeting. Offer to help with:
- Order inquiries (tracking, status, cancellations)
- Product information (search, details, availability)
- General support (returns, shipping, warranty, payments)

Keep it short and welcoming."""
        
        return self.client.generate(
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=0.7,
        )
    
    def _handle_direct(self, message: str) -> str:
        """Handle queries that don't need sub-agents."""
        prompt = f"""Customer message: "{message}"

Provide a helpful response. If you can't help with their specific request,
kindly explain what you can help with (orders, products, support)."""
        
        return self.client.generate(
            prompt=prompt,
            system_instruction=self.system_prompt,
            temperature=0.7,
        )
    
    def process(self, user_message: str) -> AgentResponse:
        """Process a user message through the supervisor pipeline."""
        
        # Add message to memory
        self.memory.add_message("user", user_message)
        
        # Step 1: Route the query
        routing_result = self._route_query(user_message)
        
        # Extract entities for context
        entities = routing_result.get("extracted_entities", {})
        if entities.get("order_id"):
            self.memory.set_context("order_id", entities["order_id"])
        if entities.get("product_id"):
            self.memory.set_context("product_id", entities["product_id"])
        if entities.get("topic"):
            self.memory.set_context("last_topic", entities["topic"])
        
        # Step 2: Call sub-agent(s)
        sub_agent_responses: list[AgentResponse] = []
        
        primary_route = routing_result.get("route_to")
        if primary_route and primary_route != "NONE":
            agent_type = self._get_agent_type(primary_route)
            if agent_type and agent_type in self.sub_agents:
                context = {
                    "order_id": self.memory.get_context("order_id"),
                    "product_id": self.memory.get_context("product_id"),
                    "history": self.memory.get_history(max_turns=3),
                }
                response = self.sub_agents[agent_type].process(user_message, context)
                sub_agent_responses.append(response)
        
        # Handle multiple routes if needed
        if routing_result.get("requires_multiple"):
            for route in routing_result.get("additional_routes", []):
                agent_type = self._get_agent_type(route)
                if agent_type and agent_type in self.sub_agents:
                    context = {
                        "order_id": self.memory.get_context("order_id"),
                        "product_id": self.memory.get_context("product_id"),
                    }
                    response = self.sub_agents[agent_type].process(user_message, context)
                    sub_agent_responses.append(response)
        
        # Step 3: Compose final response
        final_content = self._compose_response(
            user_message,
            routing_result,
            sub_agent_responses
        )
        
        # Add response to memory
        self.memory.add_message("assistant", final_content)
        
        # Collect all tool calls
        all_tool_calls = []
        for r in sub_agent_responses:
            all_tool_calls.extend(r.tool_calls)
        
        return AgentResponse(
            content=final_content,
            agent_type=AgentType.SUPERVISOR,
            tool_calls=all_tool_calls,
            metadata={
                "routing": routing_result,
                "sub_agents_used": [r.agent_type.value for r in sub_agent_responses],
            }
        )
    
    def reset_memory(self):
        """Reset the conversation memory."""
        self.memory = ConversationMemory()
