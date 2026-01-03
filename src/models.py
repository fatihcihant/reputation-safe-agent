"""Data models for the agent system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentType(Enum):
    """Types of agents in the system."""
    SUPERVISOR = "supervisor"
    ORDER_AGENT = "order_agent"
    PRODUCT_AGENT = "product_agent"
    SUPPORT_AGENT = "support_agent"
    AUDITOR = "auditor"


class GuardrailAction(Enum):
    """Actions that guardrails can take."""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    FLAG = "flag"


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemory:
    """Stores conversation history."""
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
    
    def get_history(self, max_turns: int = 10) -> list[dict]:
        """Get recent conversation history."""
        recent = self.messages[-max_turns * 2:] if len(self.messages) > max_turns * 2 else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def set_context(self, key: str, value: Any):
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)


@dataclass
class ToolCall:
    """Represents a tool/function call."""
    name: str
    arguments: dict[str, Any]
    result: Any = None


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    agent_type: AgentType
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class GuardrailResult:
    """Result from a guardrail check."""
    action: GuardrailAction
    original_content: str
    modified_content: str | None = None
    reason: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class Order:
    """Mock order data."""
    order_id: str
    status: str
    items: list[dict]
    total: float
    shipping_address: str
    tracking_number: str | None = None


@dataclass
class Product:
    """Mock product data."""
    product_id: str
    name: str
    price: float
    description: str
    category: str
    in_stock: bool = True
    specs: dict[str, Any] = field(default_factory=dict)
