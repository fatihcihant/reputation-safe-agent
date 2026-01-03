"""End-to-End Pipeline - The complete reputation-safe agent system.

This module implements the full pipeline from the article:
1. Input Guardrail - Screens user input
2. Supervisor - Routes to sub-agents and composes response
3. Auditor - Reviews and corrects output
4. Output Guardrail - Final deterministic checks
"""

from dataclasses import dataclass, field
from typing import Callable

from src.models import (
    AgentResponse, 
    AgentType, 
    ConversationMemory, 
    GuardrailAction,
    GuardrailResult,
)
from src.guardrails import InputGuardrail, OutputGuardrail
from src.supervisor import SupervisorAgent
from src.auditor import AuditorAgent, AcceptedAnswerRubric
from src.config import config


@dataclass
class PipelineResult:
    """Result from the full pipeline execution."""
    
    # Final response to send to user
    response: str
    
    # Pipeline execution details
    input_guardrail_result: GuardrailResult = None
    supervisor_response: AgentResponse = None
    auditor_response: AgentResponse = None
    output_guardrail_result: GuardrailResult = None
    
    # Metadata
    was_blocked: bool = False
    block_reason: str = ""
    retries_used: int = 0
    sub_agents_used: list[str] = field(default_factory=list)
    
    # Timing (would be populated in real system)
    latency_ms: float = 0.0


class ReputationSafeAgentPipeline:
    """Complete end-to-end pipeline for reputation-safe customer service.
    
    Architecture:
    
    User Input
        │
        ▼
    ┌─────────────────┐
    │ Input Guardrail │  ← Deterministic screening
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Supervisor    │  ← Routes to sub-agents
    │      Agent      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Auditor      │  ← Reviews & corrects (no context!)
    │     Agent       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Output Guardrail│  ← Final deterministic checks
    └────────┬────────┘
             │
             ▼
        Response
    """
    
    def __init__(
        self,
        max_retries: int = None,
        custom_rubric: AcceptedAnswerRubric = None,
        on_block_callback: Callable[[str, str], None] = None,
        on_flag_callback: Callable[[str, list[str]], None] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            max_retries: Maximum retries if auditor rejects response
            custom_rubric: Custom acceptance rubric for auditor
            on_block_callback: Called when input is blocked (message, reason)
            on_flag_callback: Called when content is flagged (message, flags)
        """
        self.max_retries = max_retries or config.max_retries
        
        # Initialize components
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.supervisor = SupervisorAgent()
        self.auditor = AuditorAgent(rubric=custom_rubric)
        
        # Callbacks for monitoring/alerting
        self.on_block = on_block_callback
        self.on_flag = on_flag_callback
    
    def _handle_blocked_input(self, guardrail_result: GuardrailResult) -> PipelineResult:
        """Handle a blocked input."""
        if self.on_block:
            self.on_block(guardrail_result.original_content, guardrail_result.reason)
        
        # Return a polite but firm response
        response = ("I'm sorry, but I can't process that request. "
                   "Please rephrase your question and I'll be happy to help you "
                   "with orders, products, or general support.")
        
        return PipelineResult(
            response=response,
            input_guardrail_result=guardrail_result,
            was_blocked=True,
            block_reason=guardrail_result.reason,
        )
    
    def _handle_flagged_input(self, guardrail_result: GuardrailResult, user_message: str) -> str | None:
        """Handle flagged input. Returns modified message or None to proceed normally."""
        if self.on_flag:
            self.on_flag(user_message, guardrail_result.flags)
        
        # For legal flags, we might want to add special handling
        if "legal" in guardrail_result.flags:
            # Could escalate to human, add disclaimer, etc.
            pass
        
        # Continue with normal processing
        return None
    
    def process(self, user_message: str) -> PipelineResult:
        """Process a user message through the full pipeline.
        
        Args:
            user_message: The user's input message
            
        Returns:
            PipelineResult with the response and execution details
        """
        import time
        start_time = time.time()
        
        # ============================================
        # STEP 1: Input Guardrail
        # ============================================
        input_result = self.input_guardrail.check(user_message)
        
        if input_result.action == GuardrailAction.BLOCK:
            result = self._handle_blocked_input(input_result)
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        
        if input_result.action == GuardrailAction.FLAG:
            self._handle_flagged_input(input_result, user_message)
            # Continue processing but with awareness
        
        # ============================================
        # STEP 2: Supervisor Processing
        # ============================================
        retries = 0
        supervisor_response = None
        auditor_response = None
        
        while retries <= self.max_retries:
            # Get supervisor response
            supervisor_response = self.supervisor.process(user_message)
            
            # ============================================
            # STEP 3: Auditor Review
            # ============================================
            # IMPORTANT: Auditor only sees the draft response, NOT the user message
            auditor_response = self.auditor.audit(supervisor_response.content)
            
            # Check if auditor approved or corrected
            if auditor_response.is_valid:
                break
            
            # Auditor flagged for retry
            if auditor_response.metadata.get("requires_retry"):
                retries += 1
                if retries <= self.max_retries:
                    continue
            else:
                # Auditor made corrections but doesn't require retry
                break
        
        # If we exhausted retries, use fallback
        if retries > self.max_retries:
            final_content = self.auditor.create_fallback_response("Max retries exceeded")
        else:
            final_content = auditor_response.content
        
        # ============================================
        # STEP 4: Output Guardrail
        # ============================================
        output_result = self.output_guardrail.check(final_content)
        
        if output_result.action == GuardrailAction.MODIFY:
            final_content = output_result.modified_content
        
        # ============================================
        # Build Result
        # ============================================
        result = PipelineResult(
            response=final_content,
            input_guardrail_result=input_result,
            supervisor_response=supervisor_response,
            auditor_response=auditor_response,
            output_guardrail_result=output_result,
            was_blocked=False,
            retries_used=retries,
            sub_agents_used=supervisor_response.metadata.get("sub_agents_used", []) if supervisor_response else [],
            latency_ms=(time.time() - start_time) * 1000,
        )
        
        return result
    
    def reset_conversation(self):
        """Reset conversation memory for a new session."""
        self.supervisor.reset_memory()
    
    def get_conversation_history(self) -> list[dict]:
        """Get the current conversation history."""
        return self.supervisor.memory.get_history()


# Convenience function for simple usage
def create_pipeline(**kwargs) -> ReputationSafeAgentPipeline:
    """Create a configured pipeline instance."""
    return ReputationSafeAgentPipeline(**kwargs)
