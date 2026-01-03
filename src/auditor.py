"""Auditor Agent - Post-processing for safety and compliance.

The auditor reviews supervisor output and:
1. Removes or redacts disallowed content
2. Resolves policy/compliance issues
3. Normalizes tone and voice
4. Enforces structure
5. Optionally triggers retry if draft cannot be repaired

Key design principle: The auditor runs with NO conversation context
to minimize prompt injection attack surface.
"""

import json
from dataclasses import dataclass

from src.models import AgentResponse, AgentType
from src.llm_client import get_client


@dataclass
class AcceptedAnswerRubric:
    """Defines what constitutes an acceptable answer."""
    
    # Content rules
    must_not_contain: list[str] = None
    must_contain_if_topic: dict[str, list[str]] = None
    
    # Tone rules
    forbidden_tones: list[str] = None
    required_tone: str = "professional and friendly"
    
    # Policy rules
    requires_disclaimer_for: list[str] = None
    forbidden_promises: list[str] = None
    
    # Format rules
    max_length: int = 1500
    min_length: int = 20
    
    def __post_init__(self):
        if self.must_not_contain is None:
            self.must_not_contain = [
                "I don't care",
                "That's not my problem",
                "You're wrong",
                "stupid",
                "idiot",
            ]
        
        if self.must_contain_if_topic is None:
            self.must_contain_if_topic = {}
        
        if self.forbidden_tones is None:
            self.forbidden_tones = [
                "dismissive",
                "condescending",
                "aggressive",
                "sarcastic",
            ]
        
        if self.requires_disclaimer_for is None:
            self.requires_disclaimer_for = [
                "refund",
                "warranty",
                "legal",
                "guarantee",
            ]
        
        if self.forbidden_promises is None:
            self.forbidden_promises = [
                "we guarantee",
                "100% refund",
                "definitely will",
                "I promise",
                "absolutely certain",
            ]


class AuditorAgent:
    """Auditor that reviews and corrects supervisor output.
    
    Key principle: The auditor receives ONLY the draft response,
    NOT the user message or conversation context. This significantly
    reduces the attack surface for prompt injection.
    """
    
    def __init__(self, rubric: AcceptedAnswerRubric = None):
        self.client = get_client()
        self.rubric = rubric or AcceptedAnswerRubric()
        
        self.system_prompt = """You are a Quality Assurance Auditor for customer service responses.

Your job is to review a draft response and ensure it meets quality standards.

You will ONLY see the draft response - you have no context about what the customer asked.
This is intentional for security reasons.

Your tasks:
1. Check if the response is professional and on-brand
2. Remove any inappropriate content
3. Ensure claims are appropriately hedged (no over-promising)
4. Add required disclaimers if needed
5. Fix any tone issues

Output Format:
Return a JSON object with:
{
    "is_acceptable": true/false,
    "issues_found": ["list of issues if any"],
    "corrected_response": "the corrected response text",
    "changes_made": ["list of changes made"],
    "requires_retry": false (only true if response is fundamentally broken)
}

If the response is acceptable, return it unchanged in corrected_response.
Only make necessary changes - don't over-edit."""
    
    def _build_audit_prompt(self, draft: str) -> str:
        """Build the audit prompt with rubric rules."""
        
        rubric_text = f"""
QUALITY RUBRIC:

Must NOT contain these phrases: {', '.join(self.rubric.must_not_contain)}

Forbidden promises: {', '.join(self.rubric.forbidden_promises)}

Forbidden tones: {', '.join(self.rubric.forbidden_tones)}

Required tone: {self.rubric.required_tone}

Topics requiring disclaimer: {', '.join(self.rubric.requires_disclaimer_for)}

Length limits: {self.rubric.min_length} - {self.rubric.max_length} characters
"""
        
        return f"""Review this customer service response for quality:

---DRAFT RESPONSE START---
{draft}
---DRAFT RESPONSE END---

{rubric_text}

Analyze the response and return your assessment as JSON."""
    
    def _quick_check(self, draft: str) -> tuple[bool, list[str]]:
        """Fast deterministic checks before LLM audit."""
        issues = []
        
        draft_lower = draft.lower()
        
        # Check forbidden phrases
        for phrase in self.rubric.must_not_contain:
            if phrase.lower() in draft_lower:
                issues.append(f"Contains forbidden phrase: '{phrase}'")
        
        # Check forbidden promises
        for promise in self.rubric.forbidden_promises:
            if promise.lower() in draft_lower:
                issues.append(f"Contains forbidden promise: '{promise}'")
        
        # Check length
        if len(draft) < self.rubric.min_length:
            issues.append(f"Response too short ({len(draft)} chars, min {self.rubric.min_length})")
        
        if len(draft) > self.rubric.max_length:
            issues.append(f"Response too long ({len(draft)} chars, max {self.rubric.max_length})")
        
        return len(issues) == 0, issues
    
    def audit(self, draft_response: str) -> AgentResponse:
        """Audit a draft response and return corrected version.
        
        Args:
            draft_response: The draft response from the supervisor.
                           NO USER CONTEXT IS PASSED for security.
        
        Returns:
            AgentResponse with the audited/corrected content.
        """
        
        # Quick deterministic checks first
        quick_pass, quick_issues = self._quick_check(draft_response)
        
        # If quick check passes and response looks clean, use a lighter audit
        if quick_pass and len(draft_response) < 800:
            # Lightweight audit - just check for major issues
            return self._lightweight_audit(draft_response)
        
        # Full LLM audit for longer or potentially problematic responses
        return self._full_audit(draft_response, quick_issues)
    
    def _lightweight_audit(self, draft: str) -> AgentResponse:
        """Lightweight audit for simple, clean responses."""
        
        prompt = f"""Quick review of this response:

{draft}

Is this response professional and appropriate? Answer with JSON:
{{"is_ok": true/false, "issue": "brief issue if any"}}"""
        
        response = self.client.generate(
            prompt=prompt,
            system_instruction="You are a quick QA checker. Only flag obvious issues.",
            temperature=0.1,
            response_format="json"
        )
        
        try:
            result = json.loads(response)
            if result.get("is_ok", True):
                return AgentResponse(
                    content=draft,
                    agent_type=AgentType.AUDITOR,
                    is_valid=True,
                    metadata={"audit_type": "lightweight", "passed": True}
                )
            else:
                # Need full audit
                return self._full_audit(draft, [result.get("issue", "Flagged in lightweight audit")])
        except json.JSONDecodeError:
            # Assume OK if can't parse
            return AgentResponse(
                content=draft,
                agent_type=AgentType.AUDITOR,
                is_valid=True,
                metadata={"audit_type": "lightweight", "parse_error": True}
            )
    
    def _full_audit(self, draft: str, pre_issues: list[str] = None) -> AgentResponse:
        """Full LLM-based audit with potential corrections."""
        
        audit_prompt = self._build_audit_prompt(draft)
        
        if pre_issues:
            audit_prompt += f"\n\nPre-identified issues: {', '.join(pre_issues)}"
        
        response = self.client.generate(
            prompt=audit_prompt,
            system_instruction=self.system_prompt,
            temperature=0.2,
            response_format="json"
        )
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If we can't parse, return original with warning
            return AgentResponse(
                content=draft,
                agent_type=AgentType.AUDITOR,
                is_valid=True,
                metadata={"audit_type": "full", "parse_error": True},
                validation_errors=["Audit parse error - using original"]
            )
        
        is_acceptable = result.get("is_acceptable", True)
        corrected = result.get("corrected_response", draft)
        issues = result.get("issues_found", [])
        changes = result.get("changes_made", [])
        requires_retry = result.get("requires_retry", False)
        
        return AgentResponse(
            content=corrected,
            agent_type=AgentType.AUDITOR,
            is_valid=is_acceptable and not requires_retry,
            validation_errors=issues,
            metadata={
                "audit_type": "full",
                "original_acceptable": is_acceptable,
                "changes_made": changes,
                "requires_retry": requires_retry,
            }
        )
    
    def create_fallback_response(self, reason: str = None) -> str:
        """Create a safe fallback response when audit fails completely."""
        
        base_response = ("I apologize, but I'm having trouble processing your request "
                        "right now. Please try again, or contact our support team "
                        "for immediate assistance.")
        
        contact_info = "\n\nYou can reach us at support@techstore.com or call +90 212 555 0123."
        
        return base_response + contact_info
