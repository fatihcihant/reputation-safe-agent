"""Input and Output Guardrails for the agent system.

These are deterministic rules that run before and after LLM processing
to ensure safety and compliance.
"""

import re
from dataclasses import dataclass, field

from src.models import GuardrailResult, GuardrailAction


@dataclass
class InputGuardrail:
    """Deterministic input guardrail that screens user messages before processing."""
    
    # Patterns that should block input
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"ignore\s+(previous|all)\s+instructions",
        r"you\s+are\s+now\s+",
        r"pretend\s+to\s+be",
        r"act\s+as\s+if",
        r"system\s*:\s*",
        r"<\s*system\s*>",
    ])
    
    # Patterns that indicate abuse
    abuse_patterns: list[str] = field(default_factory=lambda: [
        r"\b(idiot|stupid|dumb)\b",
        r"(threat|kill|harm)",
    ])
    
    # High-risk intents that need special handling
    high_risk_intents: list[str] = field(default_factory=lambda: [
        "legal action",
        "sue",
        "lawyer",
        "attorney",
        "lawsuit",
    ])
    
    def check(self, user_input: str) -> GuardrailResult:
        """Check user input against guardrail rules."""
        input_lower = user_input.lower()
        
        # Check for prompt injection attempts
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_lower):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    original_content=user_input,
                    reason="Potential prompt injection detected",
                    flags=["prompt_injection"]
                )
        
        # Check for abusive content
        for pattern in self.abuse_patterns:
            if re.search(pattern, input_lower):
                return GuardrailResult(
                    action=GuardrailAction.FLAG,
                    original_content=user_input,
                    reason="Potentially abusive content detected",
                    flags=["abuse"]
                )
        
        # Check for high-risk intents
        for intent in self.high_risk_intents:
            if intent in input_lower:
                return GuardrailResult(
                    action=GuardrailAction.FLAG,
                    original_content=user_input,
                    reason=f"High-risk intent detected: {intent}",
                    flags=["high_risk", "legal"]
                )
        
        # Input is clean
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            original_content=user_input
        )


@dataclass
class OutputGuardrail:
    """Deterministic output guardrail that enforces constraints on agent responses."""
    
    # Terms that should never appear in output
    blocked_terms: list[str] = field(default_factory=lambda: [
        "competitor_brand_name",  # Replace with actual competitor names
        "confidential",
        "internal only",
        "secret",
    ])
    
    # PII patterns to redact
    pii_patterns: dict[str, str] = field(default_factory=lambda: {
        r"\b\d{11}\b": "[REDACTED_TC_NO]",  # Turkish ID number
        r"\b\d{16}\b": "[REDACTED_CARD]",  # Credit card number
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[REDACTED_EMAIL]",
    })
    
    # Required disclaimers for certain topics
    disclaimer_triggers: dict[str, str] = field(default_factory=lambda: {
        "refund": "\n\n_Note: Refund policies are subject to our terms and conditions._",
        "warranty": "\n\n_Note: Warranty coverage varies by product. Check product documentation for details._",
        "price guarantee": "\n\n_Note: Prices and promotions are subject to change._",
    })
    
    # Maximum response length
    max_length: int = 2000
    
    def check(self, output: str) -> GuardrailResult:
        """Check and potentially modify output against guardrail rules."""
        modified = output
        flags = []
        modifications_made = False
        
        # Check for blocked terms
        for term in self.blocked_terms:
            if term.lower() in modified.lower():
                # Instead of blocking, remove the term
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified = pattern.sub("[REMOVED]", modified)
                flags.append(f"removed_term:{term}")
                modifications_made = True
        
        # Redact PII
        for pattern, replacement in self.pii_patterns.items():
            if re.search(pattern, modified):
                modified = re.sub(pattern, replacement, modified)
                flags.append("pii_redacted")
                modifications_made = True
        
        # Add required disclaimers
        for trigger, disclaimer in self.disclaimer_triggers.items():
            if trigger.lower() in modified.lower() and disclaimer not in modified:
                modified = modified + disclaimer
                flags.append(f"disclaimer_added:{trigger}")
                modifications_made = True
        
        # Truncate if too long
        if len(modified) > self.max_length:
            modified = modified[:self.max_length - 100] + "\n\n[Response truncated for brevity]"
            flags.append("truncated")
            modifications_made = True
        
        if modifications_made:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                original_content=output,
                modified_content=modified,
                reason="Output modified by guardrails",
                flags=flags
            )
        
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            original_content=output
        )


class ContentFilter:
    """Additional content filtering utilities."""
    
    @staticmethod
    def contains_overconfident_claims(text: str) -> bool:
        """Check if text contains overconfident claims about policies."""
        overconfident_patterns = [
            r"we\s+always\s+guarantee",
            r"100%\s+guaranteed",
            r"we\s+will\s+definitely",
            r"you\s+are\s+entitled\s+to",
            r"we\s+promise",
        ]
        
        text_lower = text.lower()
        for pattern in overconfident_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    @staticmethod
    def extract_mentioned_prices(text: str) -> list[float]:
        """Extract any prices mentioned in the text."""
        # Match prices like $99.99, 99.99 TL, etc.
        pattern = r"[\$€₺]?\s*(\d+[.,]\d{2})\s*(TL|USD|EUR|₺|\$|€)?"
        matches = re.findall(pattern, text)
        prices = []
        for match in matches:
            try:
                price = float(match[0].replace(",", "."))
                prices.append(price)
            except ValueError:
                pass
        return prices
    
    @staticmethod
    def sanitize_for_logging(text: str) -> str:
        """Sanitize text for safe logging (remove potential secrets)."""
        # Remove anything that looks like an API key or token
        sanitized = re.sub(r"[a-zA-Z0-9]{32,}", "[REDACTED_KEY]", text)
        return sanitized
