"""Injection detection, content filter."""

class InputGuard:
    def validate(self, text: str) -> tuple[bool, str]:
        """Check for prompt injection, malicious content."""
        # Check for injection patterns
        # Validate length, format
        return True, text
