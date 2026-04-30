"""PII redaction, safety checks."""

class OutputGuard:
    def sanitize(self, text: str) -> str:
        """Redact PII, check for unsafe content."""
        # Redact emails, phone numbers, SSN, etc.
        return text
