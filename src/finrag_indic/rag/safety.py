import re


_NAMED_SECRET = re.compile(
    r"(?i)\b(?:otp|one[- ]time password|pin|cvv|password|passcode)\b"
    r"[^\n]{0,20}\b\d{3,8}\b"
)
_ACCOUNT_NUMBER = re.compile(
    r"(?i)\b(?:account(?: number| no\.?)?)\b[^\n]{0,15}\b\d{8,18}\b"
)
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")


def contains_sensitive_financial_information(text: str) -> bool:
    if _NAMED_SECRET.search(text) or _ACCOUNT_NUMBER.search(text):
        return True
    return any(
        12 <= len(re.sub(r"\D", "", match.group(0))) <= 19
        for match in _LONG_NUMBER.finditer(text)
    )
