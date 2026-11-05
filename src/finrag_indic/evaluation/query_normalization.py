import re


_ABBREVIATIONS = {
    "si": "Simple Interest",
    "ci": "Compound Interest",
    "emi": "Equated Monthly Instalment",
    "fd": "Fixed Deposit",
    "rd": "Recurring Deposit",
    "upi": "Unified Payments Interface",
    "atm": "Automated Teller Machine",
    "kyc": "Know Your Customer",
}
_FINANCIAL_CONTEXT = re.compile(
    r"\b(?:interest|loan|bank|banking|money|finance|financial|payment|"
    r"investment|invest|deposit|account|transaction|credit|debit|"
    r"ब्याज|ऋण|बैंक|पैसा|भुगतान|निवेश|खाता|பணம்|வங்கி|முதலீடு)\b",
    re.IGNORECASE,
)
_DEFINITION_CONTEXT = re.compile(r"\b(?:what is|define|meaning of|explain)\b", re.IGNORECASE)


def normalize_financial_query(question: str) -> str:
    """Expand supported abbreviations for retrieval while preserving the question."""
    has_context = bool(_FINANCIAL_CONTEXT.search(question))
    result = question
    for abbreviation, expansion in _ABBREVIATIONS.items():
        pattern = re.compile(rf"(?<!\w){re.escape(abbreviation)}(?!\w)", re.IGNORECASE)
        if pattern.search(result) and (
            has_context or _DEFINITION_CONTEXT.search(result)
        ):
            result = pattern.sub(f"{abbreviation.upper()} ({expansion})", result)
    return result
