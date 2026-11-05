from finrag_indic.evaluation.query_normalization import normalize_financial_query


def test_financial_abbreviation_is_expanded_case_insensitively() -> None:
    assert normalize_financial_query("What is si?") == (
        "What is SI (Simple Interest)?"
    )


def test_contextual_abbreviation_expansion_preserves_original_words() -> None:
    result = normalize_financial_query("Explain EMI and FD for a bank loan.")
    assert "EMI (Equated Monthly Instalment)" in result
    assert "FD (Fixed Deposit)" in result
    assert "Explain" in result


def test_unrelated_word_is_not_replaced() -> None:
    assert normalize_financial_query("The signal is strong.") == "The signal is strong."
