import json
import logging

from finrag_indic.logging import JsonFormatter


def test_json_formatter_preserves_unicode_and_context() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="தமிழ் हिन्दी",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "தமிழ் हिन्दी"
    assert payload["request_id"] == "request-1"
    assert payload["level"] == "INFO"

