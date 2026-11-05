from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_chat_shell_renders_without_backend() -> None:
    app_path = Path(__file__).parents[2] / "frontend" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "FinRAG-Indic"
    assert app.chat_input[0].placeholder == "Ask a financial-literacy question"
    assert app.selectbox[0].label == "Response language"
    assert app.button[0].label == "Clear chat"
