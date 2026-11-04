import httpx
import streamlit as st

from finrag_indic.config import get_settings


settings = get_settings()

st.set_page_config(page_title="FinRAG-Indic", page_icon="₹", layout="centered")
st.title("FinRAG-Indic")
st.caption("Multilingual financial literacy research assistant")

st.info(
    "Phase 2 foundation is active. Document retrieval and answer generation "
    "are intentionally unavailable until later approved phases."
)

language = st.selectbox("Language", ("English", "हिन्दी", "தமிழ்"))
st.chat_input("Question answering is not implemented in Phase 2", disabled=True)

with st.sidebar:
    st.subheader("System status")
    if st.button("Check backend"):
        try:
            response = httpx.get(f"{str(settings.frontend_api_url).rstrip('/')}/health", timeout=3)
            response.raise_for_status()
            st.success(f"Backend: {response.json()['status']}")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            st.error(f"Backend unavailable: {exc}")
    st.caption(f"Selected interface language: {language}")

