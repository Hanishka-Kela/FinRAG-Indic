import streamlit as st

from finrag_indic.config import get_settings
from finrag_indic.frontend_client import BackendClient, BackendClientError


settings = get_settings()
client = BackendClient(
    str(settings.frontend_api_url), timeout=settings.frontend_request_timeout_seconds
)
language_options = {"English": "en", "हिन्दी": "hi", "தமிழ்": "ta"}

st.set_page_config(page_title="FinRAG-Indic", page_icon="₹", layout="centered")
st.title("FinRAG-Indic")
st.caption(
    "A multilingual financial-literacy chatbot grounded in official Indian financial sources"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Chat settings")
    language_label = st.selectbox("Response language", tuple(language_options))
    selected_language = language_options[language_label]
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("Backend status")
    try:
        backend_status = client.status()
        if backend_status["status"] == "ready":
            st.success("Connected and ready")
        else:
            st.warning("Connected with limited functionality")
        st.caption(f"Provider: {backend_status['llm_provider']}")
        availability = (
            "available" if backend_status["retrieval_index_available"] else "missing"
        )
        st.caption(f"Retrieval index: {availability}")
    except (BackendClientError, KeyError):
        st.error("Backend unavailable. Start FastAPI and try again.")


def show_sources(message: dict[str, object]) -> None:
    citations = message.get("citations", [])
    evidence = {
        item.get("citation_id"): item
        for item in message.get("evidence", [])  # type: ignore[union-attr]
        if isinstance(item, dict)
    }
    if not citations:
        return
    with st.expander("Sources"):
        for citation in citations:  # type: ignore[union-attr]
            if not isinstance(citation, dict):
                continue
            citation_id = citation.get("citation_id", "Source")
            page = citation.get("page_number")
            page_label = f", page {page}" if page else ""
            st.markdown(
                f"**[{citation_id}] {citation.get('title', 'Official source')}"
                f"{page_label}**  \n[{citation.get('document_id')}]"
                f"({citation.get('source_url')})"
            )
            passage = evidence.get(citation_id, {}).get("passage")
            if passage:
                st.caption(str(passage))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("needs_clarification"):
                st.info("A clarification is needed before giving personalized guidance.")
            show_sources(message)

question = st.chat_input("Ask a financial-literacy question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving official sources and preparing an answer..."):
            try:
                result = client.chat(question=question, language=selected_language)
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                    "evidence": result.get("evidence", []),
                    "needs_clarification": result.get("needs_clarification", False),
                }
                st.markdown(result["answer"])
                if result.get("abstained"):
                    if result.get("needs_clarification"):
                        st.info("A clarification is needed before giving personalized guidance.")
                    else:
                        st.info(
                            "The assistant abstained because grounded evidence was insufficient."
                        )
                show_sources(assistant_message)
                latency = result.get("latency", {})
                if latency.get("total_ms") is not None:
                    st.caption(f"Total latency: {latency['total_ms']:.0f} ms")
                st.session_state.messages.append(assistant_message)
            except BackendClientError as exc:
                retry = " You can try again." if exc.retryable else ""
                error_message = f"Unable to answer: {exc}.{retry}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "citations": [],
                        "evidence": [],
                    }
                )
