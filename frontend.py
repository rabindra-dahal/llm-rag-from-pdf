import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Gemma 3 1B PDF Chat", page_icon="📄")
st.title("📄 Chat with PDF (Gemma 3 1B + LM Studio)")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF ingestion
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Process PDF"):
        with st.spinner("Ingesting and embedding PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json()["message"])
                    st.session_state.messages = []  # Reset chat on new PDF
                else:
                    st.error(res.json().get("detail", "Upload failed."))
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend. Make sure uvicorn is running.")

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{BACKEND_URL}/chat", json={"query": prompt})
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    if sources:
                        answer += f"\n\n*Sources: Page {', '.join(map(str, sources))}*"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(res.json().get("detail", "Error generating response."))
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend.")
