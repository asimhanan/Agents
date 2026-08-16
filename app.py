import streamlit as st
from google_doc_agent import run_agent  # reuse the agent as-is, no changes needed

st.set_page_config(page_title="Tariq Doc Agent", page_icon="📄")
st.title("📄 Tariq Doc Agent")
st.caption("Paste a public Google Doc link and ask a question about its contents.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

if prompt := st.chat_input("e.g. Read this doc and summarize it: https://docs.google.com/..."):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(prompt)
            except Exception as e:
                response = f"⚠️ Something went wrong: {e}"
        st.markdown(response)
    st.session_state.history.append(("assistant", response))
