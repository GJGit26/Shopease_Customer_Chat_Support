"""
Simple Streamlit chat UI for the ShopEase customer support bot.
Uses retrieve.py + chat_gemini.py under the hood — no changes needed to those files.

Setup:
    pip install streamlit
    (plus requirements.txt: python-dotenv, supabase, requests)

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
from chat_gemini import generate_response

st.set_page_config(page_title="ShopEase Support Bot", page_icon="🛍️")
st.title("🛍️ ShopEase Customer Support")
st.caption("RAG-powered support bot — Voyage AI embeddings + Supabase + Gemini")

# Keep chat history in session state so it persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": "..."}

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_query = st.chat_input("Apna sawaal type karo (Hindi/English/Hinglish)...")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate + show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            try:
                # Pass prior turns (excluding the one we just added) as history
                history = st.session_state.messages[:-1]
                result = generate_response(user_query, chat_history=history)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources used"):
                        for s in sources:
                            st.markdown(f"- **{s['title']}** ({s['category']}) — similarity: {s['similarity']:.2f}")

            except Exception as e:
                answer = f"Sorry, kuch error aa gaya: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar: reset button
with st.sidebar:
    st.subheader("Options")
    if st.button("🔄 Clear chat"):
        st.session_state.messages = []
        st.rerun()
