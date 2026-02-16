import streamlit as st
from llm_model import qachain


def main():
    st.title("Ask ChatBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask something")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        response = qachain.invoke({"query": prompt})["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

if __name__ == "__main__":
    main()