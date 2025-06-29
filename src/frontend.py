import streamlit as st
import requests
import time

uploaded_files = st.file_uploader(
    "Upload PDF files", accept_multiple_files=True
)

if st.button("Upload") and uploaded_files:
    files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
    requests.post("http://127.0.0.1:8000/upload", files=files)



# Streamed response emulator
def response_generator(query):

    res = requests.post("http://127.0.0.1:8000/generate", json={"query":query})
    response = res.json()["response"]

    for word in response.split():
        yield word + " "
        time.sleep(0.3)

st.title("Simple RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# to start, streamlit run frontend.py