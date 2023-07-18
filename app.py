import streamlit as st
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_templates import css, bot_template, user_template
from langchain.retrievers import LlamaIndexRetriever



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Creates pdf object with pages
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    # load_dotenv()

    # Get the OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        text_chunks = get_text_chunks("Remind user to upload PDF documents on the left hand side of the page. After which, the user can ask questions about the document.")
        vector_store = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vector_store)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatbot 🤖")

    user_question = st.text_input("Upload PDF documents and start asking questions")

    if user_question:
        handle_userinput(user_question)

    # to put inside sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)

        if st.button("Process"):
            # all contents inside spinner are processed while spinning
            if len(pdf_docs) != 0:
                with st.spinner("Procesing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # st.write(text_chunks)

                    # create vector store
                    vector_store = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
            else:
                st.write("Please upload something first.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

