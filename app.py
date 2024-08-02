import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        return data

# This will load the PDF file
def load_pdf_data(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        ingestor = FileIngestor(pdf)
        documents.extend(ingestor.handlefileandingest())
    return documents

# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

# Function for loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': normalize_embedding}
    )

# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={'prompt': prompt}
    )

# Prettifying the response
def get_response(query, chain):
    response = chain({'query': query})
    return response["result"]

def user_input(user_question, embedding_model, api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if 'vectorstore' not in st.session_state:
        st.error("No PDF data available. Please upload and process PDF files first.")
        return

    vectorstore = st.session_state.vectorstore
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.3)

    chain = load_qa_chain(retriever, llm, prompt)
    response = get_response(user_question, chain)
    st.write("Reply: ", response)

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = load_pdf_data(pdf_docs)
                text_chunks = split_docs(raw_text)
                vectorstore = create_embeddings(text_chunks, embed)
                st.session_state.vectorstore = vectorstore
                st.success("Done")
        else:
            st.error("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and api_key:
        if 'vectorstore' in st.session_state:
            user_input(user_question, embed, api_key)
        else:
            st.error("Please process PDF files first to create the FAISS index.")

if __name__ == "__main__":
    main()
