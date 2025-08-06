import streamlit as st
from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

Prompt_TEMPLATE = """ you are helpful assistant.use the provided context to answer the query. If you don't know answer state that ypu don't know .be concise and fectual.
query:{user_query}
context:{document_context}
Answer:
"""

PDF_storage_path = "document_store/pdfs/"
Embedding_Model = OllamaEmbeddings(model="deepseek-r1:1.5b")
Document_vector_db = InMemoryVectorStore(Embedding_Model)
language_Model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.9)

def save_uploaded_file(uploaded_file):
    file_path = PDF_storage_path + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def index_documents(chunks):
    Document_vector_db.add_documents(chunks)

def find_related_documents(query):
    return Document_vector_db.similarity_search(query)

def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(Prompt_TEMPLATE)
    response_chain = prompt | language_Model
    return response_chain.invoke({"user_query": query, "document_context": context})

st.title("my app")
st.markdown("ask anything from this app")
uploaded_PDF = st.file_uploader("upload your desired pdf from which you want to ask question",
                                type="pdf")
if uploaded_PDF:
    file_path = save_uploaded_file(uploaded_PDF)
    raw_docs = load_pdf_documents(file_path)
    chunks = chunk_documents(raw_docs)
    index_documents(chunks)
    st.success("Documents process successfully")

user_query = st.chat_input("Ask a question")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    with st.spinner("analyzing"):
        related_docs = find_related_documents(user_query)
    answer = generate_answer(user_query, related_docs)
    with st.chat_message("assistant", avatar = "🤖"):
        st.write(answer)