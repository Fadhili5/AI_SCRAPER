import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Template for question answering
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""

# Initialize LLM and embedding models
embeddings = OllamaEmbeddings(model="llama3")
llm = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)

# Load content from a webpage
def load_page(url):
    try:
        loader = SeleniumURLLoader(urls=[url])
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading page: {e}")
        return []

# Split documents into smaller chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(documents)

# Create FAISS vector store from documents
def create_vector_store(documents):
    if not documents:
        st.error("No documents to index.")
        return None
    try:
        st.write("Creating FAISS vector store...")
        faiss_store = FAISS.from_documents(documents, embedding=embeddings)
        st.write("FAISS vector store created successfully.")
        return faiss_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# Retrieve similar chunks based on the query
def retrieve_docs(query, vector_store):
    return vector_store.similarity_search(query)

# Answer question using LLM and context
def answer_question(question, context):
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("🕷️ AI Crawler")

url = st.text_input("Enter a URL to crawl:")

vector_store = None

if url:
    documents = load_page(url)
    chunked_documents = split_text(documents)

    if not chunked_documents:
        st.error("No readable content found on the page.")
        st.stop()

    st.success(f"Loaded and split {len(chunked_documents)} chunks.")
if st.checkbox("Show loaded documents"):
    for i, doc in enumerate(documents):
        st.write(f"Document{i}:\n{doc.page_content[:500]}...")
    vector_store = create_vector_store(chunked_documents)

if st.checkbox("Show document chunks"):
    for i, chunk in enumerate(chunked_documents):
        st.write(f"Chunk {i} (Size {len(chunk.page_content)}):\n{chunk.page_content[:500]}...")

question = st.chat_input("Ask something about the page...")

if question:
    st.chat_message("user").write(question)

    if vector_store:
        retrieved = retrieve_docs(question, vector_store)
        context = "\n\n".join(doc.page_content for doc in retrieved)

        answer = answer_question(question, context)
        st.chat_message("assistant").write(answer)
    else:
        st.error("Vector store was not created. Ensure you have successfully loaded and processed the documents.")