from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.create_documents([text])

# Use HuggingFace for free local embeddings (like "all-MiniLM-L6-v2")
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(docs):
    embedder = get_embedder()
    return FAISS.from_documents(docs, embedder)

def get_relevant_chunks(store, query: str, k=3):
    return store.similarity_search(query, k=k)
