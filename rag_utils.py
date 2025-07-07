from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Optional: for future FAISS caching
import os

# ----------------------------------------
# 1. Embedding Function
# ----------------------------------------
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------
# 2. Split Text into Chunks with Metadata
# ----------------------------------------
def split_text_with_metadata(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    for doc in docs:
        doc.metadata["source"] = source
    return docs

# ----------------------------------------
# 3. Create Vector Store (FAISS)
# ----------------------------------------
def create_vector_store(documents: list):
    embedder = get_embedder()
    return FAISS.from_documents(documents, embedder)

# ----------------------------------------
# 4. Get Top-k Relevant Chunks from Vector Store
# ----------------------------------------
def get_relevant_chunks(store, question: str, k: int = 5):
    return store.similarity_search(question, k=k)

# ----------------------------------------
# (Optional) 5. Save & Load FAISS Index
# ----------------------------------------

INDEX_PATH = "faiss_index"

def save_vector_store(store):
    store.save_local(INDEX_PATH)

def load_vector_store():
    if os.path.exists(INDEX_PATH):
        embedder = get_embedder()
        return FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    return None
