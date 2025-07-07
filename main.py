from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

import fitz  # PyMuPDF

from openrouter_chat import ask_question_with_chunks
from rag_utils import (
    split_text_with_metadata,
    create_vector_store,
    get_relevant_chunks,
    load_vector_store,
    save_vector_store
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extract text from a single PDF file using PyMuPDF
def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ Multi-PDF Q&A Route with source metadata
@app.post("/ask-multi-pdf")
async def ask_from_multiple_pdfs(
    files: List[UploadFile] = File(...),
    question: str = Form(...)
):
    try:
        all_docs = []

        for file in files:
            filename = file.filename
            text = extract_text_from_pdf(file)
            docs = split_text_with_metadata(text, source=filename)
            all_docs.extend(docs)
        store = load_vector_store()
        if store is None:
            store = create_vector_store(all_docs)
            save_vector_store( store)
        top_chunks = get_relevant_chunks(store, question)
        answer = ask_question_with_chunks(top_chunks, question)

        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
