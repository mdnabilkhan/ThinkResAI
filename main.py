from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from openrouter_chat import ask_question_with_chunks
import fitz  # PyMuPDF
from rag_utils import split_text, create_vector_store, get_relevant_chunks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.post("/ask-pdf")
async def ask_from_pdf(file: UploadFile = File(...), question: str = Form(...)):
    try:
        text = extract_text_from_pdf(file)
        docs = split_text(text)
        store = create_vector_store(docs)
        top_chunks = get_relevant_chunks(store, question)
        answer = ask_question_with_chunks(top_chunks, question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
