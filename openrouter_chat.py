from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "meta-llama/llama-3-8b-instruct:free"


def ask_question_with_chunks(chunks: list, question: str) -> str:
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "Answer only using the document context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
