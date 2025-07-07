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
    context = ""
    used_sources = set()

    for chunk in chunks:
        context += f"{chunk.metadata.get('source', 'Unknown')}:\n{chunk.page_content}\n\n"
        used_sources.add(chunk.metadata.get("source", "Unknown"))

    prompt = f"""
You are an intelligent assistant. Answer the user's question based on the following context:

{context}

Question: {question}
Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    final_answer = response.choices[0].message.content.strip()

    # 👇 Attach sources in final response
    return f"{final_answer}\n\n(Sources: {', '.join(used_sources)})"

