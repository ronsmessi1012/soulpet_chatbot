import json
import os
import faiss
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import logging
import random
from dotenv import load_dotenv
load_dotenv()

GEN_AI_API_KEY = os.getenv("API_KEY")
# IMPORTANT: Replace with your actual API key! # <---  Hardcoded API Key - USE WITH EXTREME CAUTION

# Configure Gemini API
genai.configure(api_key=GEN_AI_API_KEY)

# Logging setup
logging.basicConfig(level=logging.ERROR)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "API is running", "endpoints": ["/chat", "/suggestions"]}

# Load Q&A data
with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Predefined greetings and thank you responses
greeting_variants = ["hi", "hello", "hey", "heyy", "heyyy", "hello!!", "hi there"]
thank_you_variants = ["thanks", "thank you", "thx", "thank you so much", "thank you!", "thanks a lot"]

predefined_replies = {
    "greeting": [
        "Hi there! How can I assist you today?",
        "Hello! Looking to convert your 2D layout into 3D? I'm here to help!",
        "Hey! Feel free to ask anything about our design services."
    ],
    "thanks": [
        "You're most welcome! 😊",
        "Glad to help! Let me know if you have more questions.",
        "Anytime! Feel free to reach out again."
    ]
}

# Embedder function
def get_embedding(text: str):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        raise RuntimeError(f"Embedding error: {e}")

# Lazy FAISS initialization
questions = None
index = None

def initialize_index():
    global questions, index
    if questions is None:
        try:
            questions = [item["question"] for item in qa_data]
            question_embeddings = [get_embedding(q) for q in questions]
            dimension = len(question_embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.vstack(question_embeddings))
        except Exception as e:
            logging.error(f"Error initializing index: {e}")
            raise

# Gemini chat model
chat_model = genai.GenerativeModel("models/gemini-2.0-flash")

@app.post("/chat")
async def chat(req: Request):
    if index is None:
        try:
            initialize_index()
        except Exception as e:
            logging.error(f"Error initializing index: {e}")
            return {"error": "Failed to initialize index"}

    data = await req.json()
    user_input = data.get("user_input", "").strip().lower()

    if not user_input:
        return {"error": "No user_input provided"}

    # Handle greetings and thank yous immediately
    if any(greet in user_input for greet in greeting_variants):
        return {
            "reply": random.choice(predefined_replies["greeting"]),
            "suggested_questions": random.sample(questions, 3)
        }

    if any(thank in user_input for thank in thank_you_variants):
        return {
            "reply": random.choice(predefined_replies["thanks"]),
            "suggested_questions": random.sample(questions, 3)
        }

    try:
        user_embedding = get_embedding(user_input)
        D, I = index.search(np.array([user_embedding]).astype('float32'), k=4)

        top_match_idx = I[0][0]
        top_match = qa_data[top_match_idx]

        # Generate suggestions (excluding the top match)
        suggested_questions = []
        for idx in I[0]:
            if idx != top_match_idx and len(suggested_questions) < 3:
                suggested_questions.append(qa_data[idx]["question"])

        # Create prompt
        prompt = f"""
You are a helpful assistant. Use only the information provided in the context below to answer the user's question.

Context: {top_match['answer']}
Question: {user_input}
Answer:
"""
        response = chat_model.generate_content(prompt)

        return {
            "reply": response.text.strip(),
            "suggested_questions": suggested_questions
        }

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return {"error": "An error occurred while processing your request."}

@app.get("/suggestions")
async def get_suggestions():
    return {
        "suggested_questions": random.sample(
            [item["question"] for item in qa_data], 3
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")