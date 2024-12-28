import google.generativeai as genai
from chromadb import Documents,EmbeddingFunction,Embeddings
import os
from dotenv import load_dotenv

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input:Documents) -> Embeddings:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as environment variable")
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        return genai.embed_content(model= "models/text-embedding-004",task_type="RETRIEVAL_DOCUMENT",title="Custom query")