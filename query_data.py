from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")
PROMPT_TEMPLATE = """
Please answer this: {question} based on the following context:

{context}

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_fn = GoogleGenerativeAIEmbeddings(model= "models/text-embedding-004")
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_fn)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results[0][1])
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find relevant result")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Prompt: {prompt}")

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response_text = model.generate_content(prompt).text

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()