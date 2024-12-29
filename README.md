# gemini-rag
A Gemini based RAG

Clone the repository:
`git clone `

Setup python virtual env: 
```
virtualenv venv/
env\scripts\activate
```

Install required packages
`pip install -r requirements.txt`

create a .env file with the GOOGLE_API_KEY and CHROMA_PATH initialised
create a data folder and add pdf files as the required data source
from the root of the package run the following to create database and query

`python create_database.py`

Query the RAG model using the following:
`python query_data.py <Prompt here>`