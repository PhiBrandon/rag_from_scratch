import ollama
import litellm
import uuid
import os
import pandas as pd
from sqlalchemy import create_engine, text
from llama_index.llms.litellm import LiteLLM
from dotenv import load_dotenv

load_dotenv()
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONNECTION")

# Set up litellm callbacks and verbosity
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
litellm.set_verbose = False

# Function to generate embeddings using the ollama library
def embed_arctic(prompt):
    return ollama.embeddings(model="snowflake-arctic-embed:latest", prompt=prompt)

# Function to insert job descriptions and their embeddings into the database
def insert_embeddings(conn, dd):
    for i, d in enumerate(dd):
        if i > 200:
            break
        print(i)
        out = embed_arctic(d)
        stmt = text("""INSERT INTO embed_descriptions (description, embedding) VALUES(:description, :embedding)""")
        package = {"description": d, "embedding": out["embedding"]}
        with conn.connect() as c:
            c.execute(stmt, package)
            c.commit()

# Function to perform a similarity search on the embeddings
def similarity_search(conn):
    query = "**Federal Project - Applicant must be a United States Citizen or Permanent Residents, with the ability to obtain a Public Trust**"
    embedded_q = embed_arctic(query)
    package_1 = {"q_embed": str(embedded_q["embedding"])}
    find_stmt = text("""SELECT description from embed_descriptions ORDER BY embedding <-> :q_embed LIMIT 2""")
    with conn.connect() as c:
        result = c.execute(find_stmt, package_1)
        for r in result:
            print(r)

# Function to perform a cosine similarity search on the embeddings
def cosine_search(conn):
    query = """ What remote jobs have salaries greater than 150k """
    embedded_q = embed_arctic(query)
    package_2 = {"q_embed": str(embedded_q["embedding"])}
    find_stmt = text("""Select 1 - (embedding <=> :q_embed) as cosine_similarity, description from embed_descriptions order by cosine_similarity desc Limit 50""")
    with conn.connect() as c:
        result = c.execute(find_stmt, package_2)
        return [result, query]

# Main Area

# Read job descriptions from a CSV file
description_df = pd.read_csv("ai_engineer_2024-04-12_01-01-59-873.csv")
dd = description_df["description"].to_list()

# Connect to a PostgreSQL database
conn = create_engine(POSTGRES_CONN_STRING)

# Perform cosine similarity search
result = cosine_search(conn)

# Store the top results
top_results = []
for r in result[0]:
    print(r)
    top_results.append(r[1])

# Generate a unique identifier (UUID) for tracing purposes
current_uuid = uuid.uuid4()

# Prepare additional keyword arguments for LiteLLM models
llm_kwargs = {
    "metadata": {
        "generation_name": "rag_generation",
        "trace_name": "RAG_task_haiku_instruct",
        "version": "0.0.1",
        "trace_id": str(current_uuid),
        "input": {
            "documents": str(top_results),
            "prompt": result[1],
        },
    },
}

# Initialize LiteLLM models with different model configurations
llama_3_llm = LiteLLM(
    model="together_ai/microsoft/WizardLM-2-8x22B",
    additional_kwargs=llm_kwargs,
)
claude_llm = LiteLLM(
    model="claude-3-haiku-20240307",
    max_tokens=4000,
    additional_kwargs=llm_kwargs,
)

# Prepare the context using the top results
context = f"Context: {str(top_results)}"

# Use the claude_llm model to generate a response based on the context and the question
out = claude_llm.complete(f"{context}\nOnly use the Context to answer the question:\n {result[1]} ")

# Print the generated response
print(out)