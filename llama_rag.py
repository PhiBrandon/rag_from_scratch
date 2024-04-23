import ollama
import litellm
import uuid
import psycopg2
import os
import pandas as pd
from sqlalchemy import make_url
from llama_index.llms.litellm import LiteLLM
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from dotenv import load_dotenv


jobs = pd.read_csv("1713798254.834989-combined-data_scientist.csv")
job_data = jobs["description"].tolist()

load_dotenv()
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONNECTION")


# Set up litellm callbacks and verbosity
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
litellm.set_verbose = False


# Create documents from job_description list
job_documents = []
for desc in job_data:
    current_doc = Document(text=desc)
    job_documents.append(current_doc)

job_documents[0]

# Connect to database
db_name = "llama_index_vector_db"
conn = psycopg2.connect(POSTGRES_CONN_STRING)
conn.autocommit = True

# Create the db for vectors
with conn.cursor() as c:
    c.execute(f"CREATE DATABASE {db_name}")

# Create a URL from DB string, and initialize Llama Index vectorstore
url = make_url(POSTGRES_CONN_STRING)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="data_scientist_jobs",
    embed_dim=1024,
)

# Create storage context to be referenced for this vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Ollama Arctic Embeddings
ollama_embedding = OllamaEmbedding(model_name="snowflake-arctic-embed:latest")
Settings.embed_model = ollama_embedding

index = VectorStoreIndex.from_documents(
    job_documents, storage_context=storage_context, show_progress=True
)

# Meat of the work
# Kwargs for logging purposes
current_uuid = uuid.uuid4()
llm_kwargs = {
    "metadata": {
        "generation_name": "rag_generation",
        "trace_name": "RAG_task_phi3",
        "version": "0.0.1",
        "trace_id": str(current_uuid),
    },
}

# LiteLLM LLama 3 Model LLM
""" llm = LiteLLM(
    model="together_ai/meta-llama/Llama-3-70b-chat-hf",
    additional_kwargs=llm_kwargs,
) """

""" llm = LiteLLM(
    model="claude-3-haiku-20240307",
    additional_kwargs=llm_kwargs,
) """

llm = LiteLLM(
    model="ollama/phi3:instruct",
    additional_kwargs=llm_kwargs,
)

Settings.llm = llm
# Create an index from our documents
# Create query engine from the index
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("Remote jobs between 150k-200k")
print(response.response)
