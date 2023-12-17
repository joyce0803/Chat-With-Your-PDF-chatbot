import os
import chromadb
from chromadb.config import Settings


# define the chroma settings

# CHROMA_SETTINGS = chromadb.Client(Settings(
#     chroma_db_impl = "duckdb+parquet",
#     persist_directory = "db",
#     anonymized_telemetry = False
# ))

# CHROMA_SETTINGS = chromadb.PersistentClient(
#     chroma_db_impl="new_configuration"
#     path="db", 
#     settings=Settings(anonymized_telemetry=False))