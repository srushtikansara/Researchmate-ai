import sys
sys.path.insert(0, ".")

print("Step 1: Loading embeddings...")
from modules.embeddings import get_embedding_model
model = get_embedding_model()
print("✅ Embeddings OK")

print("\nStep 2: Loading vector store...")
from modules.vector_db import get_vector_store
vs = get_vector_store()
print(f"✅ Indexed files: {list(vs._indexed_files)}")

print("\nStep 3: Testing retrieval...")
from modules.retriever import get_retriever
retriever = get_retriever()
result = retriever.retrieve("summarize the paper")
print(f"✅ Retrieved {len(result.chunks)} chunks")

print("\nStep 4: Testing LLM...")
from modules.rag_chain import get_rag_chain
rag = get_rag_chain()
response = rag.answer("Summarize the paper in one sentence.")
print(f"Answer: {response.answer}")
print(f"Error: {response.error}")