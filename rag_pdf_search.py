"""
RAG PDF Search
-------------
This script:
1. Loads PDFs from a local folder
2. Splits them into chunks
3. Embeds them with Sentence Transformers
4. Stores embeddings in FAISS
5. Retrieves top-k chunks for a query
6. Uses a Hugging Face model to answer based on retrieved context

Dependencies (install before running):
    pip install faiss-cpu sentence-transformers transformers pypdf
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

# =======================
# 1. Load PDFs from Folder
# =======================
# Change this to the folder where your PDFs are stored
folder_path = r"C:/Users/WIN/Desktop/ml"

documents = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(text)

print(f"Loaded {len(documents)} documents (pages) from PDFs.")

# =======================
# 2. Split into Chunks
# =======================
def split_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for doc in documents:
    all_chunks.extend(split_text(doc))

documents = all_chunks
print(f"After chunking: {len(documents)} chunks")

# =======================
# 3. Create Embeddings
# =======================
print("Creating embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)
embedding_dim = document_embeddings.shape[1]

# =======================
# 4. Store in FAISS Index
# =======================
print("Storing in FAISS index...")
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(document_embeddings).astype('float32'))

# =======================
# 5. Retrieval Function
# =======================
def retrieve(query, top_k=3):
    """Retrieves top_k most relevant chunks for a query."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    return [documents[i] for i in indices[0]]

# =======================
# 6. Hugging Face Generator
# =======================
print("Loading Hugging Face generator...")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_answer(query):
    """Generates an answer to the query using retrieved context."""
    context_docs = retrieve(query)
    context_text = "\n".join(context_docs)
    
    prompt = f"Answer the question using the following context:\n{context_text}\nQuestion: {query}\nAnswer:"
    
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']

# =======================
# 7. Test Query
# =======================
if __name__ == "__main__":
    query = "what is supervised learning ?"
    print("Query:", query)
    print("Answer:", rag_answer(query))
