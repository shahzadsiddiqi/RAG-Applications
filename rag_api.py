from fastapi import FastAPI, UploadFile, File
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

# =======================
# Config
# =======================
folder_path = r"C:/Users/WIN/Desktop/ml"  # Folder where PDFs are stored

# =======================
# Load PDFs and Build Index
# =======================
documents = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(text)

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Split all documents into smaller chunks
all_chunks = []
for doc in documents:
    all_chunks.extend(split_text(doc))

documents = all_chunks

# Embeddings and FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)
embedding_dim = document_embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(document_embeddings).astype('float32'))

def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    return [documents[i] for i in indices[0]]

# Generator Model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_answer(query):
    context_docs = retrieve(query)
    context_text = "\n".join(context_docs)

    # ========= FIX: truncate context =========
    max_chars = 1500  # ~512 tokens (safe for flan-t5-base)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]
    # ========================================

    prompt = f"Answer the question using the following context:\n{context_text}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']

# =======================
# FastAPI App
# =======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG PDF Search API is running"}

@app.post("/query")
def query_api(q: Query):
    answer = rag_answer(q.question)
    return {"question": q.question, "answer": answer}

# =======================
# Upload PDF and Index
# =======================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = os.path.join(folder_path, file.filename)

    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Read PDF and split into chunks
    reader = PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.extend(split_text(text))

    # Add to documents and FAISS
    documents.extend(text_chunks)
    new_embeddings = embedding_model.encode(text_chunks)
    index.add(np.array(new_embeddings).astype('float32'))

    return {"message": f"{file.filename} uploaded and indexed successfully!"}
