import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset
df = pd.read_csv("F:/working-projects/customer_support_rag/customer_support_tickets2.csv")

# Combine subject + description + resolution into knowledge base text
df["knowledge_text"] = (
    "Subject: " + df["Ticket Subject"].fillna("") +
    "\nDescription: " + df["Ticket Description"].fillna("") +
    "\nResolution: " + df["Resolution"].fillna("")
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
embeddings = model.encode(df["knowledge_text"].tolist(), convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and dataframe
faiss.write_index(index, "tickets_index.faiss")
with open("tickets_meta.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ FAISS index built and saved!")
