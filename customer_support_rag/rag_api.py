import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load index and metadata
index = faiss.read_index("tickets_index.faiss")
with open("tickets_meta.pkl", "rb") as f:
    df = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM (change to GPT if you want)
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def retrieve(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)
    results = [df.iloc[i]["knowledge_text"] for i in indices[0]]
    return results

def rag_answer(query):
    context = "\n\n".join(retrieve(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in a helpful way:"
    response = generator(prompt, max_length=200, do_sample=True)[0]['generated_text']
    return response

if __name__ == "__main__":
    while True:
        q = input("Ask a support question: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("🤖", rag_answer(q))
