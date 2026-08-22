from pypdf import PdfReader
import chromadb
import ollama

# CONFIG 
PDF_PATH = "test_company.pdf"  
CHUNK_SIZE = 500            
CHUNK_OVERLAP = 50           
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Parse PDF 
def parse_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# Chunk text 
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Embed a piece of text 
def embed(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

# Build the vector store 
def build_vector_store(chunks):
    client = chromadb.Client()  # in-memory, resets each run
    collection = client.create_collection(name="rag_collection")

    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[chunk]
        )
    return collection

# Retrieve relevant chunks 
def retrieve(collection, question, top_k=3):
    q_vector = embed(question)
    results = collection.query(query_embeddings=[q_vector], n_results=top_k)
    return results["documents"][0]

# Generate answer 
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question based only on the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# Main

def main():
    print("Parsing PDF...")
    text = parse_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks and building vector store...")
    collection = build_vector_store(chunks)

    print("\nReady! Ask questions about the PDF (type 'quit' to exit)\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            break

        relevant_chunks = retrieve(collection, question)
        answer = generate_answer(question, relevant_chunks)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()