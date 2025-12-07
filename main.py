import os
import sys

# Import LangChain modules as requested in the assignment hints
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

def main():
    # 1. Load the provided text file
    print("Loading data...")
    try:
        loader = TextLoader("./speech.txt")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading speech.txt: {e}")
        return

    # 2. Split the text into manageable chunks
    print("Splitting text...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings and store them in a local vector store (ChromaDB)
    # Using sentence-transformers/all-MiniLM-L6-v2 as requested
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the vector store locally
    db = Chroma.from_documents(texts, embeddings)
    
    # 4. Initialize Ollama with Mistral 7B
    print("Initializing Ollama (Mistral)...")
    llm = Ollama(model="mistral:latest")

    # 5. Create a simple retrieval chain
    print("Building RAG pipeline...")
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    def get_answer(query):
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}"
        return llm.invoke(prompt)

    print("\n---Initialized ---")
    print("Ask a question based on the speech (Type 'exit' to quit):")

    # Interactive Loop
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            # Retrieve chunks and generate answer
            response = get_answer(query)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()