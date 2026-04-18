import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# Use a HuggingFace embedding model so we don't consume Groq/OpenAI credits for embeddings.
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "faiss_index"
GUIDELINES_PATH = "medical_guidelines.txt"

def build_vector_store():
    print("Building local vector store from guidelines...")
    if not os.path.exists(GUIDELINES_PATH):
        raise FileNotFoundError(f"{GUIDELINES_PATH} not found.")

    loader = TextLoader(GUIDELINES_PATH)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n\n")
    docs = text_splitter.split_documents(documents)

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Create the FAISS vector store and save to disk
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("Vector store built successfully.")
    return vectorstore

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    
    if not os.path.exists(VECTOR_STORE_PATH):
        vectorstore = build_vector_store()
    else:
        # Load from disk if it exists
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
    return vectorstore.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    # Test vector store creation
    build_vector_store()
