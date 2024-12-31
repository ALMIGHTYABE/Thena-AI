import os
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set page configuration with favicon and title
st.set_page_config(
    page_title="Athen-AI",
    page_icon="icons/thena.png",
    layout="centered"
)

# Set your Claude API Key
try:
    CLAUDE_API_KEY = st.secrets['CLAUDE_API_KEY']
except:
    CLAUDE_API_KEY = os.environ['CLAUDE_API_KEY']

if CLAUDE_API_KEY is None:
    st.error("CLAUDE_API_KEY environment variable is not set.")
    st.stop()

# Initialize the Anthropic client with the API key
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Load and chunk document
def load_and_chunk_document(file_path, delimiter="\n\n"):
    """Load and split the document by paragraphs or sections."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    chunks = content.split(delimiter)
    return chunks

# Embed texts using Sentence Transformers
def embed_texts(texts, model_name="all-mpnet-base-v2"):
    """Embed text chunks using a more powerful embedding model."""
    model = SentenceTransformer(model_name)
    return model.encode(texts)

# Build a FAISS index for efficient retrieval
def build_faiss_index(embeddings):
    """Build a FAISS index for the given embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Find the most relevant chunk using the FAISS index
def find_most_relevant_chunk(query, chunks, index, embedding_model):
    """Retrieve the top 3 most relevant chunks and combine them."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=3)
    combined_chunks = " ".join(chunks[i] for i in indices[0])
    # print(f"Retrieved chunks for query '{query}': {combined_chunks}")  # Debugging
    return combined_chunks

# Generate a response using Claude API
def generate_claude_response(context, query):
    prompt = f"""
    You are Athen-AI, providing direct information about Thena's DeFi ecosystem. Follow these rules:
    1. No greetings or questions
    2. Give only 100% certain information
    3. If information is unavailable, say you do not know.
    4. Use warm, clear language
    5. Be concise but complete

    Context: {context}
    Query: {query}
    """
    
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.dict()['content'][0]['text']
    except Exception as e:
        print(f"Error generating Claude response: {e}")
        response_text = "Sorry, there's an error with Athen-AI."
    
    return response_text

tone = '''
## Core Guidelines
- No greetings or questions
- Direct, accurate information only
- Simple, clear language
- Explain technical terms
- Warm, professional tone
- Guide users through steps when needed
- If uncertain: "I don't know"
'''

# Preprocess the document and build the index
def preprocess_document(file_path):
    """Preprocess the document by chunking and embedding its content."""
    chunks = load_and_chunk_document(file_path)
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    chunk_embeddings = embed_texts(chunks)
    index = build_faiss_index(np.array(chunk_embeddings))
    return chunks, index, embedding_model

# Load the document on the backend
document_path = "thena_docs.md"  # Change this to the path of your document
chunks, index, embedding_model = preprocess_document(document_path)

# Streamlit app layout
st.title("💬 Chat with Athen-AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("🤔 Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Find the most relevant chunk and generate a response
    relevant_chunk = find_most_relevant_chunk(prompt, chunks, index, embedding_model)
    response = generate_claude_response(relevant_chunk, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
