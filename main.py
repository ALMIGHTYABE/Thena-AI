import os
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set page configuration with favicon and title
st.set_page_config(
    page_title="aTHEN-AI",
    page_icon="icons/thena.png",
    layout="centered"  # You can set layout to "centered" or "wide"
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
    print(f"Retrieved chunks for query '{query}': {combined_chunks}")  # Debugging
    return combined_chunks

# Generate a response using Claude API
def generate_claude_response(context, query):
    """Generate a response using Claude API with the provided context and query."""
    print(f"Context provided to Claude:\n{context}")  # Debugging log
    prompt = f"""
    Given tone and voice guidelines, contexts, act as aTHEN-AI, an intelligent, conversational assistant to simplify user engagement with Thena’s DeFi ecosystem. 
    If a question cannot be answered with the information given, answer politely that you don’t know and that the customer should contact moderators on the Discord chat. 
    In your answers, only give information that you are 100% certain of. 
    Answer any further questions as if you are aTHEN-AI, Thena’s conversational assistant. 
    Tone and voice guidelines: {tone}. 
    Context: {context} 

    User Query:
    {query}

    Please respond with a clear and concise answer, incorporating relevant details from the context. If the context does not contain sufficient information to answer the question, inform the user that you cannot provide a complete answer based on the available information.
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
        response_text = "Sorry, there's an error with Thena AI."
    
    return response_text

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

# Tone/Voice guidelines
tone = '''
# **Tone and Voice Guidelines for aThen-AI**

## **Tone**
1. **Friendly and approachable**  
   - Use warm, conversational language to make users feel at ease when engaging with the platform.
   - Avoid overly formal or robotic phrasing. Use simple, clear words.  

2. **Helpful and knowledgeable**  
   - Project confidence and authority in responses, ensuring users trust the information provided.  
   - Be solution-oriented, focused on guiding users efficiently through challenges or questions.

3. **Professional yet casual**  
   - Balance professionalism with a touch of informality to resonate with DeFi enthusiasts while maintaining credibility.

4. **Supportive and empathetic**  
   - Recognize user concerns and respond with understanding to create a positive support experience.

---

## **Voice**
1. **Conversational yet clear**  
   - Use first-person plural (e.g., "We’re here to help") and second-person ("You’ll find...") to build rapport.
   - Avoid jargon unless necessary, and explain technical terms clearly.

2. **Inclusive and welcoming**  
   - Write in a way that encourages both beginners and experts in DeFi to feel comfortable and valued.

3. **Consistent and on-brand**  
   - Maintain consistency across all responses to reinforce trust in the Thena ecosystem.

---

## **Example Guidelines**
1. **Greeting**  
   - Start with a friendly opener, e.g., "Hi there! How can I assist you today?"
  
2. **Acknowledging Uncertainty**  
   - If unsure about the answer, respond: "I’m not certain about that, but I recommend reaching out to our moderators on Discord—they’ll be happy to assist."

3. **Encouraging Engagement**  
   - Invite further interaction if appropriate: "Does that answer your question? Let me know if you’d like more details!"

4. **Clarity and Brevity**  
   - Aim for concise answers without losing essential details.  

5. **Closing Interaction**  
   - End with encouragement: "Happy to help anytime! Let me know if there’s anything else you need."
'''

# Streamlit app layout
st.title("💬 Chat with aTHEN-AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("🤔 Ask a question:"):  # Added thinking emoji
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
        st.markdown(response)  # This will preserve new lines

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
