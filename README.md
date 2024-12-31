# Athen-AI

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation and Running Locally](#installation-and-running-locally)
- [Code Structure](#code-structure)
- [Usage Examples](#usage-examples)
- [Conclusion](#conclusion)
- [License](#license)
- [Contact](#contact)

## Overview

Athen-AI is an intelligent conversational assistant designed to enhance user engagement with Thena’s DeFi ecosystem. It leverages advanced natural language processing and machine learning techniques to provide users with relevant information and assistance regarding Thena's products and services.

## Features

- **Conversational Interface**: Users can interact with Athen-AI through a chat interface, asking questions and receiving informative responses.
- **Document Processing**: The assistant can load and chunk documents, allowing it to provide contextually relevant answers based on the content of Thena's documentation.
- **Embedding and Retrieval**: Utilizes Sentence Transformers and FAISS for efficient text embedding and retrieval, ensuring quick access to relevant information.
- **Claude API Integration**: Generates responses using the Claude API, ensuring high-quality and context-aware answers.

## Installation and Running Locally

To set up the Athen-AI environment and run the application, follow these steps:

1. **Clone the Repository**:
   Open your terminal and clone the repository using the following command:
   ```bash
   git clone <repository-url>
   ```
   Replace `<repository-url>` with the actual URL of your repository.

2. **Navigate to the Project Directory**:
   Change into the project directory:
   ```bash
   cd <project-directory>
   ```
   Replace `<project-directory>` with the name of the cloned repository.

3. **Set Up the Environment**:
   Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine. Then run the setup script to create a conda environment and install the required packages:
   ```bash
   bash init_setup.sh
   ```

4. **Setting Up Environment Variables**:
   Before running the application, you need to set up the required environment variables, specifically your Claude API key. You can do this in two ways:

   - **Using Streamlit Secrets**:
     Create a file named `secrets.toml` in the `.streamlit` directory of your project (create the directory if it doesn't exist) and add the following content:
     ```toml
     [general]
     CLAUDE_API_KEY = "your_claude_api_key_here"
     ```

   - **Using Environment Variables**:
     Alternatively, set the environment variable directly in your terminal:
     ```bash
     export CLAUDE_API_KEY="your_claude_api_key_here"
     ```
     Make sure to run this command in the terminal before starting the Streamlit application.

5. **Verify the Setup**:
   You can verify that the environment variable is set correctly by running:
   ```bash
   echo $CLAUDE_API_KEY
   ```
   This should output your Claude API key. If it does not, please check your setup.

6. **Prepare Your Document**:
   Ensure you have the document you want to process (e.g., `thena_docs.md`) in the project directory or update the `document_path` variable in `main.py` to point to the correct file location.

7. **Run the Streamlit Application**:
   Start the Streamlit application by executing the following command:
   ```bash
   streamlit run main.py
   ```

8. **Open the Application in Your Browser**:
   After running the command, Streamlit will provide a local URL (usually `http://localhost:8501`). Open this URL in your web browser to access the Athen-AI chat interface.

9. **Interact with Athen-AI**:
   Type your questions in the chat input and interact with the assistant.

## Code Structure

### Main Components

- **API Key Management**: The application retrieves the Claude API key from environment variables or Streamlit secrets.
- **Document Loading and Chunking**: The `load_and_chunk_document` function reads a document and splits it into manageable chunks for processing.
- **Text Embedding**: The `embed_texts` function uses Sentence Transformers to convert text chunks into embeddings for efficient retrieval.
- **FAISS Indexing**: The `build_faiss_index` function creates an index for the embeddings, allowing for quick searches.
- **Query Processing**: The `find_most_relevant_chunk` function retrieves the most relevant document chunks based on user queries.
- **Response Generation**: The `generate_claude_response` function interacts with the Claude API to generate responses based on the context and user query.

### Streamlit Interface

- The Streamlit app is initialized with a title and chat interface.
- User messages are stored in the session state, allowing for a continuous conversation.
- The assistant's responses are generated based on the most relevant document chunks retrieved from the FAISS index.

## Usage Examples

Here are some example questions you can ask Athen-AI:
- "What is the purpose of the THENA ecosystem?"
- "How do I provide liquidity on THENA?"
- "Can you explain the tokenomics of THE?"

## Conclusion

Athen-AI serves as a powerful tool for enhancing user engagement within the Thena ecosystem. By leveraging advanced technologies and adhering to specific communication guidelines, it aims to provide users with a seamless and informative experience in the DeFi space.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please reach out to me.