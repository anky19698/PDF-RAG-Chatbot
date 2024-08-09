# PDF-based RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of uploaded PDF documents. The chatbot uses natural language processing techniques to extract relevant information from PDFs and generate responses using a language model.

## Features

- PDF upload functionality
- Text extraction and chunking from PDFs
- Similarity search using FAISS vector store
- Integration with Hugging Face models for text embedding and generation
- Streamlit-based user interface for easy interaction
- Chat history management

## Requirements

- Python 3.7+
- Streamlit
- Langchain
- Hugging Face Transformers
- SpaCy
- PDFMiner
- FAISS

## Installation

1. Clone this repository:
  `git clone https://github.com/yourusername/pdf-rag-chatbot.git
  cd pdf-rag-chatbot`
2. Install the required packages:
   `pip install -r requirements.txt`
3. Download the SpaCy model:
   `python -m spacy download en_core_web_sm`
4. Set up your Hugging Face API token in Streamlit secrets.

## Usage

1. Run the Streamlit app:
  `streamlit run app.py`
2. Upload a PDF file using the file uploader.

3. Ask questions about the PDF content using the chat interface.

4. Clear chat history using the "Clear Chat History" button when needed.

## How it Works

1. The app extracts text from the uploaded PDF and splits it into manageable chunks.
2. These chunks are embedded and stored in a FAISS vector store for efficient similarity search.
3. When a user asks a question, the app finds the most relevant text chunks.
4. The question, relevant context, and chat history are sent to a language model to generate a response.
5. The response is displayed to the user, and the conversation history is updated.

## Configuration

- Adjust the `max_characters` parameter in the `get_chunks` function to change the chunk size.
- Modify the `model_id` in the `get_llm_model` function to use a different Hugging Face model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
