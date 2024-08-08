from langchain.vectorstores import FAISS
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import spacy
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
import streamlit as st
import base64


def get_chunks(pdf_path, max_characters):
    
    pdf_text = extract_text(pdf_path)
    # Determine the path to the SpaCy model
    spacy_model_path = "./spacy_models/en_core_web_sm/en_core_web_sm-3.7.1"
    # spacy_model_path = "./spacy_models/en_core_web_sm/en_core_web_sm-3.7.1"
    
    # Load the SpaCy model from the local path
    nlp = spacy.load(spacy_model_path)
    
    doc = nlp(pdf_text)
    sentences =  [sent.text.strip() for sent in doc.sents]
    combined_chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence exceeds the max_characters limit
        if len(current_chunk) + len(sentence) + 1 > max_characters:
            # If current_chunk is not empty, add it to the list
            if current_chunk:
                combined_chunks.append(Document(page_content=current_chunk.strip()))
                # Start a new chunk
                current_chunk = sentence
            else:
                # If the sentence itself is longer than the max limit, add it directly
                combined_chunks.append(Document(page_content=sentence.strip()))
        else:
            # Add sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        combined_chunks.append(Document(page_content=current_chunk.strip()))

    return combined_chunks

@st.cache_resource
def create_faiss_db(_documents):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(documents=_documents, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_llm_model():
    # Set up Hugging Face Token
    hf_token = st.secrets['hf_token']
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

    # Set up the LLM model
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_id = "openai-community/gpt2-xl"
    hf_endpoint = HuggingFaceEndpoint(
        endpoint_url=model_id,
        task="text-generation",
        temperature= 1,
        max_new_tokens= 800,
        token=hf_token
    )

    return hf_endpoint

def generate_response(query, context, model, chat_history):
    try:
        recent_history = chat_history[-5:]
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    except:
        history_text = ""

    input_prompt = f"""
    You are a helpful assistant. Your task is to answer user questions based solely on the information provided in the following corpus. If the question cannot be answered using this corpus, politely inform the user to try again.

    Corpus:
    {context}

    Chat History:
    {history_text}

    Remember:
    1. Only use information from the provided corpus to answer questions.
    2. If the question cannot be answered using the corpus, politely direct the customer to contact the business.
    3. Provide concise yet informative answers.
    4. Consider Chat History to keep up with Context
    5: Answer in 3-5 Lines and Don't Include Any Punctuations or text before the Actual Answer!

    User Question: {query}
    """

    response = model.invoke(input_prompt)
    return response.strip()
    

def main():
    # Title of the Streamlit app
    st.title("PDF-based RAG Chatbot")

    # File uploader widget to allow users to upload a PDF file
    uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")

    # Check if a PDF file has been uploaded
    if uploaded_file is not None:

        # # Display PDF
        # pdf_contents = uploaded_file.read()
        # user_pdf_base64 = base64.b64encode(pdf_contents).decode('utf-8')
        # st.write(f'<iframe src="data:application/pdf;base64,{user_pdf_base64}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)
        # st.write("")
        
        # Extract chunks of text from the uploaded PDF
        chunks = get_chunks(uploaded_file, 1000)
        
        # Initialize the language model for generating responses
        model = get_llm_model()
        
        # Create or load a vector store to handle document embeddings
        vector_store = create_faiss_db(chunks)

        # Initialize chat history in the session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            # Reset the chat history to an empty list
            st.session_state['chat_history'] = []
            # Refresh the app to reflect the cleared chat history
            # st.rerun()

        # Display the chat history
        for i, message in enumerate(st.session_state.chat_history):
            # Use Streamlit's chat message function to render each message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input box for users to type their queries
        query = st.chat_input("Ask a question about the PDF content:")

        if query:
            # Add the user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Display the user question in the chat interface
            with st.chat_message("user"):
                st.markdown(query)

            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    # Perform a similarity search to find relevant chunks of text
                    results = vector_store.similarity_search_with_score(query=query, k=3)
                    # Extract the content of the most relevant chunk
                    corpus = results[0][0].page_content

                    # Generate a response using the language model
                    response = generate_response(query=query, context=corpus, model=model, chat_history=st.session_state.chat_history)
                    # Display the generated response
                    st.write(response)    

                    # Add the assistant's response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})


# Run the main function if this script is executed
if __name__ == '__main__':
    main()


    
