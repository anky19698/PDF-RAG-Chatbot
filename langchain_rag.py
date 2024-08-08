from langchain.vectorstores import Chroma
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import spacy
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
import streamlit as st
import base64


def get_chunks(pdf_path, max_characters):
    
    pdf_text = extract_text(pdf_path)
    nlp = spacy.load("en_core_web_sm")
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
def create_chromadb(_documents):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = Chroma(embedding_function=embedding_model)
    vector_store.add_documents(documents=_documents)
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

def generate_response(query, context, model, chat_history, samples):
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
    
def get_samples():
    samples = [
    {
        "question": "What white wines do you have?",
        "answer": "We offer the following wines: The Jessup Cellars 2022 Chardonnay and the 2023  Sauvignon Blanc"
    },
    {
        "question": "What red wines is Jessup Cellars offering in 2024?",
        "answer": "Jessup Cellars offers a number of red wines across a range of varietals, from Pinot Noir and Merlot blends from the Truchard Vineyard, to blended Cabernet Sauvignon from both the Napa and Alexander Valleys, our Mendocino Rougette combining Grenache and Carignane varietals which we refer to as our 'Summer Red', to the ultimate expression of the 'Art of the Blend\" with our Juel and Table for Four Red Wines. We also offer 100% Zinfandel from 134 year old vines in the Mendocino region and our own 100% Petite Sirah grown in the Wooden Valley in Southeastern Napa County. We also offer some seasonal favorites, led by the popular whimsical Manny's Blend which should be released later in 2024 with a very special label."
    },
    {
        "question": "Please tell me more about your consulting winemaker Rob Lloyd?",
        "answer": "Rob Lloyd \nConsulting Winemaker\n\nBIOGRAPHY\nHometown: All of California\n\nFavorite Jessup Wine Pairing: Carneros Chardonnay with freshly caught Mahi-Mahi\n\nAbout: Rob\u2019s foray into wine started directly after graduating college when he came to Napa to work in a tasting room for the summer \u2013 before getting a \u2018real job\u2019. He became fascinated with wine and the science of winemaking and began to learn everything he could about the process.\n\nWhile interviewing for that \u201creal job\u201d, the interviewer asked him what he had been doing with his time since graduation. After speaking passionately and at length about wine, the interviewer said, \u201cYou seem to love that so much. Why do you want this job?\u201d Rob realized he didn't want it, actually. He thanked the man, and thus began a career in the wine industry.\n\nRob has since earned his MS in Viticulture & Enology from the University of California Davis and worked for many prestigious wineries including Cakebread, Stag\u2019s Leap Wine Cellars, and La Crema. Rob began crafting Jessup Cellars in the 2009 season and took the position of Director of Winemaking at Jessup Cellars in 2010. He now heads up our winemaking for the Good Life Wine Collective, which also includes Handwritten Wines."
    },
    {
        "question": "Tell me an interesting fact about Rob Llyod",
        "answer": "While interviewing for that \u201creal job\u201d, the interviewer asked him what he had been doing with his time since graduation. After speaking passionately and at length about wine, the interviewer said, \u201cYou seem to love that so much. Why do you want this job?\u201d Rob realized he didn't want it, actually. He thanked the man, and thus began a career in the wine industry."
    },
    {
        "question": "Are walk ins allowed?",
        "answer": " Walk-ins are welcome anytime for our Light Flight experience. Our most popular tastinig experience is the jessup Classic Tasting, which includes a flight of 5 wines perfectly paired with cheeses, accompanied by palate cleansing Marcona almonds and a chocolate surprise. The Classic Tasting is $60 perperson, but is waived with a purchase of two or more bottles of wine per person."
    }
    ]

    return samples

def main():
    # Title of the Streamlit app
    st.title("PDF-based RAG Chatbot")

    # File uploader widget to allow users to upload a PDF file
    uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")

    # Check if a PDF file has been uploaded
    if uploaded_file is not None:

        # Display PDF
        pdf_contents = uploaded_file.read()
        user_pdf_base64 = base64.b64encode(pdf_contents).decode('utf-8')
        st.write(f'<iframe src="data:application/pdf;base64,{user_pdf_base64}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)
        st.write("")
        
        # Extract chunks of text from the uploaded PDF
        chunks = get_chunks(uploaded_file, 1000)
        
        # Initialize the language model for generating responses
        model = get_llm_model()
        
        # Create or load a vector store to handle document embeddings
        vector_store = create_chromadb(chunks)
        
        # Define sample Q&A pairs for testing purposes
        samples = get_samples()

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
                    results = vector_store.similarity_search_with_relevance_scores(query=query, k=3)
                    # Extract the content of the most relevant chunk
                    corpus = results[0][0].page_content

                    # Generate a response using the language model
                    response = generate_response(query=query, context=corpus, model=model, chat_history=st.session_state.chat_history, samples=samples)
                    # Display the generated response
                    st.write(response)    

                    # Add the assistant's response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})


# Run the main function if this script is executed
if __name__ == '__main__':
    main()


    
