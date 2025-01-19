import os
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from dotenv import load_dotenv
from bs4 import SoupStrainer

# Load environment variables (like API keys) from a .env file.
load_dotenv()

# Cache the document loading process so it doesn't repeat unnecessarily.
@st.cache_data
def load_document_loader():
    # Load content from the specified Wikipedia page using WebBaseLoader.
    loader = WebBaseLoader(
        'https://en.wikipedia.org/wiki/Elon_Musk',  # URL to fetch content from.
        bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output')))  # Extract specific content.
    )
    documents = loader.load()  # Load the content.
    
    # Break the loaded content into smaller chunks for easier processing.
    recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = recursive.split_documents(documents)
    return chunks

# Load and process the content into chunks.
chunks = load_document_loader()

# Initialize an embedding model (used to convert text into numerical representations for searching).
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Get the Qdrant API key from environment variables.
api_key = os.getenv('qdrant_api_key')

# Specify the URL for the Qdrant database.
url = 'https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333'

# Connect to an existing Qdrant collection to store and retrieve embeddings.
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,  # Embedding model to use.
    url=url,          # Qdrant database URL.
    api_key=api_key,  # API key for authentication.
    prefer_grpc=True, # Use gRPC for faster performance.
    collection_name="Elon Muske"  # Collection name in Qdrant.
)

# Initialize Google's generative AI model for generating responses.
google_api = os.getenv('google_api_key')
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Set up a retriever to fetch relevant content from the Qdrant database.
num_chunks = 5  # Number of chunks to retrieve.
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

# Helper function to format the retrieved content.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define a prompt template for generating chatbot responses.
prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements.
Context: {context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Define the workflow for processing user queries.
query_fetcher = itemgetter("question")  # Extract the question from user input.
setup = {"question": query_fetcher, "context": query_fetcher | retriever | format_docs}  # Combine steps.
_chain = setup | _prompt | llm | StrOutputParser()  # Define the complete processing chain.

# Streamlit application UI starts here.

# Set the title of the Streamlit app.
st.title("Ask Anything About Elon Musk")

# Create a container for displaying chat history.
voice_recording_column, send_button_column = st.columns(2)
chat_container = st.container()
with voice_recording_column:
    voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
with send_button_column:
    send_button = st.button("Send", key="send_button", on_click=clear_input_field)

if voice_recording:
    transcribed_audio = transcribe_audio(voice_recording["bytes"])
    query=print(transcribed_audio)


# Initialize session state to keep track of chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store user and AI messages.

# Function to trigger input processing.
def send_input():
    st.session_state.send_input = True

# Input section for user queries.
with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)  # Input box for questions.
    send_button = st.button("Send", key="send_btn")  # Button to send the query.

# Process the user's query and generate a response.
if send_button or st.session_state.get("send_input") and query or voice_recording:
    with st.spinner("Processing... Please wait!"):  # Display a spinner while processing.
        response = _chain.invoke({'question': query})  # Generate the response.
    # Save user query and AI response in chat history.
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

# Display chat messages in the container.
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)  # Show messages from both user and AI.
