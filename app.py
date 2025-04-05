# Import necessary libraries and modules
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from dotenv import load_dotenv
import os
import streamlit as st


# Load the environment variables (uncomment if using .env file)
# load_dotenv()


# -----------------------------
# Step 1: Load Documents
# -----------------------------

# Load CSV data from the ecommerce dataset
loader = CSVLoader('./ecommerce.csv')

# Read the documents from the CSV file
docs = loader.load()


# -----------------------------
# Step 2: Create Embeddings
# -----------------------------

# Initialize the HuggingFace embeddings with the desired model
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')


# -----------------------------
# Step 3: Set up Qdrant Vector Store
# -----------------------------

# IMPORTANT: Replace the API key below with your secure key (preferably stored as an environment variable)
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9su4JnZUEXsmmVd7SL411yU0A3c_qo2RZHVZFkfmUe4'

# Specify the URL for the Qdrant instance
url = 'https://3b44d132-6023-47ce-ab00-4844347b0202.europe-west3-0.gcp.cloud.qdrant.io'

# Create the Qdrant vector store from an existing collection named "ecommerce"
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url=url,
    api_key=api_key,
    prefer_grpc=False,
    collection_name="ecommerce"
)


# -----------------------------
# Step 4: Set up Google Generative AI
# -----------------------------

# IMPORTANT: Replace the API key below with your secure Google API key (preferably stored as an environment variable)
google_api = "AIzaSyCZGDK9CR99TS9mHi0UNbLN5bonYwT_BGE"

# Initialize the Google Generative AI model (Gemini-1.5 Flash 002)
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)


# -----------------------------
# Step 5: Configure the Document Retriever
# -----------------------------

# Define the number of chunks to retrieve from the document store
num_chunks = 3

# Set up the retriever with the "mmr" (max marginal relevance) search type
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})


# -----------------------------
# Step 6: Define Helper Functions and Prompt Template
# -----------------------------

# Function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the chatbot prompt template
prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about ecommerce store.
Answer all questions as if you have knowledge about its category, item name, and description of the product.
If asked any other question, say: I am trained to provide information about ecommerce store only.
Context: {context}
Question: {question}
"""

# Create the ChatPromptTemplate using the provided prompt string
_prompt = ChatPromptTemplate.from_template(prompt_str)


# -----------------------------
# Step 7: Chain Setup for the Chatbot
# -----------------------------

# Use itemgetter to extract the question from the input dictionary
query_fetcher = itemgetter("question")

# Create a chain setup that:
#  - fetches the question,
#  - retrieves the context using the retriever and formatting function,
#  - passes the data to the prompt template,
#  - invokes the language model,
#  - and finally parses the output as a string.
setup = {"question": query_fetcher, "context": query_fetcher | retriever | format_docs}
_chain = setup | _prompt | llm | StrOutputParser()


# -----------------------------
# Step 8: Set up the Streamlit User Interface
# -----------------------------

# Set the title of the Streamlit app
st.title("Ecommerce Chatbot")


# Create a container for displaying chat messages
chat_container = st.container()


# Initialize the session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to flag when user input is sent (if needed for further expansion)
def send_input():
    st.session_state.send_input = True


# -----------------------------
# Step 9: Build the Input Field and Send Button
# -----------------------------

with st.container():
    # Text input for user query; label is hidden for a cleaner UI
    query = st.text_input("Please enter a query", label_visibility="collapsed", key="query")
    
    # A single send button to submit the query
    send_button = st.button("Send", key="send_btn")


# -----------------------------
# Step 10: Process Chat Logic and Generate Response
# -----------------------------

if query and send_button:
    
    # Display a spinner while processing the request
    with st.spinner("Processing... Please wait!"):
        
        # Invoke the chain with the user's query
        response = _chain.invoke({'question': query})
        
        # Optional: Print the response to the console for debugging purposes
        print(response)

    # Update the session state with the new messages
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

else:
    # If no query has been submitted yet, prompt the user to start the conversation
    with chat_container:
        st.write("Start asking questions to interact with the chatbot")


# -----------------------------
# Step 11: Display Chat Messages in the Chat Container
# -----------------------------

with chat_container:
    # Loop through all stored messages and display them in the chat container
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
