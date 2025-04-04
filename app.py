
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

loader=CSVLoader('./ecommerce.csv')

docs=loader.load()

embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9su4JnZUEXsmmVd7SL411yU0A3c_qo2RZHVZFkfmUe4'
url = 'https://3b44d132-6023-47ce-ab00-4844347b0202.europe-west3-0.gcp.cloud.qdrant.io'
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url=url,
    api_key=api_key,
    prefer_grpc=False,
    collection_name="ecommerce"
)

google_api = "AIzaSyCZGDK9CR99TS9mHi0UNbLN5bonYwT_BGE"
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

num_chunks = 3
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

history = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about ecommerce store
Answer all questions as if you have knowledge about its category, item name, discription of the product.
If ask any other question , Say : I am trained to provide information about ecommerce store only.
Context: {context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Chain setup
query_fetcher =itemgetter("question")
setup = {"question": query_fetcher,"context": query_fetcher | retriever | format_docs}
_chain = setup | _prompt | llm | StrOutputParser()

# Streamlit UI
st.title("Ecommerce Chatbot")

# Chat container to display conversation
chat_container = st.container()

# Input field for queries
with st.container():
    query = st.text_input("Please enter a query", label_visibility="collapsed", key="query")
    send_button = st.button("Send", key="send_btn")  # Single send button
# Chat logic
if send_button and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response = _chain.invoke({'question': query})  # Generate response
    with chat_container:  # Append to chat container
        st.chat_message('user').write(query)
        st.chat_message('ai').write(response)
else:
    with chat_container:
        st.write("Start asking questions to interact with the chatbot")

# Display chat messages in the container.
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)  # Show messages from both user and AI.
