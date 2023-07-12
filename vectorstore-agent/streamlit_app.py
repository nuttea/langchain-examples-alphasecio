import streamlit as st
import vertexai
from langchain.agents import load_tools, initialize_agent
from langchain import PromptTemplate, VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Initialize Vertex AI
vertexai.init(project="cloud-llm-preview4", location="us-central1")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,)
    
embeddings = VertexAIEmbeddings()

# Document Loader
loader = TextLoader("text_source/factsheet_xpresscash.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = VertexAIEmbeddings()
xpress_cash_store = Chroma.from_documents(
    texts, embeddings, collection_name="xpress-cash"
)

# Toolkit and Agent
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

xpress_cash_vectorstore_info = VectorStoreInfo(
    name="xpress_cash",
    description="Xpress Cash finance product for quick cash loan",
    vectorstore=xpress_cash_store,
)
toolkit = VectorStoreToolkit(vectorstore_info=xpress_cash_vectorstore_info, llm=llm)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# Streamlit app
st.subheader('LangChain Search')

# Source text input
search_query = st.text_area("Query", height=200)

# If the 'Search' button is clicked
if st.button("Search"):
    # Validate inputs
    if not search_query.strip():
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              result = agent_executor.run(search_query)
              st.success(result)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
