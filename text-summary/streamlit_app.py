import os, streamlit as st
import vertexai
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.chains.summarize import load_summarize_chain

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

# Streamlit app
st.subheader('LangChain Text Summary')

# Get OpenAI API key and source text input
source_text = st.text_area("Source Text", height=200)

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not source_text.strip():
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Split the source text
              text_splitter = CharacterTextSplitter()
              texts = text_splitter.split_text(source_text)

              # Create Document objects for the texts (max 3 pages)
              docs = [Document(page_content=t) for t in texts[:3]]

              # Initialize the OpenAI module, load and run the summarize chain
              chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
              summary = chain.run(docs)

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
