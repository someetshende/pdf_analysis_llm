import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain import LLMChain, OpenAI

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load PDF file
@st.cache_data
def load_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [GitHub](https://github.com/someetshende)
 
    ''')
    
    st.write('Project By Someet Shende')
    
# Streamlit App
def main():
    st.title("Document Search Application")

    # File Upload
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Load PDF and process text
        raw_text = load_pdf(uploaded_file)
        
        # Text Splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        # Create FAISS retriever
        docsearch = FAISS.from_texts(texts, embeddings)
        retriever = docsearch.as_retriever()

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

        # Initialize zero-shot agent
        tools = [
            Tool(
                name="Resume Report",
                func=qa_chain.run,
                description="It's a resume of a candidate.",
            )
        ]

        zero_shot_agent = initialize_agent(
            agent="zero-shot-react-description",
            tools=tools,
            llm=OpenAI(),
            verbose=True,
            max_iterations=3,
        )

        # User Input
        user_input = st.text_input("Ask a question:")

        if st.button("Get Answer"):
            # Get answer from the agent
            answer = zero_shot_agent(user_input)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
