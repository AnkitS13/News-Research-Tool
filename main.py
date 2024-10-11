import os
import langchain
import streamlit as st
import pickle
import time
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS


st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    #split data 
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
     # Assuming 'docs' is a list of 'Document' objects
    doc_texts = [doc.page_content for doc in data]  # Extract the text content from each Document

# Create FAISS vector index using document texts and embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorindex_faiss = FAISS.from_texts(doc_texts, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_faiss, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Define the QA pipeline using a pre-trained model
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            class HuggingFaceQAWithSourcesChain:
                def __init__(self, retriever, qa_pipeline):
                    self.retriever = retriever
                    self.qa_pipeline = qa_pipeline
    
                def __call__(self, inputs, return_only_outputs=False):
                    # Retrieve relevant documents from the FAISS vector store 
                    docs = self.retriever.get_relevant_documents(inputs['question'])
        
                    # Combine all retrieved document texts into one
                    context = " ".join([doc.page_content for doc in docs])
        
                    # Use the Hugging Face QA pipeline to answer the question based on the context
                    result = self.qa_pipeline(question=inputs['question'], context=context)
                    
            retriever = vectorstore.as_retriever()
            hf_chain = HuggingFaceQAWithSourcesChain(retriever=retriever, qa_pipeline=qa_pipeline)
            result = hf_chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])