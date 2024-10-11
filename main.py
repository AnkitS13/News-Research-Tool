import os
import streamlit as st
import pickle
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings


st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    
    # Extract the text content from the documents
    doc_texts = [doc.page_content for doc in data]

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
            
            # Initialize the Hugging Face QA pipeline
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

            # Define the custom QA chain with Hugging Face and FAISS
            class HuggingFaceQAWithSourcesChain:
                def __init__(self, retriever, qa_pipeline):
                    self.retriever = retriever
                    self.qa_pipeline = qa_pipeline
    
                def __call__(self, inputs, return_only_outputs=False):
                    # Retrieve relevant documents from the FAISS vector store
                    docs = self.retriever.get_relevant_documents(inputs['question'])
                    
                    if not docs:
                        print("No documents retrieved.")
                        return {"answer": "No answer found", "sources": "No source available"}
        
                    # Combine all retrieved document texts into one context
                    context = " ".join([doc.page_content for doc in docs])

                    # Use the Hugging Face QA pipeline to answer the question
                    result = self.qa_pipeline(question=inputs['question'], context=context)
                    
                    # Extract the answer
                    answer = result.get('answer', "No answer found")

                    # Collect sources from document metadata
                    sources = "\n".join([doc.metadata.get('source', 'No source available') for doc in docs])
        
                    # Return the answer and sources
                    if return_only_outputs:
                        return {"answer": answer, "sources": sources}
                    
                    return {"answer": answer, "context": context, "sources": sources}
            
            # Create retriever from vector store
            retriever = vectorstore.as_retriever()

            # Initialize the custom QA chain
            hf_chain = HuggingFaceQAWithSourcesChain(retriever=retriever, qa_pipeline=qa_pipeline)
            
            # Run the query through the chain
            result = hf_chain({"question": query}, return_only_outputs=True)
            
            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display the sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
            else:
                st.write("No sources available.")















# import os
# import langchain
# import streamlit as st
# import pickle
# import time
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from sentence_transformers import SentenceTransformer
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.vectorstores import FAISS


# st.title("News Research Tool 📈")

# st.sidebar.title("News Article URLs")

# urls=[]
# for i in range(3):
#     url=st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store_openai.pkl"

# main_placeholder = st.empty()

# if process_url_clicked:
#     #load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     #split data 
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#      # Assuming 'docs' is a list of 'Document' objects
#     doc_texts = [doc.page_content for doc in data]  # Extract the text content from each Document

# # Create FAISS vector index using document texts and embeddings model
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorindex_faiss = FAISS.from_texts(doc_texts, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorindex_faiss, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)

#             # Define the QA pipeline using a pre-trained model
#             qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
#             class HuggingFaceQAWithSourcesChain:
#                 def __init__(self, retriever, qa_pipeline):
#                     self.retriever = retriever
#                     self.qa_pipeline = qa_pipeline
    
#                 def __call__(self, inputs, return_only_outputs=False):
#                     # Retrieve relevant documents from the FAISS vector store 
#                     docs = self.retriever.get_relevant_documents(inputs['question'])
        
#                     # Combine all retrieved document texts into one
#                     context = " ".join([doc.page_content for doc in docs])
        
#                     # Use the Hugging Face QA pipeline to answer the question based on the context
#                     result = self.qa_pipeline(question=inputs['question'], context=context)

#                     # Debugging: Check if the QA pipeline returned a result
#                     print("QA Pipeline result:", result)

#                     # Extract the answer
#                     answer = result.get('answer', "No answer found")  # Handle cases where no answer is returned
        
#                     # Return only the answer
#                     if return_only_outputs:
#                         return {"answer": answer}
#                     return {"answer": answer, "context": context}
                    
#             retriever = vectorstore.as_retriever()
#             hf_chain = HuggingFaceQAWithSourcesChain(retriever=retriever, qa_pipeline=qa_pipeline)
#             langchain.debug = True
#             result = hf_chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)