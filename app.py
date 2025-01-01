import sys
import os

sys.path.append(os.path.abspath('/home/onyxia/work/RAG/data'))
sys.path.append(os.path.abspath('/home/onyxia/work/RAG/pipelines'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" 

import streamlit as st
import torch
from model_pipeline import ModelLoader
from dataloaders import pdf_loader, format_docs
from vector_DB_creation import vector_DB
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Configure GPU memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Set up paths
current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(current_directory, "data")
cv_file_path = os.path.join(data_directory, "CV RAZIG_Ilias_en.pdf")

# Load the candidate's CV
cv = pdf_loader(cv_file_path)

# Load embeddings model
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cpu' if GPU is not available
    encode_kwargs={'normalize_embeddings': True}
)

# Load vector database
db_loader = vector_DB("vector_DB", data_directory)
try:
    vectorstore = db_loader.load_vector_DB(huggingface_embeddings)
except Exception as e:
    st.error(f"Error loading vector DB: {e}")
    vectorstore = db_loader.create_vector_DB("api_response.json", huggingface_embeddings)

# Load the language model
#model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
loader = ModelLoader(model_name)
pipeline = loader.load_pipeline()

# Configure retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Define prompt template
template = """
Tu es un expert du recrutement. Un candidat a fourni son CV suivant :
{cv_text}

Voici des offres d'emploi avec leur description et leur identifiant :
{context}

{question}
"""

prompt = PromptTemplate(input_variables=["cv_text", "context", "question"], template=template)
llm = HuggingFacePipeline(pipeline=pipeline)
texte = format_docs(cv)

# Create a RAG Chain
rag_chain = (
    {
        "cv_text": RunnableLambda(lambda x: format_docs(cv)),
        "context": retriever | RunnableLambda(lambda x: format_docs(x)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# Streamlit UI
st.title("Chatbot RAG pour le Recrutement")
st.write("Posez des questions sur les offres d'emploi en fonction du CV fourni.")

# Input question from the user
user_question = st.text_input("Entrez votre question ici :")

if user_question:
    try:
        response = rag_chain.invoke(user_question)
        st.subheader("Réponse du chatbot :")
        st.write(response)
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
