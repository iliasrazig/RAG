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
cv_file_path = os.path.join(data_directory, "CV.pdf")


# Streamlit UI
st.title("Assistant Chatbot RAG pour la recherche d'emploi")
st.write("Téléchargez votre CV et posez des questions sur les offres d'emploi en fonction du CV fourni.")

uploaded_file = st.file_uploader("Téléchargez votre CV (PDF uniquement)")

if uploaded_file is not None:
    try:
        # Load the candidate's CV
        with open(cv_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cv = pdf_loader(cv_file_path)
        st.success("CV téléchargé avec succès !")
        
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
        cv_text = None
else:
    st.info("Veuillez téléverser un fichier pour continuer.")
    cv_text = None


# We load the langchain embedding model that is on hugging face
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device':'cpu'},
    # model_kwargs={'device':'cpu'} if you don't have a gpu
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
Tu es un assistant en recrutement expert et ton objectif est d'aider un recruteur à identifier la meilleure offre d'emploi pour un candidat donné. Voici les informations fournies :

1. **CV du candidat** : 
{cv_text}

2. **Offres d'emploi disponibles** : 
{context}

### Consigne :
Analyse attentivement les informations du CV du candidat et les détails des offres d'emploi. Réponds à la question posée.

### Format attendu :
- **ID de l'offre recommandée** : [Identifiant de l'offre]
- **Explication** : Explique pourquoi cette offre est adaptée au profil du candidat. Mets en avant les correspondances entre les compétences du candidat et les exigences du poste.
- **URL de l'offre** : [URL]

### Règles à suivre :
- Si plusieurs offres conviennent, choisis celle qui correspond le mieux aux compétences principales du candidat.
- Justifie ton choix de manière concise mais claire.

### Question : 
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

# Input question from the user
user_question = st.text_input("Entrez votre question ici :")

if user_question:
    try:
        response = rag_chain.invoke(user_question)
        st.subheader("Réponse du chatbot :")
        st.write(response["answer"])
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
