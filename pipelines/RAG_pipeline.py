import sys
import os

sys.path.append(os.path.abspath('/home/onyxia/work/RAG/data'))


import transformers
import torch
from model_pipeline import ModelLoader
from dataloaders import pdf_loader
from dataloaders import format_docs
from vector_DB_creation import vector_DB
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

torch.cuda.reset_peak_memory_stats()

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_directory)
data_directory = os.path.join(parent_dir, "data")
cv_file_path = os.path.join(data_directory, "CV RAZIG_Ilias_en.pdf")
cv = pdf_loader(cv_file_path)

# We load the langchain embedding model that is on hugging face
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device':'cpu'},
    # model_kwargs={'device':'cpu'} if you don't have a gpu
    encode_kwargs={'normalize_embeddings': True}
)

db_loader = vector_DB("vector_DB", data_directory)
try :
    vectorstore = db_loader.load_vector_DB(huggingface_embeddings)
except :
    vectorstore = db_loader.create_vector_DB("api_response.json",huggingface_embeddings)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
loader = ModelLoader(model_name)
pipeline = loader.load_pipeline()



# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

template = """
Tu es un assistant en recrutement expert et ton objectif est d'aider un recruteur à identifier la meilleure offre d'emploi pour un candidat donné. Voici les informations fournies :

1. **CV du candidat** : 
{cv_text}

2. **Offres d'emploi disponibles** : 
{context}

### Consigne :
Analyse attentivement les informations du CV du candidat et les détails des offres d'emploi. Réponds à la question posée en suivant le format ci-dessous :

### Format attendu :
- **ID de l'offre recommandée** : [Identifiant de l'offre]
- **Explication** : Explique pourquoi cette offre est adaptée au profil du candidat. Mets en avant les correspondances entre les compétences du candidat et les exigences du poste.
- **URL de l'offre** (si disponible) : [URL]

### Règles à suivre :
- Si plusieurs offres conviennent, choisis celle qui correspond le mieux aux compétences principales du candidat.
- Justifie ton choix de manière concise mais claire.
- Si des informations sont manquantes pour justifier pleinement ton choix, mentionne-les.

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

test = rag_chain.invoke("Quelle est la meilleure offre pour ce candidat en se basant sur son cv ?")
print(test)