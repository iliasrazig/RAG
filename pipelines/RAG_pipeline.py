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
    model_kwargs={'device':'cuda'},
    # model_kwargs={'device':'cpu'} if you don't have a gpu
    encode_kwargs={'normalize_embeddings': True}
)

db_loader = vector_DB("vector_DB", data_directory)
try :
    vectorstore = db_loader.load_vector_DB(huggingface_embeddings)
except :
    vectorstore = db_loader.create_vector_DB("api_response.json",huggingface_embeddings)

model_name = "meta-llama/Llama-2-7b-chat-hf"
loader = ModelLoader(model_name)
pipeline = loader.load_pipeline()



# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """
Tu es un expert du recrutement. Un candidat a fourni son CV suivant :
{cv_text}

Voici des offres d'emploi avec leur description et leur identifiant :
{context}

Utilise le CV et le contexte pour répondre à la question suivante.
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

test = rag_chain.invoke("Parmi les offres ci-dessus, quelle est l'offre qui correspond le mieux au CV du candidat ?")
print(test)