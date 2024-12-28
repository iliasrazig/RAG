import transformers
import os
import torch
from model_pipeline import ModelLoader
from dataloaders import pdf_loader
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


current_directory = os.getcwd()
cv_file_path = os.path.join(current_directory, "CV RAZIG_Ilias_en.pdf")
cv = pdf_loader(cv_file_path)
vectorstore = FAISS.load_local("faiss_index", huggingface_embeddings)

model_name = "meta-llama/Llama-2-7b-chat-hf"
loader = ModelLoader(model_name)
pipeline = loader.load_pipeline()



# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """
Tu es un expert du recrutement. Un candidat a fourni son CV suivant :
{cv_text}

Voici des offres d'emploi avec leur description, leur identifiant et d'autres informations disponibles :
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

test = rag_chain.invoke("Quelle est la meilleure offre d'emploi pour ce candidat ?")
print(test)