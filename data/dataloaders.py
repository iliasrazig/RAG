import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


data_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(data_directory, "api_response.json")
cv_file_path = os.path.join(data_directory, "CV RAZIG_Ilias_en.pdf")

with open(json_file_path, 'r') as j:
    data = json.loads(j.read())


# functions for loading the files in the local directory

def document_loader(file, chunk_size = 700, chunk_overlap = 50):

    loader = JSONLoader(
        file_path = file,
        jq_schema = '.resultats[]',
        text_content = False
    )

    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)
    return docs_after_split

def pdf_loader(file):
    loader = PyPDFLoader(
        file_path=file
    )

    doc = loader.load()
    return doc

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)