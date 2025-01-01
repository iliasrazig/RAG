import os
from dataloaders import document_loader
from dataloaders import pdf_loader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


class vector_DB :
    def __init__(self, database_name, save_dir):
        self.database_name = database_name
        self.save_dir = save_dir
        self.database_file_path = os.path.join(save_dir, database_name)

    def create_vector_DB(self, document, embedding_model):
        document_path = os.path.join(self.save_dir, document)
        # We split the the json into documents
        docs_after_split = document_loader(document_path, chunk_size = 2000, chunk_overlap  = 200)

        # Creation of the vectorDB with the embeddings
        vectorstore = FAISS.from_documents(docs_after_split, embedding_model)
        vectorstore.save_local(self.database_file_path)
        return vectorstore

    def load_vector_DB(self, embedding_model):
        vectorstore = FAISS.load_local(self.database_file_path, embedding_model, allow_dangerous_deserialization=True)
        return vectorstore

