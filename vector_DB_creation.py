from dataloaders import document_loader
from dataloaders import pdf_loader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


current_directory = os.getcwd()
json_file_path = os.path.join(current_directory, "api_response.json")

with open(json_file_path, 'r') as j:
    data = json.loads(j.read())

# We split the the json into documents
docs_after_split = document_loader(json_file_path, chunk_size = 700, chunk_overlap  = 50)


# We load the langchain embedding model that is on hugging face
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# Creation of the vectorDB with the embeddings
vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)
vectorstore.save_local("faiss_index")