from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

## Extract data from the pdf file
def load_pdf_file(data):
    if not os.path.exists(data):
        os.makedirs(data, exist_ok=True)
        print(f"Created missing directory: {data} (put your PDFs here!)")

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

## Split data into the Text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

## Download the Embeddings from the huggingface
def download_huggingface_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

