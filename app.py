# from flask import Flask,render_template,jsonify,request
# from src.helper import download_huggingface_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import*
# import os

# app = Flask(__name__)

# load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# embeddings = download_huggingface_embeddings()

# index_name='medicalbot'

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
#     )

# retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

# ######
# #####
# ####
# #####
# ####
# #####

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get",methods=["GET","POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input":msg})
#     print("Response:",response["answer"])
#     return str(response["answer"])






# if __name__ == '__main__':
#     app.run(host="0.0.0.0",port=8080,debug=True)


from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.helper import download_huggingface_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embeddings
embeddings = download_huggingface_embeddings()

# Pinecone index
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# Create chain
prompt = ChatPromptTemplate.from_template(MEDICAL_BOT_PROMPT)  # from src/prompt.py
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})
    print("Bot:", response["answer"])

    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
