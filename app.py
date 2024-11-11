from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages.base import BaseMessage
from fastapi import UploadFile, File, HTTPException
from langchain_openai import OpenAIEmbeddings
from typing import Any
import uuid
import pylibmagic
from fastapi import Form, BackgroundTasks
from fastapi import Query
import io
import pinecone
from PyPDF2 import PdfReader
import docx
import shutil
import uvicorn
from typing import List, Dict
import numpy as np
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import ServerlessSpec
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec
# from langchain.document_loaders import DirectoryLoader
from langchain_core.runnables import RunnableParallel
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, validator, ValidationError
from tqdm.auto import tqdm
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse, JSONResponse
import pinecone
import glob
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from uuid import uuid4
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
import getpass
import os
import concurrent.futures
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader  # For reading PDF files
import docx
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class MyModel(BaseModel):
    message: BaseMessage


class Config:
    arbitrary_types_allowed = True


app = FastAPI(
    title="LangChain Server",
    version="o1",
    description="",
)
# Set all CORS enabled origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["Processing-Time"] = str(process_time)
    return response


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
directory = os.getenv("directory")
base_directory = os.getenv("base_directory")


if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Verify that the keys are loaded
# print(f"OpenAI API Key: {os.environ.get('OPENAI_API_KEY')}")
# print(f"Pinecone API Key: {os.environ.get('PINECONE_API_KEY')}")
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")


pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
import time

index_name = "vitafy-prod"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)




@app.post("/api/artificial-intelligence/upload")
async def upload_files(
    tenantId: str = Form(...),
    files: List[UploadFile] = File(...),
    background_task: BackgroundTasks = BackgroundTasks(),
):
    """Upload multiple PDF, DOCX, and TXT files"""
    dir_name = str(uuid4())
    tenant_directory = os.path.join(base_directory, dir_name)
    os.makedirs(tenant_directory, exist_ok=True)

    # Allowed file extensions
    allowed_extensions = {".pdf", ".docx", ".txt"}

    # Tracking file names to check for duplicates
    fileName = set()

    # Saving each uploaded file
    for file in files:
        if file.filename in fileName:
            raise HTTPException(
                status_code=400, detail=f"Duplicate file detected: {file.filename}"
            )

        fileName.add(file.filename)

        # Check for file extension
        _, extension = os.path.splitext(file.filename)
        if extension.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF, DOCX, and TXT files are allowed.",
            )

        # Define the destination path for the uploaded file
        destination = os.path.join(tenant_directory, file.filename)

        print('creating doc ' + destination)
        # Save the uploaded file to the tenant's directory
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)



    # Load the documents from the tenant's directory
    docs = load_docs(tenant_directory, tenantId)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", #response time is 9s  #infloat/e5-base-V2 has 3.53sec response time.
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    chunk_size = 1000 
    chunk_overlap = 20

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    split_docs = []
    chunk_ids = []

    uploaded_documents_path = "do_not_delete_uploaded_documents.json"

    if not os.path.exists(uploaded_documents_path):
        with open(uploaded_documents_path, "w") as f:
            json.dump({}, f)

    with open(uploaded_documents_path, "r") as f:
        upload_documents = json.load(f)

    for doc in docs:
        curr_split_docs = text_splitter.split_documents([doc])

         # Generate a unique document ID
        document_id = str(uuid4())

        # Create unique IDs for each chunk with the document ID as a prefix
        curr_chunk_ids = [f"{document_id}_chunk_{i+1}" for i in range(len(curr_split_docs))]

        split_docs = split_docs + curr_split_docs
        chunk_ids = chunk_ids + curr_chunk_ids

        upload_documents[document_id] =  {"fileName": doc.metadata['filename'], "id": document_id, "tenantId": tenantId}

    # Add documents to vector store with unique chunk IDs
    vectorstore.add_documents(documents=split_docs, ids=chunk_ids)

    with open(uploaded_documents_path, "w") as f:
        json.dump(upload_documents, f)

    shutil.rmtree(tenant_directory)

   
    

def load_docs(directory, tenantId):
    loader = DirectoryLoader(directory)
    docs = loader.load()
    
    for doc in docs:
        doc.metadata['tenantId'] = tenantId

        # Extract the filename from the 'source' path
        doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        print("ooooooooo" + doc.metadata['filename'])

    return docs


#     return JSONResponse(
#         content={"message": "Files uploaded successfully."},
#         status_code=200,
#     )



@app.get("/api/artificial-intelligence/tenant_files")
async def retrieve_files(tenantId: str = Query(...)):
    uploaded_documents_path = "do_not_delete_uploaded_documents.json"


    with open(uploaded_documents_path, "r") as f:
        upload_documents = json.load(f)

    tenantFiles = []
    for item in list(upload_documents.values()):
        if item.get('tenantId') == tenantId:
            tenantFiles.append(item)

    return {"data": tenantFiles}



def initializeVectorStore():
    embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",                             #response time is 9s  #infloat/e5-base-V2 has 3.53sec response time.
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore

llm = ChatOpenAI(
    #model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0.0,
)
@app.get("/api/artificial-intelligence/prompts")
async def prompts_keyword(tenantId: str = Query(...), keyword: str = Query(...)):
    try:
        print(tenantId, keyword)
        vectorStore = initializeVectorStore()
        #retriever = vectorStore.as_retriever()
        retriever = vectorStore.as_retriever(
    
    #search_type="similarity",
    search_kwargs={
        "k": 1,
            "filter" : {
        'tenantId': {'$eq': tenantId}  
    
    },
            }
)
        newQa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            #retriever=vectorStore.as_retriever(),
            return_source_documents=True,
        )

        #answer = newQa.invoke({"query": keyword})
        answer = newQa({"query": keyword})


        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}




class DeleteRequest(BaseModel):
    prefix: str
@app.delete("/api/artificial-intelligence/delete", summary="Delete documents by prefix", description="Delete all documents in the Pinecone index that match the given prefix.")
async def delete_documents(request: DeleteRequest):
    """
    Deletes all document IDs in the Pinecone index that start with the given prefix.

    Parameters:
    - **prefix**: The prefix string to filter and delete documents by.

    Returns:
    - A JSON response with a success message or an error if no IDs were found with the given prefix.
    """
    prefix = request.prefix
    # List all IDs with the given prefix
    ids_to_delete = [id for id in index.list(prefix=prefix)]
    
    print(ids_to_delete)

    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="No documents found with the given prefix.")

    # Delete the IDs from the index
    index.delete(ids=ids_to_delete)
    return {"message": f"Deleted {len(ids_to_delete)} documents with prefix '{prefix}'."}





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)