from typing import List
import requests
import fitz
from io import BytesIO

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs_from_urls(pdf_urls:List[str])->List[Document]:
    documents=[]
    for url in pdf_urls:
        response = requests.get(url,timeout=20)
        response.raise_for_status()

        pdf_stream =BytesIO(response.content)
        pdf= fitz.open(stream=pdf_stream,filetype="pdf")

        full_text=""
        for page in pdf:
            full_text+=page.get_text()
        
        documents.append(
            Document(
                page_content=full_text,
                metadata={"source":url}
            )
        )

    return documents
def build_RAG_agent(pdf_urls:List[str]):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    splitter= RecursiveCharacterTextSplitter(
        chunk_size =800,
        chunk_overlap=150,
        separators=["\n\n","\n","."," "]
    )
    chunks=splitter.split_documents(load_pdfs_from_urls(pdf_urls))
    vector_store.add_documents(chunks)
