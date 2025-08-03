import streamlit as st
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter, MarkdownHeaderTextSplitter

import os 
env_path = os.path.join('.env')
load_dotenv(env_path)
uploaded_file = st.file_uploader("Please upload a file")
document_intelligence_endpoint = st.secrets["AZURE_ENDPOINT"]
document_intelligence_key = st.secrets["AZURE_KEY"]
def document_layout_analysis(document):
    """
      It take pdf or image and extract the fields of tables 
      and their relatons.
    """
    doc_intelligence_client = DocumentIntelligenceClient(endpoint=document_intelligence_endpoint,credential=AzureKeyCredential(document_intelligence_key))
    #STYLES 
    features = []
    poller = doc_intelligence_client.begin_analyze_document(
         "prebuilt-layout",
         document, 
         content_type="application/octet-stream",
         output_content_format="markdown",
         features= features

    ) 
    result= poller.result()
    return result
if uploaded_file:
    with st.spinner("Analyzing the document...."):
        doc = document_layout_analysis(uploaded_file)

        st.write("## Content: ")
        st.write(doc["content"])

        st.write("## DI keys: ")
        st.write(doc.keys())
        
        st.write("## Sections: ")
        st.write(doc["sections"])
        
        st.write("## Paragraphs")
        st.write(doc["paragraphs"])

        st.write("## Tables:")
        st.write(doc["tables"])

        st.write("## Pages:")
        st.write(doc["pages"])


headers_to_split_on = [
  ("#", "Header 1"),
  ("##","Header 2"),
  ("###","Header 3")
  ]        

#Semantic chunking for Markdown by Langchain
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

st.write(text_splitter.headers_to_split_on)

#Custom Semantic Chunking


def semantic_chunking(document):
    """
    It takes the documents and extract. 
    """
    paragraphs = document["paragraphs"]
    tables = document["tables"]
    chunks = []
    for paragraph in paragraphs:
        chunks.append(paragraph)
    for table in tables:
        chunks.append(table)    