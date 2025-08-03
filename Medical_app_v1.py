# pyright: ignore[reportMissingImports]
#import fitz
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from google.api_core.client_options import ClientOptions
from langchain.vectorstores.azuresearch import AzureSearch

from typing import Optional
from google.cloud import documentai  # type: ignore
from pinecone import Pinecone
import openai
import os
from dotenv import load_dotenv
import streamlit as st
import json
import shutil
from streamlit_feedback import streamlit_feedback
import easyocr

#Semantic chunking, Fxied  chunking and the merging between both of them.
#Semantic Chunkinhg has still issues with small the closed ranked relevant context to the query.
#Fixed Chunking the problem cannot capture the relations between the headers or pieces of information.
#The problem here after we have uploaded a document, DI cannot exract ocr from handwritten accurately.
#So we will use a rephrase the textual content came from DI with LLMs to solve this problem.
def handle_ocr_easy(lang,path):
     
        # Create an OCR reader object
        reader = easyocr.Reader([str(lang)])

        # Read text from an image
        result = reader.readtext(path)

        return result
# Load environment variables
# Configure page
st.set_page_config(
    page_title="Medical Assistant unofficial",
    page_icon="🤖",
    layout="wide"
)

def format_context(context):
     """
        It took the context and reformatted the context.
     """
     formatted_context= ""
     # iterate over the list of context
     for idx, doc in enumerate(context):
          formatted_context+=f"**Context**{str(idx)}: {doc.page_content}\n\n"
     return formatted_context
          
def generate_embeddings():
    embeddings_model = OpenAIEmbeddings(chunk_size=1000)
    return embeddings_model

def extract_text_from_pdf(pdf_file):
     """
      It takes the pdf and extracts the textual content from it.
     """
     reader = PdfReader(pdf_file)
     raw_text= ""
     for idx, page in enumerate(reader.pages):
          text = page.extract_text()
          if text:
               raw_text+=text
     return raw_text          
#Tokenize into character-level
# 1000 characters could cut the complete sentence or word
# so chunk_overlap is to take the previous some chracter from the first
# chunk and added to the first groups of characters for the chunk
def create_vector_database_by_fixed_faiss(raw_text):
    # Chunk the text
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    # Split the text
    texts = text_splitter.split_text(raw_text)
    # Save the vector of texts of pdf into FAISS
    vec_db = FAISS.from_texts(texts, generate_embeddings())
    return vec_db

def fixed_chunking(raw_text):
    """
    It takes the input and split it into fixed splits.
    """ 
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    # Split the text
    texts = text_splitter.split_text(raw_text) 
    return texts
def create_vector_database_by_semantic_faiss(docs):
    """
     it is suitable for semantic chunking
    """
    vec_db = FAISS.from_texts(docs, generate_embeddings())
    return vec_db



def delete_db(path):
     """
     Remove Vector Database
     """
     shutil.rmtree(path)


# what's the difference between textual and document
# Document level has meta data could support us to highlight the piece in that documnet
# page number,etc
def document_handler(file_path):
    """
    Document handler takes the document structure and save in db
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs,embeddings)
    return db



def simulate_search_through_via_embeddings(list_of_documents):
   """
   It takes list of documents 
   """
   list_of_documents = [ 
       Document(page_content="foo",metadata=dict(page=1)),
       Document(page_content="bar",metadata=dict(page=2)),
       Document(page_content="foo",metadata=dict(page=3)),
       Document(page_content="bar burr", metadata=dict(page=4)),
       Document(page_content="foo",metadata=dict(page=3)),
       Document(page_content="bar bruh",metadataa=dict(page=3))
   ] 
   #initlialize embeddings of openAI for textual content
   embeddings = OpenAIEmbeddings()
   # apply embeddings over the list of documents with metadata
   db = FAISS.from_documents(list_of_documents,embeddings)
   results_with_scores = db.similarity_search_with_score("foo")
   for doc, score in results_with_scores:
                print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")        
   
# Vec db should have the way of embeddings when create or load 
# Because when the user enters the query vecdb will be embedded this query
# And search through wether KNN will search each embeddings of each chunk and order and choose the lowest score.
# ANN is clustering all similar groups with Least Square hashing into one buckets
# When the query comes, they will search with LSH which bucket is very similar to the embeddings of that query 
def retrieve_relevant_context(query, vec_db,k=5):
       if vec_db != None:
            #This function runs ANN search on the VectorDatabases
            docs = vec_db.similarity_search(query,k=k)
            return docs
       else:
            return None        
       

#Initialize session state for messages
def initialize_session_state():
     """
      it intializes the streamlit variables
     """
     if "messages" not in st.session_state:
          st.session_state.messages = load_history()
     if "crtdata" not in st.session_state:
          st.session_state.crtdata = None
     if "last_context" not in st.session_state:    
          st.session_state.last_context = None
     if "uploaded_file" not in st.session_state:
          st.session_state.uploaded_file=None      

env_path = os.path.join('.env')
load_dotenv(env_path)
document_intelligence_endpoint = st.secrets["AZURE_DI_ENDPOINT"]
document_intelligence_key = st.secrets["AZURE_DI_KEY"]
ai_search_key =  st.secrets["AI_SEARCH_API"]
ai_search_end =  st.secrets["AI_SEARCH_END"]
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

with st.sidebar:
      st.header("LLMs Parameter Changer")
      temperature=st.slider("temperature metric",min_value=0.0,max_value=1.0,value=0.7,step=0.1)   
      model_name=st.selectbox('LLMs type',options=['gpt-4o','gpt-4.1','Gemini','Llama','Mistral','claude'])
      MAX_HISTORY_LENGTH = st.number_input(
                            "max history length:", 
                            min_value=0, 
                            max_value=100, 
                            value=5,
                            step=1)
      paper_content = st.sidebar.file_uploader("Please Upload a pdf document")   
      create_database = st.button('Create a VedtorDatabase')
      load_database = st.button('Load a VedtorDatabase')
      save_database = st.button('Save a VedtorDatabase Local')
      delete_database = st.button('Delete a VectorDatabase')
      #this because not all LLMs has system message
     
# Stream chat response
def post_processors_LLMs(transcript):
    """
      It takes the output of system message and extra relevant context.

    """
    system_prompt="""
                     The following is an OCR output.
                     Please improve it and make it more readable.
                     Do not summarize the content.
                  """
    system_msg = [{"role":"system", "content":system_prompt}]
    user_msg = [{"role":"user", "content":transcript}]
    messages = system_msg + user_msg
    # create a request to the model such open ai request
    postprocessed_transcript= client.chat.completions.create(model="gpt-4o",
                                       messages= messages,      
                                       temperature = 0)
    #st.sidebar.write(postprocessed_transcript.choices[0].message.content)
    return postprocessed_transcript.choices[0].message.content




# Stream chat response
def stream_chat_response(chat_history,system_msg_content, model_name, temperature, max_history_length):
    """
      It handles model request and response.
    """
    system_msg = [{"role":"system", "content":system_msg_content}]
    #Add the first chat_history and it acculamates if the chat has more than past masseges.
    #chat_history.append({"role":"user","content":message['text']})
    #Check if the length of chat history has reached the maximum length.
    if len(chat_history) > max_history_length:
        # keep the last max_history_length_existed 
        chat_history = chat_history[-max_history_length:]

    messages = system_msg + chat_history
    # create a request to the model such open ai request
    response= client.chat.completions.create(model=model_name,
                                       messages= messages,      
                                       temperature = temperature,
                                       stream=True
        )
    # yeild like save in cache to stream it chunk by chunk
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
def semantic_chunking(document):
    """
    It takes the documents and extract. 
    """
    paragraphs = document["paragraphs"]
    tables = document["tables"]
    chunks = []
    for paragraph in paragraphs:
        chunks.append(str(paragraph))
    for table in tables:
        chunks.append(str(table))    
    return chunks    

# The problem when we retrieve the k relevant contexts
def semantic_chunking_with_fixed(document):
    """
    It takes the documents and extract. 
    """
    paragraphs = document["paragraphs"]
    tables = document["tables"]
    chunks = []
    for paragraph in paragraphs:
        chunks.append(str(paragraph))
    for table in tables:
        chunks.append(str(table))    
    #Add fixed chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(document["content"])
    chunks.extend(texts)

    return chunks    


def save_history(chat_history):
  try: 
    with  open('chat_history.json','w') as chat_hist_file:
         json.dump(chat_history,chat_hist_file)
  except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

               
def clear_history():
      """
       Clear history from the chat and file
      """
      st.session_state.messages = []
      save_history([])
      st.rerun()
def load_history():
      """
        load the previous history from file
      """
      try:
          if os.path.exists('chat_history.json'):
              with open('chat_history.json', 'r') as f:
                  return json.load(f)
      except Exception as e:
          st.error(f"Error loading chat history: {str(e)}")
      return []
def document_layout_analysis_azure(document):
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



import os
from google.cloud import documentai
from google.oauth2 import service_account

def document_layout_analysis_google(
    file_path,
    credentials_path,
    project_id,
    processor_id,
    location="us"
):
    """
    Extract text from a document using Google Cloud Document AI
    
    Args:
        file_path (str): Path to the document file
        credentials_path (str): Path to Google Cloud service account JSON file
        project_id (str): Google Cloud project ID
        processor_id (str): Document AI processor ID
        location (str): Processing location (default: "us")
    
    Returns:
        dict: Extracted text and document information
        
    Example:
        result = extract_text_from_document(
            file_path="document.pdf",
            credentials_path="service-account.json",
            project_id="my-project-id",
            processor_id="abc123def456",
            location="us"
        )
        print(result['full_text'])
    """

    # Set up credentials
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    
    # Initialize Document AI client
    client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    
    # Create processor path
    processor_name = client.processor_path(project_id, location, processor_id)
    
    # Determine MIME type based on file extension
    mime_type_map = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    
    file_extension = os.path.splitext(file_path)[1].lower()
    mime_type = mime_type_map.get(file_extension, 'application/pdf')
    
    # Read the file
    with open(file_path, "rb") as file:
        file_content = file.read()
    
    # Create the document object
    raw_document = documentai.RawDocument(
        content=file_content,
        mime_type=mime_type
    )
    
    # Configure the process request
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )
    
    # Process the document
    result = client.process_document(request=request)
    document = result.document
    st.sidebar.write(document.text)


def create_vector_database_azure(texts):
     index_name = "medical-assistant-index"
     vec_db = AzureSearch(azure_search_endpoint=ai_search_end,index_name=index_name,azure_search_key=ai_search_key,embedding_function=generate_embeddings())
     vec_db.add_texts(texts)
     return vec_db
     
def create_vector_database_pinecone(texts):
    """
      It takes the document and create a pinecone Vector database
    """
    index_name = "medical-assistant-index"
    pc = Pinecone(api_key=st.secrets['PINECONE_APIKEY'])
    vectorstore = Pinecone.from_texts(
    texts=texts,
    embedding=OpenAIEmbeddings(),
    index_name=index_name)
def main():
    # Initialize the session state which keeps the history permenantely during the session
    initialize_session_state()
    st.title("Medical Assistant unofficial")
    # Initialize OpenAI 
    clear_btn=st.button('clear chat')
    if clear_btn:
          clear_history()
    # Extract a textual content from pdf.
    vectordbName= "vectordb"
    folder_path="/home/abdo/Downloads/chat_with_pdf"
    # Create a vector database and load it 
    if create_database and paper_content:
        with st.spinner("Analyzing the Document..."):
             #doc = document_layout_analysis_google(project_id="413967592811",location="us",processor_id="f876a327eafdb8a4",uploaded_file=paper_content,mime_type="application/pdf")
             #doc=handle_ocr_easy("en","/home/abdo/Downloads/chat_with_pdf/medical.png")
             doc = document_layout_analysis_azure(paper_content)
        with st.spinner("Semantic Chunking..."):
             chunks=semantic_chunking_with_fixed(doc)      
        with st.spinner("Creating Vector database based on Semantic Chunking..."):
                try:  
                    if st.session_state.crtdata==None:
                        vec_db=create_vector_database_azure(chunks)
                        #vec_db=create_vector_database_pinecone(chunks)
                        st.session_state.crtdata = vec_db
                        st.sidebar.write("Vector DB has been created successfully")
                    else: 
                        st.sidebar.write("Vector DB has been already existed if you would like drop the last one")       
                except:
                    st.sidebar.write("Database didn't created successfully please upload your file")   
    if save_database:
        try:      
            st.session_state.crtdata.save_local(folder_path=folder_path,index_name=vectordbName) 
            #create a folder for database
            os.mkdir(vectordbName)
            shutil.move(vectordbName+".pkl",vectordbName)
            shutil.move(vectordbName+".faiss",vectordbName)
            st.sidebar.write("Database has been saved locally successfully")      
        except: 
            st.sidebar.write("Database hasnot been saved locally successfully.It might be no database existed.") 
                     
    #load the saved database
    if load_database:
            try:
                if st.session_state.crtdata ==None:
                   vec_db=FAISS.load_local(folder_path=folder_path, embeddings=generate_embeddings(),index_name=vectordbName,allow_dangerous_deserialization=True) 
                   st.session_state.crtdata = vec_db
                st.sidebar.write("Vector DB has been loaded sucessfully")
            except: 
                st.sidebar.write("Vector DB has been not loaded sucessfully")               
                    
    # Delete the database              
    if delete_database:
                try:  
                   st.session_state.crtdata = None
                   delete_db(folder_path+"/"+vectordbName)  
                   st.sidebar.write("Deleted locally Successfully!")   
                except:   
                   st.sidebar.write("No Vector Databse created please recreate one!")          
           
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
          
    #this because not all LLMs has system message we can add to prompt  
    message = st.chat_input("Type here...",accept_file=True)
    #open AI has system message field but not all LLMs are the same.
    #message ="Answer the following\n {message}:\n Given the following {Relevant_Context}"
    #message.format()
    if message:
       if  st.session_state.crtdata:
            try:
                relevant_context=retrieve_relevant_context(message['text'], st.session_state.crtdata,k=10)
                st.session_state.last_context = relevant_context
            except: 
                 st.sidebar.write('The attached didn"t parse well') 
       
       with st.chat_message('user'):
                st.session_state.messages.append({'role':'user','content':message['text'],"type":"text"}) 
                st.write(message['text'])
       #Display the persona of assistant and when get the response from the ai model
       with st.chat_message('assistant'):
            #Design System message as LLM persona
            system_msg = "Act as an Medical expert who answers about the {medical_paper}."
            full_response = ""
            response_placeholder = st.empty()  
            with st.spinner("Reading File..."):  
                 
                #Softcoded extra relevant context from querying vectordatabase to extract textual content of pdf.
                if st.session_state.last_context!=None:      
                    system_msg = system_msg.format(medical_paper=st.session_state.last_context)
                   
                    st.sidebar.text_area("Last Query For relevant context:",value=format_context(st.session_state.last_context),height=300)
                else:
                    system_msg = system_msg.format(paper="No Context Found.")    
                    st.session_state.last_context = None
                system_msg=post_processors_LLMs(system_msg)   
                for chunk in stream_chat_response(st.session_state.messages,system_msg, model_name, temperature, MAX_HISTORY_LENGTH): 
                          full_response+=chunk   
                          response_placeholder.write(full_response)
            #add the history of assistant repsonse and user input into chat messages    
            st.session_state.messages.append({'role':'assistant','content':full_response})       
             # Save updated chat history
            save_history(st.session_state.messages)
             
if __name__ == "__main__":
    main()

