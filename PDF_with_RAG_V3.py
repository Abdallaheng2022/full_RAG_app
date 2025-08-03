# pyright: ignore[reportMissingImports]
#import fitz
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import openai
import os
from dotenv import load_dotenv
import streamlit as st
import json
import shutil
from streamlit_feedback import streamlit_feedback
import easyocr
from document_intelligence import APIConfig, DocumentIntelligence # type: ignore
from azure.core.credentials import AzureKeyCredential # type: ignore

env_path = os.path.join('.env')
load_dotenv(env_path)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
document_intelligence_endpoint = st.secrets["AZURE_ENDPOINT"]
document_intelligence_key = st.secrets["AZURE_KEY"]
def handle_ocr_easy(lang,path):
     
        # Create an OCR reader object
        reader = easyocr.Reader([str(lang)])

        # Read text from an image
        result = reader.readtext(path)

        return result
# Load environment variables
# Configure page
st.set_page_config(
    page_title="Chat with Pdf with Vector_DB",
    page_icon="🤖",
    layout="wide"
)

def document_layout_analysis(document):
    """
      It take pdf or image and extract the fields of tables 
      and their relatons.
    """
    doc_intelligence_client = DocumentIntelligence(endpoint=document_intelligence_endpoint,credential=AzureKeyCredential(document_intelligence_key))
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
def create_vector_database_by_textual_content(raw_text):
    # Chunk the text
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    # Split the text
    texts = text_splitter.split_text(raw_text)
    # Save the vector of texts of pdf into FAISS
    vec_db = FAISS.from_texts(texts, generate_embeddings())
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
def retrieve_relevant_context(query, vec_db,k=4):
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



with st.sidebar:
      st.header("LLMs Parameter Changer")
      temperature=st.slider("temperature metric",min_value=0.0,max_value=1.0,value=0.7,step=0.1)   
      model_name=st.selectbox('LLMs type',options=['gpt-4.1','Gemini','Llama','Mistral','claude'])
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
def main():
    # Initialize the session state which keeps the history permenantely during the session
    initialize_session_state()    
    st.title("Chat With Pdf")
    # Initialize OpenAI 
    clear_btn=st.button('clear chat')
    if clear_btn:
          clear_history()
    # Extract a textual content from pdf.
    vectordbName= "vectordb"
    folder_path="/home/abdo/Downloads/chat_with_pdf"
    # Create a vector database and load it 
    if create_database:
        try:      
             raw_text=extract_text_from_pdf(paper_content) 
             st.session_state.uploaded_file =  paper_content    
        except: 
             st.sidebar.write("No file uploaded")  
         
        try: 
             if st.session_state.crtdata==None:
                  vec_db=create_vector_database_by_textual_content(raw_text)
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
                relevant_context=retrieve_relevant_context(message['text'], st.session_state.crtdata,k=4)
                st.session_state.last_context = relevant_context
            except: 
                 st.sidebar.write('The attached didn"t parse well') 
       
       with st.chat_message('user'):
                st.session_state.messages.append({'role':'user','content':message['text']}) 
                st.write(message['text'])
       #Display the persona of assistant and when get the response from the ai model
       with st.chat_message('assistant'):
            #Design System message as LLM persona
            system_msg = "Act as an AI expert who answers about the {paper}."
            full_response = ""
            response_placeholder = st.empty()  
            with st.spinner("Reading File..."):  
                 
                #Softcoded extra relevant context from querying vectordatabase to extract textual content of pdf.
                if st.session_state.last_context!=None:      
                    system_msg = system_msg.format(paper=st.session_state.last_context)
                    st.sidebar.text_area("Last Query For relevant context:",value=format_context(st.session_state.last_context),height=300)
                    st.session_state.last_context = None
                else:
                    system_msg = system_msg.format(paper="No Context Found.")    
                   
                for chunk in stream_chat_response(st.session_state.messages,system_msg, model_name, temperature, MAX_HISTORY_LENGTH): 
                          full_response+=chunk   
                          response_placeholder.write(full_response)
            #add the history of assistant repsonse and user input into chat messages    
            st.session_state.messages.append({'role':'assistant','content':full_response})       
             # Save updated chat history
            save_history(st.session_state.messages)
             
if __name__ == "__main__":
    main()

