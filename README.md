# Medical Assistant Unoifficial as an Experiment - Document Processing & Vector Database System

A Streamlit-based medical document analysis application that leverages Azure Document Intelligence, vector databases, and LLMs to extract, process, and query medical documents with intelligent chunking strategies.
[Direct Link](https://drive.google.com/file/d/1SEcDyDynQqssZ6ZZj-SeWAfoZ2x9Nu8y/view?usp=sharing)
## 🎯 Project Overview

This application processes medical documents or general document (PDFs, images) using Azure Document Intelligence for OCR and layout analysis, then stores the extracted content in vector databases for semantic search and retrieval. The system is designed to handle complex medical documents like clinical records, prescriptions, and medical reports.

### Example Medical Document Processing
The system can process handwritten medical records like clinical notes from Square Hospitals Ltd., extracting:
- Patient information and vital signs
- Clinical history and symptoms
- Treatment plans and prescriptions
- Diagnostic information

## 🏗️ Architecture

```
Medical Document → Azure Document Intelligence → Text Processing → Chunking Strategy → Vector Database → LLM Query
```

## 📊 Chunking Strategies & Issues

### 1. **Fixed Chunking**
```python
def fixed_chunking(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)
    return texts
```

**Issues:**
- Cannot capture relationships between headers and content sections
- May split medical terms or important context mid-sentence
- Fixed size doesn't account for document structure (tables, paragraphs)

### 2. **Semantic Chunking**
```python
def semantic_chunking(document):
    paragraphs = document["paragraphs"]
    tables = document["tables"]
    chunks = []
    for paragraph in paragraphs:
        chunks.append(str(paragraph))
    for table in tables:
        chunks.append(str(table))
    return chunks
```

**Issues:**
- Still has problems with finding closely ranked relevant context to queries
- May create chunks that are too large or too small
- Doesn't handle cross-references between document sections well

### 3. **Hybrid Approach (Semantic + Fixed)**
```python
def semantic_chunking_with_fixed(document):
    # Extract semantic elements
    chunks = semantic_chunking(document)
    
    # Add fixed chunking for comprehensive coverage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(document["content"])
    chunks.extend(texts)
    
    return chunks
```

**Benefits:**
- Combines structure-aware chunking with comprehensive text coverage
- Better retrieval accuracy for medical terminology and relationships

## 🔧 Azure Document Intelligence Integration

### Document Processing Pipeline

1. **OCR Extraction**: Handles handwritten medical notes and printed text
2. **Layout Analysis**: Identifies tables, paragraphs, and document structure
3. **Post-processing**: Uses LLM to improve OCR accuracy

```python
def document_layout_analysis_azure(document):
    doc_intelligence_client = DocumentIntelligenceClient(
        endpoint=document_intelligence_endpoint,
        credential=AzureKeyCredential(document_intelligence_key)
    )
    
    poller = doc_intelligence_client.begin_analyze_document(
        "prebuilt-layout",
        document,
        content_type="application/octet-stream",
        output_content_format="markdown"
    )
    
    return poller.result()
```

### OCR Post-Processing
```python
def post_processors_LLMs(transcript):
    system_prompt = """
    The following is an OCR output.
    Please improve it and make it more readable.
    Do not summarize the content.
    """
    # Uses GPT-4o to clean up OCR errors
```

## 🗄️ Vector Database Storage Options

### 1. **FAISS (Local Storage)**
```python
def create_vector_database_by_fixed_faiss(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)
    vec_db = FAISS.from_texts(texts, generate_embeddings())
    return vec_db
```

**Advantages:**
- Fast local retrieval
- No cloud dependencies for queries
- Complete data privacy

**Disadvantages:**
- Limited scalability
- No built-in backup/sync
- Single machine dependency

### 2. **Azure AI Search (Cloud Vector Database)**
```python
def create_vector_database_azure(texts):
    index_name = "medical-assistant-index"
    vec_db = AzureSearch(
        azure_search_endpoint=ai_search_end,
        index_name=index_name,
        azure_search_key=ai_search_key,
        embedding_function=generate_embeddings()
    )
    vec_db.add_texts(texts)
    return vec_db
```

**Advantages:**
- Scalable cloud infrastructure
- Built-in redundancy and backup
- Advanced search capabilities
- Multi-user access

**Disadvantages:**
- Network latency
- Ongoing cloud costs
- Data privacy considerations

## 🔍 Retrieval & Query System

```python
def retrieve_relevant_context(query, vec_db, k=5):
    if vec_db != None:
        docs = vec_db.similarity_search(query, k=k)
        return docs
    return None
```

The system uses **Approximate Nearest Neighbor (ANN)** search for efficient retrieval:
- Clusters similar embeddings using Locality Sensitive Hashing (LSH)
- Searches within relevant buckets for query matches
- Returns top-k most relevant document chunks
- the challenges of specifying the top-k number
  - make mixture of fixed and semantic chunking based on DI
  - can do that manually dataset for question and expected answer then testing LLMs to check whether it could output the same answer or not. 
## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit langchain openai azure-ai-documentintelligence
pip install faiss-cpu PyPDF2 easyocr python-dotenv streamlit-feedback
```

### Environment Variables
```env
AZURE_DI_ENDPOINT=your_document_intelligence_endpoint
AZURE_DI_KEY=your_document_intelligence_key
AI_SEARCH_API=your_azure_search_api_key
AI_SEARCH_END=your_azure_search_endpoint
OPENAI_API_KEY=your_openai_api_key
PINECONE_APIKEY=your_pinecone_api_key
```

### Running the Application
```bash
streamlit run filename.py
```

## 📱 Features

- **Multi-format Support**: PDF, images (PNG, JPG, TIFF)
- **Handwriting Recognition**: Processes handwritten medical notes
- **Dual Database Support**: Local FAISS and cloud Azure AI Search
- **Interactive Chat Interface**: Query documents using natural language
- **Configurable Parameters**: Adjust temperature, model selection, history length
- **Real-time Processing**: Stream responses with live feedback

## 🔧 Configuration Options

### Sidebar Controls
- **Temperature**: Controls response creativity (0.0-1.0)
- **Model Selection**: GPT-4o, GPT-4.1, Gemini, Llama, Mistral, Claude
- **Max History Length**: Conversation context window
- **Database Operations**: Create, Load, Save, Delete vector databases

## 🏥 Medical Use Cases

1. **Clinical Record Analysis**: Extract patient history, symptoms, diagnoses
2. **Prescription Processing**: Parse medication details and dosages
3. **Report Summarization**: Generate insights from medical reports
4. **Knowledge Base Search**: Find relevant medical information quickly

## ⚠️ Known Issues & Limitations

1. **Handwriting Accuracy**: Document Intelligence may struggle with poor handwriting
2. **Context Fragmentation**: Chunking may split related medical information
3. **Memory Management**: Large documents may cause performance issues
4. **Language Support**: Primarily optimized for English medical documents


---

**Note**: This application is for demonstration purposes. Always verify medical information with qualified healthcare professionals.
