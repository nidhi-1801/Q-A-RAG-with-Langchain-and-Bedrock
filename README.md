Overview
This application implements a Retrieval-Augmented Generation (RAG) system that enables users to ask questions about PDF documents. It uses AWS Bedrock for embeddings and language models, FAISS for vector storage, and Streamlit for the user interface.
Architecture Components
1. Data Ingestion Layer

PDF Loader: Reads all PDF files from the data/ directory
Text Splitter: Breaks documents into chunks of 1000 characters with 200-character overlap
Purpose: Prepares documents for embedding and retrieval

2. Embedding Layer

Model: Amazon Titan Embeddings (amazon.titan-embed-text-v1)
Function: Converts text chunks into vector representations
Purpose: Enables semantic search capabilities

3. Vector Store

Technology: FAISS (Facebook AI Similarity Search)
Storage: Local filesystem (faiss_index/)
Function: Stores and retrieves document embeddings efficiently
Search Strategy: Top-k similarity search (k=3)

4. Language Model Layer

Primary Model: Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0)
Configuration:

Max tokens: 512
Temperature: 0.7


Alternative: Llama 3 8B (implemented but not used in UI)

5. Retrieval QA Chain

Chain Type: "Stuff" (concatenates retrieved documents)
Retrieval: Fetches top 3 relevant chunks
Prompt Engineering: Custom template with context injection

6. User Interface

Framework: Streamlit
Features:

Question input field
Vector store update button
Real-time answer generation
Custom pink/gradient theme



System Workflow
Phase 1: Indexing (One-time or on-demand)

User clicks "Update Vectors" button
System loads all PDFs from data/ directory
Documents are split into manageable chunks
Each chunk is embedded using Titan Embeddings
Embeddings are stored in FAISS index
Index is saved to local filesystem

Phase 2: Query Processing (Real-time)

User enters a question and clicks "Ask Claude"
Question is embedded using the same Titan model
FAISS retrieves top 3 most similar document chunks
Retrieved chunks are injected into a prompt template
Claude 3 Sonnet generates an answer based on context
Answer is displayed to the user

Key Features
✅ Advantages

Contextual Answers: Responds based on your specific documents
Source Grounding: Reduces hallucinations by using retrieved context
Scalable: Can handle multiple large PDF documents
Cost-Effective: Uses serverless AWS Bedrock (pay-per-use)
User-Friendly: Simple, themed Streamlit interface
