"""
Document Processor Module

This module handles downloading, loading, and processing of PDF documents
for the RBI chatbot. It includes functionality for:
- Downloading PDF from URL
- Loading PDF content
- Text chunking for vector storage
"""

import requests
import pathlib
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


class DocumentProcessor:
    """Handles PDF document processing and vector store creation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def download_pdf(self, url: str, output_path: str) -> bool:
        """
        Download PDF from URL to local file.
        
        Args:
            url: URL of the PDF to download
            output_path: Local path to save the PDF
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            print(f"Downloading PDF from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write PDF content to file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF downloaded successfully to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading PDF: {str(e)}")
            return False
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF documents using PyPDFLoader.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for vector storage.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return []
    
    def create_vector_store(self, chunks: List[Document], embeddings_model: str = "models/embedding-001") -> FAISS:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            embeddings_model: Name of the embeddings model to use
            
        Returns:
            FAISS vector store
        """
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            print("Vector store created successfully")
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, vectorstore: FAISS, save_path: str):
        """
        Save vector store to local directory.
        
        Args:
            vectorstore: FAISS vector store to save
            save_path: Directory path to save the vector store
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            vectorstore.save_local(save_path)
            print(f"Vector store saved to {save_path}")
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, load_path: str, embeddings_model: str = "models/embedding-001") -> FAISS:
        """
        Load vector store from local directory.
        
        Args:
            load_path: Directory path to load the vector store from
            embeddings_model: Name of the embeddings model to use
            
        Returns:
            FAISS vector store
        """
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Vector store loaded from {load_path}")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            raise
    
    def process_rbi_document(self, pdf_url: str, pdf_path: str, vector_store_path: str) -> FAISS:
        """
        Complete pipeline to process RBI document from URL to vector store.
        
        Args:
            pdf_url: URL of the RBI PDF document
            pdf_path: Local path to save the PDF
            vector_store_path: Path to save the vector store
            
        Returns:
            FAISS vector store
        """
        # Download PDF
        if not self.download_pdf(pdf_url, pdf_path):
            raise Exception("Failed to download PDF")
        
        # Load and process PDF
        documents = self.load_pdf(pdf_path)
        if not documents:
            raise Exception("Failed to load PDF documents")
        
        # Split into chunks
        chunks = self.split_documents(documents)
        if not chunks:
            raise Exception("Failed to create document chunks")
        
        # Create vector store
        vectorstore = self.create_vector_store(chunks)
        
        # Save vector store
        self.save_vector_store(vectorstore, vector_store_path)
        
        return vectorstore


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # RBI document URL
    rbi_url = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF"
    pdf_path = "data/rbi_notification.pdf"
    vector_store_path = "data/rbi_faiss_index"
    
    try:
        vectorstore = processor.process_rbi_document(rbi_url, pdf_path, vector_store_path)
        print("Document processing completed successfully!")
    except Exception as e:
        print(f"Error processing document: {str(e)}")