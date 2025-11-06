"""
RBI Chatbot Module

This module implements the core chatbot functionality using LangChain
and Google Gemini API for answering questions about RBI documents.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from document_processor import DocumentProcessor


class RBIChatbot:
    """
    RBI Chatbot that can answer questions about RBI documents using RAG.
    """
    
    def __init__(
        self,
        vector_store_path: str,
        model_name: str = "gemini-pro",
        temperature: float = 0.0,
        k: int = 4
    ):
        """
        Initialize the RBI Chatbot.
        
        Args:
            vector_store_path: Path to the FAISS vector store
            model_name: Name of the Gemini model to use
            temperature: Temperature for text generation
            k: Number of documents to retrieve for context
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        
        # Initialize components
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.memory = None
        
        # Initialize the chatbot
        self._initialize()
    
    def _initialize(self):
        """Initialize all chatbot components."""
        try:
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
            print(f"Initialized {self.model_name} LLM")
            
            # Load vector store
            self._load_vector_store()
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )
            print(f"Created retriever with k={self.k}")
            
            # Initialize memory for conversation history
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            print("Chatbot initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            raise
    
    def _load_vector_store(self):
        """Load the FAISS vector store."""
        try:
            if not os.path.exists(self.vector_store_path):
                raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vectorstore = FAISS.load_local(
                self.vector_store_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {self.vector_store_path}")
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot and get an answer with sources.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer, source documents, and metadata
        """
        try:
            if not self.qa_chain:
                raise Exception("Chatbot not properly initialized")
            
            # Get response from the chain
            response = self.qa_chain({"question": question})
            
            # Extract source documents
            source_docs = response.get("source_documents", [])
            sources = []
            for doc in source_docs:
                sources.append({
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            result = {
                "question": question,
                "answer": response["answer"],
                "sources": sources,
                "confidence": "high" if len(source_docs) >= 2 else "medium"
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": "low"
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation turns with questions and answers
        """
        try:
            if not self.memory:
                return []
            
            messages = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    history.append({
                        "question": human_msg.content,
                        "answer": ai_msg.content
                    })
            
            return history
            
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        try:
            if self.memory:
                self.memory.clear()
                print("Conversation history cleared")
        except Exception as e:
            print(f"Error clearing conversation history: {str(e)}")
    
    def save_conversation_to_file(self, file_path: str):
        """
        Save conversation history to a text file.
        
        Args:
            file_path: Path to save the conversation log
        """
        try:
            history = self.get_conversation_history()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("RBI Chatbot Conversation Log\n")
                f.write("=" * 50 + "\n\n")
                
                for i, turn in enumerate(history, 1):
                    f.write(f"Turn {i}:\n")
                    f.write(f"Q: {turn['question']}\n")
                    f.write(f"A: {turn['answer']}\n")
                    f.write("-" * 30 + "\n\n")
            
            print(f"Conversation saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
    
    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a query without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (optional)
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.retriever:
                raise Exception("Retriever not initialized")
            
            # Use provided k or default
            search_k = k if k is not None else self.k
            
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(query)[:search_k]
            
            relevant_docs = []
            for doc in docs:
                relevant_docs.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "relevance_score": "N/A"  # FAISS doesn't provide scores by default
                })
            
            return relevant_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []


def create_chatbot_from_pdf(
    pdf_url: str,
    pdf_path: str,
    vector_store_path: str,
    force_recreate: bool = False
) -> RBIChatbot:
    """
    Create a chatbot from a PDF URL, handling document processing if needed.
    
    Args:
        pdf_url: URL of the PDF document
        pdf_path: Local path to save the PDF
        vector_store_path: Path to save/load the vector store
        force_recreate: Whether to force recreation of the vector store
        
    Returns:
        Initialized RBIChatbot instance
    """
    # Check if vector store exists
    if not os.path.exists(vector_store_path) or force_recreate:
        print("Creating vector store from PDF...")
        processor = DocumentProcessor()
        processor.process_rbi_document(pdf_url, pdf_path, vector_store_path)
    else:
        print("Using existing vector store...")
    
    # Create and return chatbot
    return RBIChatbot(vector_store_path)


if __name__ == "__main__":
    # Example usage
    rbi_url = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF"
    pdf_path = "data/rbi_notification.pdf"
    vector_store_path = "data/rbi_faiss_index"
    
    try:
        # Create chatbot
        chatbot = create_chatbot_from_pdf(rbi_url, pdf_path, vector_store_path)
        
        # Test with some questions
        test_questions = [
            "What is NBFC?",
            "Who regulates NBFCs?",
            "What are the requirements for NBFC registration?"
        ]
        
        print("\n" + "="*50)
        print("RBI Chatbot Test")
        print("="*50)
        
        for question in test_questions:
            print(f"\nQ: {question}")
            result = chatbot.ask_question(question)
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Sources: {len(result['sources'])} documents")
            print("-" * 30)
        
        # Save conversation
        chatbot.save_conversation_to_file("data/chat_log.txt")
        
    except Exception as e:
        print(f"Error: {str(e)}")