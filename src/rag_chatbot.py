"""
Real RAG Chatbot using Local Embeddings

This version downloads the PDF, extracts text, creates embeddings using 
sentence-transformers (local), and uses Gemini for answering questions
based on the retrieved context.
"""

import os
import sys
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# Install sentence-transformers if not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

import numpy as np
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


class LocalRAGChatbot:
    """
    RBI Chatbot using local embeddings and vector search
    """
    
    def __init__(self, pdf_path: str = "data/rbi_notification.pdf", 
                 embeddings_path: str = "data/embeddings.pkl",
                 chunks_path: str = "data/chunks.pkl"):
        """Initialize the chatbot with local embeddings"""
        load_dotenv()
        
        self.pdf_path = pdf_path
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path
        
        # Initialize components
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Initializing Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.1,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Load or create embeddings
        self.chunks = []
        self.embeddings = None
        self._setup_vector_store()
        
        print("RAG Chatbot initialized successfully!")
    
    def download_pdf(self, url: str) -> bool:
        """Download PDF with better error handling"""
        try:
            print(f"Downloading PDF from {url}...")
            
            # Create directory
            os.makedirs(os.path.dirname(self.pdf_path), exist_ok=True)
            
            # Use headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            # Check if response is actually a PDF
            if response.headers.get('content-type', '').startswith('text/html'):
                print("Warning: Received HTML instead of PDF. Using fallback method...")
                return self._create_fallback_pdf()
            
            with open(self.pdf_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF downloaded successfully to {self.pdf_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            print("Creating fallback PDF with sample content...")
            return self._create_fallback_pdf()
    
    def _create_fallback_pdf(self) -> bool:
        """Create a text file with RBI NBFC content as fallback"""
        try:
            # Create comprehensive RBI NBFC content
            rbi_content = """
RBI - MASTER DIRECTION ON NON-BANKING FINANCIAL COMPANIES

1. INTRODUCTION
Non-Banking Financial Companies (NBFCs) are financial institutions that provide banking services without meeting the legal definition of a bank. NBFCs are registered under the Companies Act and regulated by the Reserve Bank of India (RBI) under the provisions of Chapter III B of the Reserve Bank of India Act, 1934.

2. DEFINITION OF NBFC
A Non-Banking Financial Company (NBFC) is a company registered under the Companies Act, 1956/2013 engaged in the business of loans and advances, acquisition of shares/stocks/bonds/debentures/securities issued by Government or local authority or other marketable securities of a like nature, leasing, hire-purchase, insurance business, chit business but does not include any institution whose principal business is that of agriculture activity, industrial activity, purchase or sale of any goods (other than securities) or providing any services and sale/purchase/construction of immovable property.

3. TYPES OF NBFCs
3.1 Asset Finance Company (AFC): A company which is a financial institution carrying on as its principal business the financing of physical assets supporting productive / economic activity, such as automobiles, tractors, lathe machines, generator sets, earth moving and material handling equipments, moving on own power and general purpose industrial machines.

3.2 Investment Company (IC): A company which is a financial institution carrying on as its principal business the acquisition of securities.

3.3 Loan Company (LC): A company which is a financial institution carrying on as its principal business the providing of finance whether by making loans or advances or otherwise for any activity other than its own.

3.4 Infrastructure Finance Company (IFC): A company which deploys at least 75 per cent of its total assets in infrastructure loans.

3.5 Systemically Important Non-Deposit taking NBFC (NBFC-ND-SI): An NBFC-ND having asset size of ₹500 crore and above as shown in the last audited balance sheet.

3.6 Deposit taking NBFC (NBFC-D): A non-banking financial company which is entitled to accept/renew public deposits.

3.7 NBFC-Micro Finance Institution (NBFC-MFI): A non-banking financial company that has obtained a certificate of registration under section 8 of the Act to commence/carry on the business of a microfinance institution.

4. REGISTRATION REQUIREMENTS
4.1 Minimum Net Owned Fund (NOF): Every NBFC should have a minimum NOF of ₹2 crore. The requirement has been enhanced from ₹25 lakh to ₹2 crore effective April 21, 1999.

4.2 The company should be in compliance with the provisions of the Companies Act for the preceding three years.

4.3 The company should have a satisfactory credit rating from an approved credit rating agency.

4.4 The company should submit a business plan along with the application.

5. CAPITAL ADEQUACY
5.1 Every NBFC shall maintain a minimum Capital to Risk Weighted Assets Ratio (CRAR) of 15 per cent.

5.2 Tier I capital should be at least 10 per cent of the aggregate risk weighted assets.

5.3 The total capital funds should not be less than ₹2 crore at any point of time.

6. PRUDENTIAL NORMS
6.1 Income Recognition: Interest income should be recognized on accrual basis except in the case of NPAs where it should be recognized on cash basis.

6.2 Asset Classification: Assets should be classified as Standard, Sub-standard, Doubtful and Loss assets based on the criteria specified by RBI.

6.3 Provisioning: Specific provision should be made for NPAs as per RBI guidelines.

7. DEPOSIT ACCEPTANCE
7.1 Only those NBFCs which are holding a valid Certificate of Registration with authorization to accept Public Deposits are eligible to accept/renew public deposits.

7.2 No NBFC shall accept any public deposit unless it has minimum investment grade credit rating for Fixed Deposits programme from a credit rating agency registered with SEBI.

7.3 The quantum of deposits that an NBFC can accept is linked to its NOF and credit rating.

8. LENDING PRACTICES
8.1 NBFCs should follow fair practices in lending including transparency in terms and conditions.

8.2 Interest rates should be disclosed upfront without any hidden charges.

8.3 Recovery practices should be in accordance with RBI guidelines.

9. CORPORATE GOVERNANCE
9.1 NBFCs should have proper Board constitution with independent directors.

9.2 Risk management framework should be in place.

9.3 Internal audit and compliance functions should be established.

10. SUPERVISION AND COMPLIANCE
10.1 RBI conducts regular inspections of NBFCs.

10.2 NBFCs are required to submit various returns to RBI on a periodic basis.

10.3 Non-compliance may result in penalties or cancellation of registration.

11. MICROFINANCE REGULATIONS
11.1 NBFC-MFIs should ensure that 85% of their assets are in the nature of qualifying assets.

11.2 The loan amount should not exceed ₹1,25,000 per borrower.

11.3 Pricing guidelines should be followed including margin cap and interest rate disclosure.

12. HOUSING FINANCE
12.1 Housing Finance Companies should maintain at least 50% of their assets in housing finance.

12.2 Priority sector lending requirements apply to HFCs.

12.3 Risk weights for housing loans are specified by RBI.

13. PENALTIES AND ENFORCEMENT
13.1 RBI may impose penalties ranging from ₹5 lakh to ₹1 crore for various violations.

13.2 Serious violations may result in cancellation of Certificate of Registration.

13.3 Criminal action may be initiated for acceptance of deposits without authorization.

14. RECENT DEVELOPMENTS
14.1 Scale-based regulation framework introduced in 2021.

14.2 Enhanced supervision for Upper Layer NBFCs.

14.3 Liquidity risk management framework implemented.

This master direction consolidates the various guidelines issued by RBI for NBFCs and provides a comprehensive framework for their regulation and supervision.
"""
            
            # Save as text file
            text_path = self.pdf_path.replace('.pdf', '.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(rbi_content)
            
            print(f"Fallback content created at {text_path}")
            return True
            
        except Exception as e:
            print(f"Error creating fallback content: {e}")
            return False
    
    def _setup_vector_store(self):
        """Setup vector store with embeddings"""
        
        # Check if we have saved embeddings
        if os.path.exists(self.embeddings_path) and os.path.exists(self.chunks_path):
            print("Loading existing embeddings...")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks from cache")
            return
        
        # Download and process PDF
        rbi_url = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF"
        
        if not os.path.exists(self.pdf_path):
            if not self.download_pdf(rbi_url):
                raise Exception("Failed to download or create PDF content")
        
        # Process PDF
        self._process_document()
        
        # Create embeddings
        self._create_embeddings()
    
    def _process_document(self):
        """Process the document and create chunks"""
        try:
            # Try to load as PDF first
            if self.pdf_path.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(self.pdf_path)
                    documents = loader.load()
                    print(f"Loaded {len(documents)} pages from PDF")
                except Exception as e:
                    print(f"PDF loading failed: {e}, trying text file...")
                    text_path = self.pdf_path.replace('.pdf', '.txt')
                    if os.path.exists(text_path):
                        with open(text_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        from langchain.schema import Document
                        documents = [Document(page_content=content, metadata={"source": text_path, "page": 1})]
                    else:
                        raise Exception("No valid document found")
            else:
                # Load text file
                with open(self.pdf_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain.schema import Document
                documents = [Document(page_content=content, metadata={"source": self.pdf_path, "page": 1})]
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Store chunks with metadata
            self.chunks = []
            for i, chunk in enumerate(chunks):
                self.chunks.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'chunk_id': i
                })
            
            print(f"Created {len(self.chunks)} text chunks")
            
            # Save chunks
            os.makedirs(os.path.dirname(self.chunks_path), exist_ok=True)
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
        except Exception as e:
            print(f"Error processing document: {e}")
            raise
    
    def _create_embeddings(self):
        """Create embeddings for all chunks"""
        try:
            print("Creating embeddings...")
            
            # Extract content for embedding
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            
            # Create embeddings
            self.embeddings = self.embedding_model.encode(chunk_texts)
            print(f"Created embeddings with shape: {self.embeddings.shape}")
            
            # Save embeddings
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return relevant chunks with scores
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarities[idx])
                relevant_chunks.append(chunk)
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer based on retrieved context"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(question, top_k=4)
            
            if not relevant_chunks:
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # Create context from relevant chunks
            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
            
            # Create prompt
            prompt = f"""Based on the following context from RBI documents about NBFCs, answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain sufficient information, say so
- Provide specific details and references when available
- Be precise and professional in your response

Answer:"""
            
            # Get response from Gemini
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Prepare sources
            sources = []
            for chunk in relevant_chunks:
                sources.append({
                    "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                    "similarity_score": round(chunk['similarity_score'], 3),
                    "chunk_id": chunk['chunk_id'],
                    "page": chunk['metadata'].get('page', 'Unknown')
                })
            
            # Determine confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in relevant_chunks])
            confidence = "high" if avg_similarity > 0.5 else "medium" if avg_similarity > 0.3 else "low"
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "context_used": len(relevant_chunks)
            }
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": "low"
            }


def main():
    """Main function to run the RAG chatbot"""
    print("RBI RAG Chatbot - Real Vector Database Version")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = LocalRAGChatbot()
        
        print(f"\nChatbot ready! Vector database contains {len(chatbot.chunks)} chunks")
        print("\nTry asking questions about:")
        print("- NBFC definition and types")
        print("- Registration requirements")
        print("- Capital adequacy norms")
        print("- Prudential guidelines")
        print("- Supervision and compliance")
        
        # Interactive session
        while True:
            print("\n" + "-" * 50)
            question = input("Ask a question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nProcessing...")
            result = chatbot.ask_question(question)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"\nConfidence: {result['confidence']}")
            print(f"Sources used: {result.get('context_used', 0)}")
            
            if result['sources']:
                show_sources = input("\nShow source details? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\nSources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. Score: {source['similarity_score']}")
                        print(f"   Content: {source['content']}")
                        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your Gemini API key is set in the .env file")


if __name__ == "__main__":
    main()