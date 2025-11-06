"""
RBI Chatbot Demo - Simplified Version

This demo version simulates the chatbot functionality without requiring 
embeddings API calls, allowing you to test the interface and logic.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, List
import json


class MockRBIChatbot:
    """
    Mock RBI Chatbot for demonstration purposes.
    Uses predefined Q&A pairs instead of vector search.
    """
    
    def __init__(self):
        """Initialize the mock chatbot with predefined responses."""
        self.conversation_history = []
        self.mock_data = self._load_mock_responses()
        print("Mock RBI Chatbot initialized successfully!")
    
    def _load_mock_responses(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined responses for common RBI questions."""
        return {
            "what is nbfc": {
                "answer": "A Non-Banking Financial Company (NBFC) is a company registered under the Companies Act, 1956/2013 engaged in the business of loans and advances, acquisition of shares/stocks/bonds/debentures/securities issued by Government or local authority or other marketable securities of a like nature, leasing, hire-purchase, insurance business, chit business but does not include any institution whose principal business is that of agriculture activity, industrial activity, purchase or sale of any goods (other than securities) or providing any services and sale/purchase/construction of immovable property.",
                "sources": [
                    {"page": "5", "source": "RBI NBFC Guidelines", "content": "Definition of NBFC as per RBI Act..."},
                    {"page": "12", "source": "RBI NBFC Guidelines", "content": "Classification criteria for NBFCs..."}
                ],
                "confidence": "high"
            },
            "who regulates nbfc": {
                "answer": "NBFCs are regulated by the Reserve Bank of India (RBI) under the RBI Act, 1934 and various directions issued by RBI from time to time. RBI has the authority to supervise, inspect, and regulate the functioning of NBFCs to ensure financial stability and protect the interests of depositors and the general public.",
                "sources": [
                    {"page": "8", "source": "RBI NBFC Guidelines", "content": "Regulatory framework for NBFCs under RBI Act..."},
                    {"page": "15", "source": "RBI NBFC Guidelines", "content": "Supervisory powers of RBI over NBFCs..."}
                ],
                "confidence": "high"
            },
            "minimum capital requirement": {
                "answer": "The minimum Net Owned Fund (NOF) requirement for NBFCs is Rs. 2 crore. However, for specific categories like NBFC-MFI (Micro Finance Institution), the requirement may be different. All NBFCs must maintain this minimum capital throughout their operations and cannot operate if their NOF falls below this threshold.",
                "sources": [
                    {"page": "23", "source": "RBI NBFC Guidelines", "content": "Capital adequacy requirements for NBFCs..."},
                    {"page": "45", "source": "RBI NBFC Guidelines", "content": "NOF maintenance requirements..."}
                ],
                "confidence": "high"
            },
            "types of nbfc": {
                "answer": "There are various types of NBFCs including: 1) Asset Finance Company (AFC), 2) Investment Company (IC), 3) Loan Company (LC), 4) Infrastructure Finance Company (IFC), 5) Systemically Important Non-Deposit taking NBFC (NBFC-ND-SI), 6) Deposit taking NBFC (NBFC-D), 7) Micro Finance Institution (NBFC-MFI), 8) NBFC-Account Aggregator (NBFC-AA), and others based on their primary business activities.",
                "sources": [
                    {"page": "18", "source": "RBI NBFC Guidelines", "content": "Classification of NBFCs by business type..."},
                    {"page": "32", "source": "RBI NBFC Guidelines", "content": "Detailed description of NBFC categories..."}
                ],
                "confidence": "high"
            },
            "deposit acceptance": {
                "answer": "Only NBFCs holding Certificate of Registration with authorization to accept deposits can accept/renew public deposits. However, no NBFC shall accept any public deposit unless it has minimum investment grade credit rating for fixed deposits programme from a credit rating agency registered with SEBI. Most NBFCs today are non-deposit taking NBFCs.",
                "sources": [
                    {"page": "67", "source": "RBI NBFC Guidelines", "content": "Deposit acceptance norms for NBFCs..."},
                    {"page": "72", "source": "RBI NBFC Guidelines", "content": "Credit rating requirements for deposit taking..."}
                ],
                "confidence": "high"
            },
            "registration process": {
                "answer": "To apply for NBFC registration, an entity must submit application to RBI along with required documents including certificate of incorporation, memorandum and articles of association, business plan, details of directors and substantial shareholders, financial projections, compliance officer details, and other specified documents as per RBI guidelines. The process typically takes 4-6 months.",
                "sources": [
                    {"page": "89", "source": "RBI NBFC Guidelines", "content": "Step-by-step registration process..."},
                    {"page": "95", "source": "RBI NBFC Guidelines", "content": "Required documentation checklist..."}
                ],
                "confidence": "high"
            },
            "prudential norms": {
                "answer": "Prudential norms for NBFCs include: 1) Capital Adequacy Ratio (CAR) of minimum 15%, 2) Asset classification and provisioning norms similar to banks, 3) Exposure norms limiting investment in single borrower/group, 4) Liquidity requirements, 5) Corporate governance standards, 6) Risk management framework, 7) Regular auditing and compliance requirements.",
                "sources": [
                    {"page": "56", "source": "RBI NBFC Guidelines", "content": "Prudential norms framework for NBFCs..."},
                    {"page": "78", "source": "RBI NBFC Guidelines", "content": "Risk management requirements..."}
                ],
                "confidence": "high"
            },
            "systemically important": {
                "answer": "A systemically important NBFC (NBFC-SI) is defined as an NBFC having asset size of Rs. 500 crore or above as shown in the last audited balance sheet. These NBFCs are subject to enhanced regulatory oversight, stricter prudential norms, and additional reporting requirements due to their potential systemic impact on the financial system.",
                "sources": [
                    {"page": "34", "source": "RBI NBFC Guidelines", "content": "Definition of systemically important NBFCs..."},
                    {"page": "87", "source": "RBI NBFC Guidelines", "content": "Enhanced supervision framework..."}
                ],
                "confidence": "high"
            },
            "compliance requirements": {
                "answer": "NBFCs must comply with various requirements including: 1) Regular submission of returns to RBI, 2) Maintenance of statutory reserves, 3) Adherence to fair practices code, 4) KYC and AML compliance, 5) Customer grievance redressal mechanism, 6) Board oversight and governance, 7) Audit and internal controls, 8) Regulatory reporting and disclosures.",
                "sources": [
                    {"page": "112", "source": "RBI NBFC Guidelines", "content": "Compliance framework for NBFCs..."},
                    {"page": "134", "source": "RBI NBFC Guidelines", "content": "Reporting and disclosure requirements..."}
                ],
                "confidence": "high"
            },
            "lending practices": {
                "answer": "NBFCs must follow fair lending practices including: 1) Transparent pricing and no hidden charges, 2) Proper disclosure of terms and conditions, 3) Fair debt collection practices, 4) Grievance redressal mechanism, 5) Board-approved loan policy, 6) Credit evaluation procedures, 7) Interest rate guidelines as per RBI directions.",
                "sources": [
                    {"page": "156", "source": "RBI NBFC Guidelines", "content": "Fair practices code for NBFCs..."},
                    {"page": "167", "source": "RBI NBFC Guidelines", "content": "Lending guidelines and practices..."}
                ],
                "confidence": "high"
            },
            "asset classification": {
                "answer": "NBFCs must classify assets as: 1) Standard Assets - performing assets with no default, 2) Sub-standard Assets - assets with default for more than 90 days, 3) Doubtful Assets - assets remaining sub-standard for more than 18 months, 4) Loss Assets - assets identified as non-recoverable. Provisioning requirements apply based on classification.",
                "sources": [
                    {"page": "189", "source": "RBI NBFC Guidelines", "content": "Asset classification norms..."},
                    {"page": "201", "source": "RBI NBFC Guidelines", "content": "Provisioning requirements for NBFCs..."}
                ],
                "confidence": "high"
            },
            "rbi inspection": {
                "answer": "RBI conducts regular inspections of NBFCs to assess: 1) Compliance with regulatory guidelines, 2) Financial health and risk management, 3) Corporate governance practices, 4) Customer service and fair practices, 5) Internal controls and audit systems. Inspection frequency depends on the size and risk profile of the NBFC.",
                "sources": [
                    {"page": "234", "source": "RBI NBFC Guidelines", "content": "RBI inspection framework..."},
                    {"page": "245", "source": "RBI NBFC Guidelines", "content": "Supervisory process and procedures..."}
                ],
                "confidence": "high"
            },
            "microfinance": {
                "answer": "NBFC-MFIs (Micro Finance Institutions) are NBFCs engaged in microfinance business with specific regulations: 1) Minimum 85% of assets in qualifying microfinance loans, 2) Loan size limits for individual borrowers, 3) Interest rate guidelines, 4) Client protection measures, 5) Recovery practices regulations, 6) Specific capital and governance requirements.",
                "sources": [
                    {"page": "267", "source": "RBI NBFC Guidelines", "content": "NBFC-MFI regulations and guidelines..."},
                    {"page": "289", "source": "RBI NBFC Guidelines", "content": "Microfinance operational guidelines..."}
                ],
                "confidence": "high"
            },
            "housing finance": {
                "answer": "Housing Finance Companies (HFCs) are specialized NBFCs engaged in housing finance business. They must maintain at least 50% of their assets in housing finance activities. HFCs are subject to specific regulations regarding: loan-to-value ratios, risk weights, provisioning norms, and priority sector lending requirements.",
                "sources": [
                    {"page": "301", "source": "RBI NBFC Guidelines", "content": "Housing finance company regulations..."},
                    {"page": "315", "source": "RBI NBFC Guidelines", "content": "Housing loan guidelines and norms..."}
                ],
                "confidence": "high"
            },
            "penalties violations": {
                "answer": "RBI can impose penalties on NBFCs for violations including: 1) Monetary penalties up to Rs. 1 crore per violation, 2) Restrictions on business activities, 3) Cancellation of Certificate of Registration, 4) Prohibition from accepting deposits, 5) Replacement of management, 6) Appointment of administrator. Penalties depend on the nature and severity of violations.",
                "sources": [
                    {"page": "334", "source": "RBI NBFC Guidelines", "content": "Penalty framework for NBFCs..."},
                    {"page": "345", "source": "RBI NBFC Guidelines", "content": "Enforcement actions and procedures..."}
                ],
                "confidence": "high"
            }
        }
    
    def _find_best_match(self, question: str) -> str:
        """Find the best matching response key for a question."""
        question_lower = question.lower()
        
        # Define keyword mappings for better matching
        keyword_mappings = {
            "what is nbfc": ["what", "nbfc", "definition", "define", "meaning", "explain"],
            "who regulates nbfc": ["regulate", "rbi", "supervision", "supervise", "oversee", "control"],
            "minimum capital requirement": ["capital", "minimum", "nof", "fund", "requirement", "money"],
            "types of nbfc": ["types", "categories", "kinds", "classification", "different"],
            "deposit acceptance": ["deposit", "accept", "public", "savings", "money"],
            "registration process": ["registration", "register", "apply", "application", "process", "how to"],
            "prudential norms": ["prudential", "norms", "guidelines", "standards", "rules", "framework"],
            "systemically important": ["systemically", "important", "si", "large", "size", "asset"],
            "compliance requirements": ["compliance", "requirements", "rules", "obligations", "must", "need"],
            "lending practices": ["lending", "loan", "credit", "borrowing", "finance", "practices"],
            "asset classification": ["asset", "classification", "npa", "standard", "doubtful", "provision"],
            "rbi inspection": ["inspection", "audit", "examination", "review", "check", "assess"],
            "microfinance": ["microfinance", "mfi", "micro", "small", "rural", "poor"],
            "housing finance": ["housing", "home", "property", "real estate", "mortgage"],
            "penalties violations": ["penalty", "violation", "punishment", "fine", "action", "breach"]
        }
        
        # Calculate match scores for each topic
        best_match = None
        best_score = 0
        
        for topic, keywords in keyword_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            
            # Boost score if multiple keywords match
            if score > 1:
                score *= 1.5
            
            if score > best_score:
                best_score = score
                best_match = topic
        
        # Additional specific patterns
        patterns = {
            "what is nbfc": ["what.*nbfc", "define.*nbfc", "meaning.*nbfc", "explain.*nbfc"],
            "who regulates nbfc": ["who.*regulate", "rbi.*regulate", "who.*control"],
            "minimum capital requirement": ["minimum.*capital", "capital.*requirement", "nof.*requirement"],
            "types of nbfc": ["types.*nbfc", "categories.*nbfc", "kinds.*nbfc"],
            "deposit acceptance": ["deposit.*accept", "public.*deposit", "accept.*deposit"],
            "registration process": ["how.*register", "registration.*process", "apply.*nbfc"],
            "prudential norms": ["prudential.*norm", "regulatory.*framework"],
            "systemically important": ["systemically.*important", "nbfc.*si"],
            "compliance requirements": ["compliance.*requirement", "regulatory.*requirement"],
            "lending practices": ["lending.*practice", "loan.*practice", "fair.*practice"],
            "asset classification": ["asset.*classification", "npa.*classification"],
            "rbi inspection": ["rbi.*inspection", "audit.*nbfc", "examination.*nbfc"],
            "microfinance": ["micro.*finance", "mfi.*regulation"],
            "housing finance": ["housing.*finance", "home.*loan"],
            "penalties violations": ["penalty.*nbfc", "violation.*nbfc", "punishment.*nbfc"]
        }
        
        import re
        for topic, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, question_lower):
                    return topic
        
        # If we found a match with keywords, return it
        if best_match and best_score >= 1:
            return best_match
        
        return None
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return a mock response.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer, sources, and confidence
        """
        # Find matching response
        match_key = self._find_best_match(question)
        
        if match_key and match_key in self.mock_data:
            response_data = self.mock_data[match_key]
            result = {
                "question": question,
                "answer": response_data["answer"],
                "sources": response_data["sources"],
                "confidence": response_data["confidence"]
            }
        else:
            # Generic response for unmatched questions with better guidance
            result = {
                "question": question,
                "answer": f"I understand you're asking about '{question}', but I don't have specific information on that exact topic in my current knowledge base. However, I can help you with questions about:\n\n📌 **NBFC Basics**: Definition, types, and characteristics\n📌 **Regulation**: Who regulates NBFCs and how\n📌 **Capital Requirements**: Minimum capital and NOF requirements\n📌 **Registration**: How to register an NBFC\n📌 **Compliance**: Prudential norms and requirements\n📌 **Operations**: Deposit acceptance, lending practices\n📌 **Supervision**: RBI inspection and penalties\n📌 **Specialized NBFCs**: Microfinance, housing finance\n\nTry asking something like: 'What are prudential norms for NBFCs?' or 'How does RBI supervise NBFCs?' or 'What are the penalties for NBFC violations?'",
                "sources": [],
                "confidence": "low"
            }
        
        # Add to conversation history
        self.conversation_history.append((question, result))
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [{"question": q, "answer": r["answer"]} for q, r in self.conversation_history]
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared")
    
    def save_conversation_to_file(self, file_path: str):
        """Save conversation to file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("RBI Chatbot Demo - Conversation Log\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (question, result) in enumerate(self.conversation_history, 1):
                f.write(f"Turn {i}:\n")
                f.write(f"Q: {question}\n")
                f.write(f"A: {result['answer']}\n")
                f.write(f"Confidence: {result['confidence']}\n")
                f.write(f"Sources: {len(result['sources'])}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Conversation saved to {file_path}")


def interactive_demo_session(chatbot: MockRBIChatbot):
    """Run an interactive demo session."""
    print("\n" + "="*60)
    print("RBI Chatbot - DEMO MODE")
    print("="*60)
    print("This is a demo version with enhanced response coverage.")
    print("Try asking about: NBFC basics, regulation, capital, compliance, penalties, microfinance, etc.")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("Type 'history' to see conversation history.")
    print("Type 'save' to save conversation to file.")
    print("-"*60)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Thank you for using RBI Chatbot Demo.")
                break
            
            if question.lower() == 'history':
                history = chatbot.get_conversation_history()
                if history:
                    print("\nConversation History:")
                    print("-" * 30)
                    for i, turn in enumerate(history, 1):
                        print(f"{i}. Q: {turn['question']}")
                        print(f"   A: {turn['answer'][:100]}...")
                        print()
                else:
                    print("No conversation history available.")
                continue
            
            if question.lower() == 'save':
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/demo_chat_{timestamp}.txt"
                chatbot.save_conversation_to_file(filename)
                continue
            
            if not question:
                continue
            
            # Get response
            result = chatbot.ask_question(question)
            
            # Display response
            print(f"\nRBI Chatbot: {result['answer']}")
            print(f"\nConfidence: {result['confidence']}")
            print(f"Sources: {len(result['sources'])} documents")
            
            if result['sources']:
                show_sources = input("\nShow source details? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\nSource Documents:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. Page {source['page']}: {source['content']}")
                        print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thank you for using RBI Chatbot Demo.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main demo application."""
    print("RBI Chatbot Demo Application")
    print("=" * 50)
    print("⚠️  DEMO MODE: Using predefined responses")
    print("💡 This version doesn't require API quotas")
    
    try:
        # Initialize mock chatbot
        chatbot = MockRBIChatbot()
        
        # Main menu
        while True:
            print("\n" + "="*50)
            print("RBI Chatbot Demo - Main Menu")
            print("="*50)
            print("1. Interactive Chat Session")
            print("2. Run Sample Questions")
            print("3. Show Available Topics")
            print("4. Exit")
            
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == '1':
                interactive_demo_session(chatbot)
            
            elif choice == '2':
                sample_questions = [
                    "What is NBFC?",
                    "Who regulates NBFCs?",
                    "What are prudential norms for NBFCs?",
                    "How does RBI inspect NBFCs?",
                    "What are the penalties for violations?",
                    "What is microfinance regulation?",
                    "How are assets classified in NBFCs?",
                    "What are systemically important NBFCs?",
                    "What are the compliance requirements?",
                    "What are fair lending practices?"
                ]
                
                print("\nSample Questions Demo:")
                print("-" * 30)
                
                for i, question in enumerate(sample_questions, 1):
                    print(f"\n{i}. {question}")
                    result = chatbot.ask_question(question)
                    print(f"Answer: {result['answer'][:200]}...")
                    print(f"Confidence: {result['confidence']}")
                    input("Press Enter to continue...")
            
            elif choice == '3':
                print("\nAvailable Topics:")
                print("- NBFC definition and characteristics")
                print("- Regulatory framework and supervision")
                print("- Capital requirements and NOF")
                print("- Types and categories of NBFCs")
                print("- Deposit acceptance rules")
                print("- Registration process and requirements")
                print("- Prudential norms and compliance")
                print("- Asset classification and provisioning")
                print("- RBI inspection and supervision")
                print("- Penalties and enforcement actions")
                print("- Microfinance regulations")
                print("- Housing finance guidelines")
                print("- Fair lending practices")
                print("\nTry asking questions about these topics!")
            
            elif choice == '4':
                print("Thank you for using RBI Chatbot Demo!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()