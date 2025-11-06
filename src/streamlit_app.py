"""
RBI Chatbot Streamlit Web Interface

This module provides a web interface for the RBI chatbot using Streamlit.
It offers an intuitive chat interface with conversation history and source display.
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot import create_chatbot_from_pdf, RBIChatbot


def setup_environment():
    """Set up environment variables and API keys."""
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        st.error("GEMINI_API_KEY not found in environment variables.")
        st.error("Please set your Gemini API key in the .env file.")
        st.stop()
    
    os.environ["GOOGLE_API_KEY"] = gemini_key
    
    # Optional LangSmith setup
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rbi-chatbot")


@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot (cached to avoid reinitialization)."""
    # Use demo chatbot instead of full version to avoid PDF download issues
    try:
        from demo import MockRBIChatbot
        with st.spinner("Initializing RBI Chatbot Demo..."):
            chatbot = MockRBIChatbot()
        return chatbot
    except Exception as e:
        st.error(f"Error initializing demo chatbot: {str(e)}")
        raise


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="RBI Chatbot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup environment
    setup_environment()
    
    # Main title
    st.title("🏦 RBI Chatbot")
    st.markdown("Ask questions about RBI documents and regulations")
    
    # Initialize chatbot
    try:
        chatbot = initialize_chatbot()
        st.success("Chatbot initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **🎯 DEMO MODE ACTIVE**
        
        This chatbot can answer questions about:
        - NBFC regulations
        - RBI policies and guidelines
        - Banking and financial regulations
        - Compliance requirements
        
        **Note:** Using predefined responses for demonstration
        """)
        
        st.header("Sample Questions")
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
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.current_question = question
        
        st.header("Actions")
        if st.button("Clear Conversation"):
            if 'conversation_history' in st.session_state:
                del st.session_state.conversation_history
            try:
                chatbot.clear_conversation_history()
            except:
                pass  # Demo chatbot might not have this method
            st.rerun()
        
        if st.button("Download Conversation"):
            if 'conversation_history' in st.session_state:
                conversation_text = "\n".join([
                    f"Q: {q}\nA: {a}\n" + "-"*50 + "\n"
                    for q, a in st.session_state.conversation_history
                ])
                st.download_button(
                    label="Download as TXT",
                    data=conversation_text,
                    file_name="rbi_chatbot_conversation.txt",
                    mime="text/plain"
                )
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Question input
        question = st.text_input(
            "Ask your question:",
            value=st.session_state.current_question,
            placeholder="e.g., What is NBFC?",
            key="question_input"
        )
        
        # Clear the current question after displaying it
        if st.session_state.current_question:
            st.session_state.current_question = ""
        
        # Submit button
        if st.button("Ask Question", type="primary") or question:
            if question.strip():
                with st.spinner("Getting answer..."):
                    try:
                        # Get response from chatbot
                        result = chatbot.ask_question(question)
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append((question, result))
                        
                        # Display current response
                        st.success("✅ Answer received!")
                        
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.header("Conversation History")
            
            # Display conversations in reverse order (most recent first)
            for i, (q, result) in enumerate(reversed(st.session_state.conversation_history)):
                turn_number = len(st.session_state.conversation_history) - i
                
                with st.expander(f"Turn {turn_number}: {q[:60]}...", expanded=(i == 0)):
                    # Question
                    st.markdown(f"**Question:** {q}")
                    
                    # Answer
                    st.markdown(f"**Answer:** {result['answer']}")
                    
                    # Metadata
                    col_conf, col_sources = st.columns(2)
                    with col_conf:
                        confidence_color = {
                            "high": "🟢",
                            "medium": "🟡", 
                            "low": "🔴"
                        }.get(result.get('confidence', 'unknown'), "⚪")
                        st.markdown(f"**Confidence:** {confidence_color} {result.get('confidence', 'Unknown')}")
                    
                    with col_sources:
                        st.markdown(f"**Sources:** {len(result.get('sources', []))} documents")
                    
                    # Show sources if available
                    if result.get('sources'):
                        with st.expander("📄 Source Documents"):
                            for j, source in enumerate(result['sources'], 1):
                                st.markdown(f"**Source {j}:**")
                                st.markdown(f"- Page: {source.get('page', 'Unknown')}")
                                st.markdown(f"- Content: {source.get('content', 'No content available')[:200]}...")
                                st.markdown("---")
    
    with col2:
        st.header("Statistics")
        
        if st.session_state.conversation_history:
            total_questions = len(st.session_state.conversation_history)
            
            # Calculate statistics
            high_conf = sum(1 for _, result in st.session_state.conversation_history 
                          if result.get('confidence') == 'high')
            with_sources = sum(1 for _, result in st.session_state.conversation_history 
                             if result.get('sources'))
            avg_sources = sum(len(result.get('sources', [])) 
                            for _, result in st.session_state.conversation_history) / total_questions
            
            # Display statistics
            st.metric("Total Questions", total_questions)
            st.metric("High Confidence", f"{high_conf}/{total_questions}")
            st.metric("With Sources", f"{with_sources}/{total_questions}")
            st.metric("Avg Sources", f"{avg_sources:.1f}")
            
            # Confidence distribution
            st.subheader("Confidence Distribution")
            confidence_counts = {}
            for _, result in st.session_state.conversation_history:
                conf = result.get('confidence', 'unknown')
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            for conf, count in confidence_counts.items():
                percentage = (count / total_questions) * 100
                st.progress(percentage / 100, text=f"{conf.title()}: {count} ({percentage:.1f}%)")
        
        else:
            st.info("Start asking questions to see statistics!")
        
        # Additional information
        st.header("How to Use")
        st.markdown("""
        1. **Type your question** in the text input above
        2. **Click "Ask Question"** or press Enter
        3. **View the answer** and source documents
        4. **Use sample questions** from the sidebar for quick testing
        5. **Download conversation** history when done
        """)
        
        st.header("Features")
        st.markdown("""
        - 🤖 **AI-powered answers** using Google Gemini
        - 📚 **Source documentation** for transparency
        - 💾 **Conversation history** tracking
        - 📊 **Real-time statistics** display
        - 📱 **Responsive design** for all devices
        """)


if __name__ == "__main__":
    main()