"""
Test Module for RBI Chatbot

This module contains unit tests for the RBI chatbot components.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid import errors when dependencies not installed
        try:
            from document_processor import DocumentProcessor
            self.processor = DocumentProcessor()
        except ImportError:
            self.skipTest("Dependencies not installed")
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        self.assertEqual(self.processor.chunk_size, 1000)
        self.assertEqual(self.processor.chunk_overlap, 200)
    
    @patch('requests.get')
    def test_download_pdf_success(self, mock_get):
        """Test successful PDF download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = b"fake pdf content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test download
        result = self.processor.download_pdf("http://test.com/file.pdf", "test.pdf")
        self.assertTrue(result)
    
    @patch('requests.get')
    def test_download_pdf_failure(self, mock_get):
        """Test failed PDF download."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        # Test download
        result = self.processor.download_pdf("http://test.com/file.pdf", "test.pdf")
        self.assertFalse(result)


class TestRBIChatbot(unittest.TestCase):
    """Test cases for RBIChatbot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from chatbot import RBIChatbot
            # Mock the initialization to avoid dependency issues
            with patch.object(RBIChatbot, '_initialize'):
                self.chatbot = RBIChatbot("fake/path")
        except ImportError:
            self.skipTest("Dependencies not installed")
    
    def test_initialization_parameters(self):
        """Test chatbot initialization parameters."""
        self.assertEqual(self.chatbot.vector_store_path, "fake/path")
        self.assertEqual(self.chatbot.model_name, "gemini-pro")
        self.assertEqual(self.chatbot.temperature, 0.0)
        self.assertEqual(self.chatbot.k, 4)
    
    def test_ask_question_error_handling(self):
        """Test error handling in ask_question method."""
        # Simulate qa_chain being None
        self.chatbot.qa_chain = None
        
        result = self.chatbot.ask_question("What is NBFC?")
        
        self.assertIn("error", result["answer"].lower())
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["confidence"], "low")


class TestEvaluationDataset(unittest.TestCase):
    """Test cases for evaluation dataset and metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from evaluate import RBIChatbotEvaluator
            # Mock the chatbot
            mock_chatbot = Mock()
            self.evaluator = RBIChatbotEvaluator(mock_chatbot)
        except ImportError:
            self.skipTest("Dependencies not installed")
    
    def test_evaluation_dataset_creation(self):
        """Test creation of evaluation dataset."""
        dataset = self.evaluator.create_evaluation_dataset()
        
        # Check dataset structure
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        # Check first item structure
        first_item = dataset[0]
        self.assertIn("question", first_item)
        self.assertIn("expected_answer", first_item)
        self.assertIn("category", first_item)
    
    def test_aggregate_metrics_calculation(self):
        """Test calculation of aggregate metrics."""
        # Mock results data
        mock_results = [
            {
                "qa_score": 0.8,
                "helpfulness_score": 0.9,
                "response_length": 200,
                "has_sources": True,
                "num_sources": 3,
                "confidence": "high",
                "category": "definition"
            },
            {
                "qa_score": 0.6,
                "helpfulness_score": 0.7,
                "response_length": 150,
                "has_sources": False,
                "num_sources": 0,
                "confidence": "medium",
                "category": "regulation"
            }
        ]
        
        metrics = self.evaluator._calculate_aggregate_metrics(mock_results)
        
        # Check calculated metrics
        self.assertEqual(metrics["avg_qa_score"], 0.7)
        self.assertEqual(metrics["avg_helpfulness_score"], 0.8)
        self.assertEqual(metrics["questions_with_sources"], 1)
        self.assertEqual(metrics["questions_without_sources"], 1)
        self.assertEqual(metrics["source_coverage_rate"], 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_environment_setup(self):
        """Test environment variable setup."""
        # Test .env.example file exists
        env_example_path = os.path.join(os.path.dirname(__file__), '..', '.env.example')
        self.assertTrue(os.path.exists(env_example_path))
        
        # Test required environment variables are documented
        with open(env_example_path, 'r') as f:
            content = f.read()
            self.assertIn("GEMINI_API_KEY", content)
            self.assertIn("LANGCHAIN_API_KEY", content)
    
    def test_project_structure(self):
        """Test project directory structure."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        # Check required directories exist
        required_dirs = ['src', 'data', 'tests', '.github']
        for dir_name in required_dirs:
            dir_path = os.path.join(project_root, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} should exist")
        
        # Check required files exist
        required_files = [
            'README.md',
            'requirements.txt',
            '.env.example',
            'src/main.py',
            'src/chatbot.py',
            'src/document_processor.py',
            'src/evaluate.py',
            'src/streamlit_app.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(project_root, file_path)
            self.assertTrue(os.path.exists(full_path), f"File {file_path} should exist")
    
    def test_requirements_file(self):
        """Test requirements.txt contains necessary dependencies."""
        requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        # Check for essential dependencies
        essential_packages = [
            'langchain',
            'langchain-google-genai',
            'faiss-cpu',
            'pypdf',
            'streamlit',
            'python-dotenv',
            'requests'
        ]
        
        for package in essential_packages:
            self.assertIn(package, requirements, f"Package {package} should be in requirements.txt")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDocumentProcessor,
        TestRBIChatbot,
        TestEvaluationDataset,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")