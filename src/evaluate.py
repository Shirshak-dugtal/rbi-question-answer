"""
RBI Chatbot Evaluation Module

This module implements comprehensive evaluation of the RBI chatbot using LangSmith.
It includes dataset creation, evaluation metrics, and result analysis.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from langchain.evaluation import load_evaluator
from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.evaluation import evaluate
from chatbot import RBIChatbot, create_chatbot_from_pdf


class RBIChatbotEvaluator:
    """
    Comprehensive evaluation framework for the RBI chatbot.
    """
    
    def __init__(self, chatbot: RBIChatbot, langsmith_client: Client = None):
        """
        Initialize the evaluator.
        
        Args:
            chatbot: The RBI chatbot instance to evaluate
            langsmith_client: LangSmith client for tracking evaluations
        """
        self.chatbot = chatbot
        self.client = langsmith_client or Client()
        
        # Initialize evaluators
        self.qa_evaluator = load_evaluator("qa")
        self.criteria_evaluator = load_evaluator("criteria", criteria="helpfulness")
        self.embedding_evaluator = load_evaluator("embedding_distance")
    
    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """
        Create evaluation dataset from RBI FAQ data.
        
        Returns:
            List of evaluation examples with questions and expected answers
        """
        # RBI FAQ-based evaluation dataset
        evaluation_data = [
            {
                "question": "What is NBFC?",
                "expected_answer": "A Non-Banking Financial Company (NBFC) is a company registered under the Companies Act, 1956/2013 engaged in the business of loans and advances, acquisition of shares/stocks/bonds/debentures/securities issued by Government or local authority or other marketable securities of a like nature, leasing, hire-purchase, insurance business, chit business but does not include any institution whose principal business is that of agriculture activity, industrial activity, purchase or sale of any goods (other than securities) or providing any services and sale/purchase/construction of immovable property.",
                "category": "definition"
            },
            {
                "question": "Who regulates NBFCs?",
                "expected_answer": "NBFCs are regulated by the Reserve Bank of India (RBI) under the RBI Act, 1934 and various directions issued by RBI from time to time.",
                "category": "regulation"
            },
            {
                "question": "What is the minimum capital requirement for NBFC registration?",
                "expected_answer": "The minimum Net Owned Fund (NOF) requirement for NBFCs is Rs. 2 crore. However, for specific categories like NBFC-MFI, the requirement may be different.",
                "category": "requirements"
            },
            {
                "question": "What are the different types of NBFCs?",
                "expected_answer": "There are various types of NBFCs including Asset Finance Company (AFC), Investment Company (IC), Loan Company (LC), Infrastructure Finance Company (IFC), Systemically Important Non-Deposit taking NBFC (NBFC-ND-SI), Deposit taking NBFC (NBFC-D), Micro Finance Institution (NBFC-MFI), and others.",
                "category": "types"
            },
            {
                "question": "Can NBFCs accept deposits from public?",
                "expected_answer": "Only NBFCs holding Certificate of Registration with authorization to accept deposits can accept/renew public deposits. However, no NBFC shall accept any public deposit unless it has minimum investment grade credit rating for fixed deposits programme from a credit rating agency registered with SEBI.",
                "category": "deposits"
            },
            {
                "question": "What is the role of RBI in NBFC supervision?",
                "expected_answer": "RBI supervises NBFCs through various mechanisms including registration, ongoing supervision, compliance monitoring, inspection, and regulatory action when necessary. RBI issues directions and guidelines for NBFCs operation and ensures compliance with prudential norms.",
                "category": "supervision"
            },
            {
                "question": "What are the prudential norms for NBFCs?",
                "expected_answer": "Prudential norms for NBFCs include capital adequacy requirements, asset classification and provisioning norms, exposure norms, liquidity requirements, and corporate governance standards as prescribed by RBI.",
                "category": "prudential_norms"
            },
            {
                "question": "How to apply for NBFC registration?",
                "expected_answer": "To apply for NBFC registration, an entity must submit application to RBI along with required documents including certificate of incorporation, memorandum and articles of association, business plan, details of directors and substantial shareholders, and other specified documents as per RBI guidelines.",
                "category": "registration_process"
            },
            {
                "question": "What is systemically important NBFC?",
                "expected_answer": "A systemically important NBFC (NBFC-SI) is an NBFC having asset size of Rs. 500 crore or above as shown in the last audited balance sheet. These NBFCs are subject to enhanced regulatory oversight due to their potential systemic impact.",
                "category": "systemic_importance"
            },
            {
                "question": "What are the reporting requirements for NBFCs?",
                "expected_answer": "NBFCs are required to submit various returns to RBI including monthly, quarterly, and annual returns covering their financial position, operations, compliance status, and other regulatory requirements as specified by RBI.",
                "category": "reporting"
            }
        ]
        
        return evaluation_data
    
    def evaluate_single_question(self, question: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The question to evaluate
            expected_answer: The expected answer
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Get chatbot response
            result = self.chatbot.ask_question(question)
            actual_answer = result["answer"]
            
            # Evaluate using different metrics
            evaluations = {}
            
            # QA Evaluator (measures correctness)
            try:
                qa_result = self.qa_evaluator.evaluate_strings(
                    prediction=actual_answer,
                    reference=expected_answer,
                    input=question
                )
                evaluations["qa_score"] = qa_result.get("score", 0)
            except Exception as e:
                print(f"QA evaluation error: {e}")
                evaluations["qa_score"] = 0
            
            # Criteria Evaluator (measures helpfulness)
            try:
                criteria_result = self.criteria_evaluator.evaluate_strings(
                    prediction=actual_answer,
                    input=question
                )
                evaluations["helpfulness_score"] = criteria_result.get("score", 0)
            except Exception as e:
                print(f"Criteria evaluation error: {e}")
                evaluations["helpfulness_score"] = 0
            
            # Custom metrics
            evaluations.update({
                "response_length": len(actual_answer),
                "has_sources": len(result.get("sources", [])) > 0,
                "confidence": result.get("confidence", "unknown"),
                "num_sources": len(result.get("sources", [])),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "sources": result.get("sources", [])
            })
            
            return evaluations
            
        except Exception as e:
            print(f"Error evaluating question '{question}': {str(e)}")
            return {
                "qa_score": 0,
                "helpfulness_score": 0,
                "response_length": 0,
                "has_sources": False,
                "confidence": "error",
                "num_sources": 0,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": f"Error: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the chatbot on a complete dataset.
        
        Args:
            dataset: List of evaluation examples (optional, uses default if None)
            
        Returns:
            Comprehensive evaluation results
        """
        if dataset is None:
            dataset = self.create_evaluation_dataset()
        
        print(f"Evaluating chatbot on {len(dataset)} questions...")
        
        # Evaluate each question
        results = []
        for i, item in enumerate(dataset):
            print(f"Evaluating question {i+1}/{len(dataset)}: {item['question'][:50]}...")
            
            evaluation = self.evaluate_single_question(
                item["question"], 
                item["expected_answer"]
            )
            evaluation["category"] = item.get("category", "unknown")
            results.append(evaluation)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Create comprehensive results
        evaluation_results = {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "dataset_size": len(dataset),
            "evaluation_summary": self._create_evaluation_summary(results, aggregate_metrics)
        }
        
        return evaluation_results
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Basic statistics
        metrics = {
            "avg_qa_score": df["qa_score"].mean(),
            "avg_helpfulness_score": df["helpfulness_score"].mean(),
            "avg_response_length": df["response_length"].mean(),
            "avg_num_sources": df["num_sources"].mean(),
            "questions_with_sources": df["has_sources"].sum(),
            "questions_without_sources": (~df["has_sources"]).sum(),
            "source_coverage_rate": df["has_sources"].mean(),
            "high_confidence_responses": (df["confidence"] == "high").sum(),
            "medium_confidence_responses": (df["confidence"] == "medium").sum(),
            "low_confidence_responses": (df["confidence"] == "low").sum(),
            "error_count": df["error"].notna().sum() if "error" in df.columns else 0
        }
        
        # Category-wise analysis
        if "category" in df.columns:
            category_stats = df.groupby("category").agg({
                "qa_score": "mean",
                "helpfulness_score": "mean",
                "has_sources": "mean",
                "num_sources": "mean"
            }).to_dict("index")
            metrics["category_performance"] = category_stats
        
        return metrics
    
    def _create_evaluation_summary(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
        """Create a human-readable evaluation summary."""
        summary_lines = [
            "RBI Chatbot Evaluation Summary",
            "=" * 40,
            f"Total Questions Evaluated: {len(results)}",
            f"Average QA Score: {metrics.get('avg_qa_score', 0):.2f}",
            f"Average Helpfulness Score: {metrics.get('avg_helpfulness_score', 0):.2f}",
            f"Source Coverage Rate: {metrics.get('source_coverage_rate', 0):.2%}",
            f"Average Response Length: {metrics.get('avg_response_length', 0):.0f} characters",
            f"Average Sources per Response: {metrics.get('avg_num_sources', 0):.1f}",
            "",
            "Confidence Distribution:",
            f"  High: {metrics.get('high_confidence_responses', 0)}",
            f"  Medium: {metrics.get('medium_confidence_responses', 0)}",
            f"  Low: {metrics.get('low_confidence_responses', 0)}",
            "",
            f"Questions with Sources: {metrics.get('questions_with_sources', 0)}",
            f"Questions without Sources: {metrics.get('questions_without_sources', 0)}",
            f"Errors: {metrics.get('error_count', 0)}"
        ]
        
        return "\n".join(summary_lines)
    
    def save_evaluation_results(self, results: Dict[str, Any], file_path: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            file_path: Path to save the results
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Evaluation results saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving evaluation results: {str(e)}")
    
    def generate_evaluation_report(self, results: Dict[str, Any], report_path: str):
        """
        Generate a detailed evaluation report.
        
        Args:
            results: Evaluation results dictionary
            report_path: Path to save the report
        """
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # Write summary
                f.write(results["evaluation_summary"])
                f.write("\n\n")
                
                # Write detailed results
                f.write("Detailed Results:\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(results["individual_results"], 1):
                    f.write(f"Question {i}: {result['question']}\n")
                    f.write(f"Expected: {result['expected_answer'][:100]}...\n")
                    f.write(f"Actual: {result['actual_answer'][:100]}...\n")
                    f.write(f"QA Score: {result['qa_score']:.2f}\n")
                    f.write(f"Helpfulness Score: {result['helpfulness_score']:.2f}\n")
                    f.write(f"Sources: {result['num_sources']}\n")
                    f.write(f"Confidence: {result['confidence']}\n")
                    f.write(f"Category: {result.get('category', 'N/A')}\n")
                    f.write("-" * 30 + "\n\n")
            
            print(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            print(f"Error generating evaluation report: {str(e)}")


def run_comprehensive_evaluation():
    """Run a comprehensive evaluation of the RBI chatbot."""
    # Configuration
    rbi_url = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF"
    pdf_path = "data/rbi_notification.pdf"
    vector_store_path = "data/rbi_faiss_index"
    
    try:
        print("Setting up RBI Chatbot for evaluation...")
        
        # Create chatbot
        chatbot = create_chatbot_from_pdf(rbi_url, pdf_path, vector_store_path)
        
        # Create evaluator
        evaluator = RBIChatbotEvaluator(chatbot)
        
        # Run evaluation
        print("Running comprehensive evaluation...")
        results = evaluator.evaluate_dataset()
        
        # Save results
        results_file = "data/evaluation_results.json"
        report_file = "data/evaluation_report.txt"
        
        evaluator.save_evaluation_results(results, results_file)
        evaluator.generate_evaluation_report(results, report_file)
        
        # Print summary
        print("\n" + results["evaluation_summary"])
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the evaluation
    results = run_comprehensive_evaluation()