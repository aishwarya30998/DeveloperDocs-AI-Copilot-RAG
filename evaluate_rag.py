"""
Evaluate RAG pipeline using RAGAS framework.

Measures:
- Faithfulness: Answer accuracy vs. retrieved context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: How precise the retrieved context is
- Context Recall: Coverage of relevant information
"""
import logging
import sys
from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_rag_pipeline, settings
from src.config import EVALS_DIR, RESULTS_DIR

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("WARNING: RAGAS not installed. Install with: pip install ragas")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Evaluation dataset
TEST_QUERIES = [
    {
        "question": "How do I create a FastAPI application?",
        "ground_truth": "You create a FastAPI application by importing FastAPI and creating an instance: from fastapi import FastAPI; app = FastAPI()"
    },
    {
        "question": "What are path parameters in FastAPI?",
        "ground_truth": "Path parameters are variables in the URL path that FastAPI can extract and pass to your endpoint function."
    },
    {
        "question": "How do I add request validation?",
        "ground_truth": "FastAPI uses Pydantic models for request validation. You define a model with type hints and use it as a parameter type."
    },
    {
        "question": "What is dependency injection in FastAPI?",
        "ground_truth": "Dependency injection allows you to declare dependencies that FastAPI will resolve and inject into your endpoint functions."
    },
    {
        "question": "How do I handle authentication in FastAPI?",
        "ground_truth": "FastAPI provides security utilities for OAuth2, JWT tokens, and API keys. You can use dependencies to protect endpoints."
    },
]


def run_evaluation():
    """Run RAGAS evaluation on the RAG pipeline."""
    
    if not RAGAS_AVAILABLE:
        logger.error("RAGAS not available. Please install it.")
        return
    
    logger.info("=" * 60)
    logger.info("RAG Evaluation with RAGAS")
    logger.info("=" * 60)
    
    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = create_rag_pipeline()
    
    # Prepare evaluation data
    logger.info(f"\nRunning evaluation on {len(TEST_QUERIES)} queries...")
    
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for item in TEST_QUERIES:
        question = item["question"]
        logger.info(f"\nProcessing: {question}")
        
        # Get response from pipeline
        response = pipeline.query(question, top_k=5)
        
        # Extract data for RAGAS
        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(response["answer"])
        evaluation_data["ground_truth"].append(item["ground_truth"])
        
        # Get context from retrieved chunks
        contexts = []
        retrieved_chunks = pipeline.retriever.retrieve(question, top_k=5)
        for chunk in retrieved_chunks:
            contexts.append(chunk["content"])
        evaluation_data["contexts"].append(contexts)
        
        logger.info(f"  Answer length: {len(response['answer'])} chars")
        logger.info(f"  Contexts retrieved: {len(contexts)}")
    
    # Create dataset
    dataset = Dataset.from_dict(evaluation_data)
    
    # Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Running RAGAS metrics...")
    logger.info("=" * 60)
    
    try:
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        
        metrics = {
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
        }
        
        for metric_name, score in metrics.items():
            logger.info(f"{metric_name.replace('_', ' ').title()}: {score:.4f}")
        
        # Overall score
        overall_score = sum(metrics.values()) / len(metrics)
        logger.info(f"\nOverall Score: {overall_score:.4f}")
        
        # Interpretation
        logger.info("\n" + "=" * 60)
        logger.info("Interpretation")
        logger.info("=" * 60)
        logger.info("Scores range from 0 to 1 (higher is better)")
        logger.info("Target scores for production:")
        logger.info("  • Faithfulness: > 0.80 (answers are accurate)")
        logger.info("  • Answer Relevancy: > 0.70 (answers address the question)")
        logger.info("  • Context Precision: > 0.75 (retrieved context is relevant)")
        logger.info("  • Context Recall: > 0.80 (all relevant info is retrieved)")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"ragas_eval_{timestamp}.json"
        
        results_dict = {
            "timestamp": timestamp,
            "metrics": metrics,
            "overall_score": overall_score,
            "test_queries": TEST_QUERIES,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": 5,
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)


def simple_accuracy_test():
    """Simple accuracy test without RAGAS."""
    logger.info("Running simple accuracy test...")
    
    pipeline = create_rag_pipeline()
    
    correct = 0
    total = len(TEST_QUERIES)
    
    for item in TEST_QUERIES:
        question = item["question"]
        response = pipeline.query(question)
        
        # Simple check: does answer contain key terms?
        answer_lower = response["answer"].lower()
        ground_truth_lower = item["ground_truth"].lower()
        
        # Extract key terms from ground truth
        key_terms = [term for term in ground_truth_lower.split() if len(term) > 4]
        
        # Check if at least 50% of key terms are in answer
        matches = sum(1 for term in key_terms if term in answer_lower)
        if matches / len(key_terms) >= 0.5:
            correct += 1
            logger.info(f"✓ {question}")
        else:
            logger.info(f"✗ {question}")
    
    accuracy = correct / total
    logger.info(f"\nSimple Accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    if RAGAS_AVAILABLE:
        run_evaluation()
    else:
        logger.warning("RAGAS not available. Running simple test instead.")
        simple_accuracy_test()
