"""
Test Query Evaluation Suite for Agentic AI Knowledge Assistant.
Executes the 6 mandatory test queries and validates answers and refusal handling.
"""

import sys
import time
from src.graph import run_rag_pipeline

TEST_QUERIES = [
    {
        "id": 1,
        "question": "What is Agentic AI according to the eBook?",
        "expected_type": "in_scope"
    },
    {
        "id": 2,
        "question": "How do AI agents differ from traditional automation systems?",
        "expected_type": "in_scope"
    },
    {
        "id": 3,
        "question": "What are the core components of an Agentic Architecture?",
        "expected_type": "in_scope"
    },
    {
        "id": 4,
        "question": "What role does memory play in Agentic AI workflows?",
        "expected_type": "in_scope"
    },
    {
        "id": 5,
        "question": "Who won the 2022 FIFA World Cup?",
        "expected_type": "out_of_scope"
    },
    {
        "id": 6,
        "question": "What is the capital of France and its population?",
        "expected_type": "out_of_scope"
    }
]

REFUSAL_TEXT = "I cannot answer based on the provided document."


def run_tests():
    """Execute evaluation test suite."""
    print("=" * 80)
    print("[TEST SUITE] AGENTIC AI KNOWLEDGE ASSISTANT - SAMPLE QUERIES EVALUATION")
    print("[INFO] Pausing 15s between queries to respect Gemini free-tier (5 RPM) limit.")
    print("=" * 80)
    
    passed_count = 0
    total_tests = len(TEST_QUERIES)
    
    for idx, item in enumerate(TEST_QUERIES):
        q_id = item["id"]
        question = item["question"]
        expected_type = item["expected_type"]
        
        print(f"\n--- [Test #{q_id}] Query: \"{question}\" ---")
        
        try:
            res = run_rag_pipeline(question)
            answer = res["final_answer"]
            score = res["confidence_score"]
            chunks_count = len(res["retrieved_context"])
            
            print(f"[SCORE] Confidence Score: {score}")
            print(f"[CHUNKS] Chunks Retrieved: {chunks_count}")
            print(f"[ANSWER] Answer:\n{answer}\n")
            
            if expected_type == "out_of_scope":
                if REFUSAL_TEXT.lower() in answer.lower():
                    print("[RESULT] PASSED (Correctly refused out-of-scope query)")
                    passed_count += 1
                else:
                    print("[RESULT] WARNING (Expected strict refusal response)")
            else:
                if answer.strip() != REFUSAL_TEXT:
                    print("[RESULT] PASSED (Answered grounded domain query)")
                    passed_count += 1
                else:
                    print("[RESULT] CHECK (Query was marked in-scope but document context might be missing)")
                    
        except Exception as e:
            print(f"[RESULT] FAILED with error: {e}")
        
        # Respect free-tier 5 RPM limit — wait 15s between queries
        if idx < total_tests - 1:
            print("[WAIT] Sleeping 15s before next query to respect API rate limit...")
            time.sleep(15)
            
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_tests} Tests Evaluated")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()


