import json
import requests
import time

API_BASE_URL = "http://localhost:8000/api"

def run_evaluation(eval_file="evaluation/sample_eval_set.json"):
    print(f"Loading evaluation set from {eval_file}...")
    with open(eval_file, "r") as f:
        eval_set = json.load(f)
        
    print(f"Found {len(eval_set)} questions to evaluate.\n")
    
    passed = 0
    
    for i, item in enumerate(eval_set):
        question = item["question"]
        expected_contains = item["expected_answer_contains"]
        should_be_found = item["should_be_found"]
        
        print(f"[{i+1}/{len(eval_set)}] Question: {question}")
        
        start_time = time.time()
        try:
            res = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": question, "prompt_type": "strict"}
            )
            
            if res.status_code == 200:
                answer = res.json().get("answer", "")
                
                # Check criteria
                if expected_contains.lower() in answer.lower():
                    print("✅ PASSED: Answer contains expected text.")
                    passed += 1
                else:
                    if not should_be_found and "not available" in answer.lower():
                        print("✅ PASSED: Guardrail triggered successfully.")
                        passed += 1
                    else:
                        print(f"❌ FAILED: Expected '{expected_contains}' but got:\n{answer}")
            else:
                print(f"❌ FAILED: API Error {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ FAILED: Connection error: {e}")
            
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s\n")
        
    print(f"--- Evaluation Complete ---")
    print(f"Score: {passed}/{len(eval_set)} ({(passed/len(eval_set))*100:.1f}%)")

if __name__ == "__main__":
    print("Ensure the FastAPI server is running on http://localhost:8000")
    run_evaluation()
