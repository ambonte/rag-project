import json
from app.retriever import ask

# Ground truth Q&A pairs based on our document

EVAL_SET = [
     {
        "question": "How many days of sick leave do employees get?",
        "expected_keywords": ["10"],  
    },
    {
        "question": "What is the parental leave policy for primary caregivers?",
        "expected_keywords": ["16 weeks", "primary", "paid"],
    },
    {
        "question": "How much is the annual learning and development budget?",
        "expected_keywords": ["1,500", "learning", "development"],
    },
    {
        "question": "How many days notice must an employee give when resigning?",
        "expected_keywords": ["2 weeks", "notice", "written"],
    },
    {
        "question": "What percentage of health insurance premium does the company cover?",
        "expected_keywords": ["80%", "premium", "health"],
    },
    {
        "question": "How many days can employees carry over from annual leave?",
        "expected_keywords": ["5", "carry over", "annual"],
    },
    {
        "question": "What happens to sick leave at the end of the year?",
        "expected_keywords": ["resets", "January", "not", "carry"],
    },
    {
        "question": "How long does the 401k company match vesting take?",
        "expected_keywords": ["2-year", "vesting", "cliff"],
    },
]


def score_answer(answer: str, expected_keywords: list[str]) -> dict:
    """Check how many expected keywords appear in the answer.
     
    Simple keyword matching eval.
    Limitation: LLMs often paraphrase correctly without repeating
    the question's exact words. A production system would use an
    LLM-as-judge approach (e.g. ask GPT-4 'is this answer correct?')
    for more reliable scoring. See README for details.
    """
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missed = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    score = len(matched) / len(expected_keywords)
    return {"score": score, "matched": matched, "missed": missed}


def run_evals():
    """Run all eval questions and print a report."""
    print("Running RAG evaluation...\n")
    print("=" * 60)

    results = []
    total_score = 0

    for i, item in enumerate(EVAL_SET):
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        result = ask(question)
        answer = result["answer"]
        scoring = score_answer(answer, expected_keywords)

        total_score += scoring["score"]

        status = "PASS" if scoring["score"] >= 0.6 else "FAIL"

        print(f"\nQ{i+1}: {question}")
        print(f"Answer: {answer[:150]}...")
        print(f"Status: {status} | Score: {scoring['score']:.0%}")

        if scoring["missed"]:
            print(f"Missing keywords: {scoring['missed']}")

        results.append({
            "question": question,
            "answer": answer,
            "score": scoring["score"],
            "status": status,
            "matched": scoring["matched"],
            "missed": scoring["missed"],
        })

    avg_score = total_score / len(EVAL_SET)

    print("\n" + "=" * 60)
    print(f"OVERALL SCORE: {avg_score:.0%} ({sum(1 for r in results if r['status'] == 'PASS')}/{len(EVAL_SET)} passed)")
    print("=" * 60)

    with open("eval_results.json", "w") as f:
        json.dump({
            "overall_score": avg_score,
            "results": results
        }, f, indent=2)

    print("\nFull results saved to eval_results.json")
    return avg_score


if __name__ == "__main__":
    run_evals()