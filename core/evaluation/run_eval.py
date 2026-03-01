import json
from core.rag.rag import search_documents
from core.evaluation.retrieval_eval import precision_at_k

def run_retrieval_tests():
    with open("tests/financial_eval.json") as f:
        tests = json.load(f)

    results = []

    for test in tests:
        query = test["query"]
        expected = test["expected_keywords"]

        docs = search_documents.invoke({"query": query})

        score = precision_at_k(docs, expected, k=5)

        results.append({
            "query": query,
            "precision@5": score
        })

    return results

if __name__ == "__main__":
    results = run_retrieval_tests()

    for r in results:
        print(r)