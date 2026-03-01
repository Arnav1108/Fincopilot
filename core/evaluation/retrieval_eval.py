def precision_at_k(retrieved_docs, expected_keywords, k=5):
    retrieved = retrieved_docs[:k]
    hits = 0

    for doc in retrieved:
        if any(keyword.lower() in doc["content"].lower() for keyword in expected_keywords):
            hits += 1

    return hits / k