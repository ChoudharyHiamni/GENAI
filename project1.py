# document_similarity.py
# Mini NLP Project: Document Similarity using Bag of Words & Cosine Similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Sample Documents
# -------------------------------
documents = [
    "I love machine learning and artificial intelligence",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a part of machine learning",
    "I enjoy watching movies and web series"
]

# -------------------------------
# Step 2: Convert Text → Bag of Words
# -------------------------------
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Bag of Words Matrix:\n", bow_matrix.toarray())

# -------------------------------
# Step 3: Calculate Cosine Similarity
# -------------------------------
similarity_matrix = cosine_similarity(bow_matrix)
print("\nCosine Similarity Matrix:\n", similarity_matrix)

# -------------------------------
# Step 4: Find Most Similar Documents
# -------------------------------
def find_most_similar_documents(similarity_matrix, documents):
    results = []
    for i in range(len(documents)):
        similarity_scores = list(enumerate(similarity_matrix[i]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        most_similar_index = similarity_scores[1][0]
        similarity_score = similarity_scores[1][1]

        results.append({
            "document": documents[i],
            "most_similar": documents[most_similar_index],
            "score": round(similarity_score, 2)
        })
    return results

# Run the function
results = find_most_similar_documents(similarity_matrix, documents)

# Display results
for result in results:
    print("\nDocument:")
    print(result["document"])
    print("Most Similar Document:")
    print(result["most_similar"])
    print("Similarity Score:", result["score"])
