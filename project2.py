# ----------------------------
# FAQ Chatbot using BoW + Cosine Similarity
# ----------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Prepare FAQ data
faqs = [
    {"question": "How do I reset my password?", "answer": "Click on 'Forgot Password' and follow the instructions."},
    {"question": "How can I update my profile?", "answer": "Go to your profile settings and edit your information."},
    {"question": "What is the refund policy?", "answer": "Refunds are processed within 7 days of your request."},
    {"question": "How do I contact customer support?", "answer": "You can contact support via email or the live chat on our website."},
    {"question": "How can I change my subscription plan?", "answer": "Go to your account settings and select 'Change Plan'."}
]

# Step 2: Extract questions
questions = [faq["question"].lower() for faq in faqs]  # lowercase for consistency

# Step 3: Convert questions to Bag of Words
vectorizer = CountVectorizer(stop_words='english')
faq_matrix = vectorizer.fit_transform(questions)

# Step 4: Get user query
user_query = input("Ask a question: ").lower()

# Step 5: Convert user query to vector
query_vector = vectorizer.transform([user_query])

# Step 6: Calculate cosine similarity
similarity_scores = cosine_similarity(query_vector, faq_matrix)[0]  # [0] to get array

# Step 7: Find most similar question
most_similar_index = similarity_scores.argmax()
best_score = similarity_scores[most_similar_index]

# Step 8: Return answer (set a threshold for low similarity)
threshold = 0.1  # adjust if needed
if best_score < threshold:
    print("Sorry, I don't know the answer to that.")
else:
    print("Answer:", faqs[most_similar_index]["answer"])




