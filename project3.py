# ----------------------------
# TF-IDF FAQ Chatbot (Interactive)
# ----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Prepare FAQ data
faqs = [
    {"question": "How do I reset my password?", "answer": "Click on 'Forgot Password' and follow the instructions."},
    {"question": "How can I update my profile?", "answer": "Go to your profile settings and edit your information."},
    {"question": "What is the refund policy?", "answer": "Refunds are processed within 7 days of your request."},
    {"question": "How do I contact customer support?", "answer": "You can contact support via email or the live chat on our website."},
    {"question": "How can I change my subscription plan?", "answer": "Go to your account settings and select 'Change Plan'."}
]

# Step 2: Extract and preprocess questions
questions = [faq["question"].lower() for faq in faqs]

# Step 3: Convert questions to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
faq_matrix = vectorizer.fit_transform(questions)

print("Welcome to the FAQ Chatbot! Type 'exit' to quit.\n")

# Step 4: Chat loop
while True:
    user_query = input("Ask a question: ").lower()
    
    if user_query == "exit":
        print("Goodbye!")
        break

    # Step 5: Vectorize user query
    query_vector = vectorizer.transform([user_query])

    # Step 6: Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, faq_matrix)[0]

    # Step 7: Find best match
    most_similar_index = similarity_scores.argmax()
    best_score = similarity_scores[most_similar_index]

    # Step 8: Return answer if above threshold
    threshold = 0.1
    if best_score < threshold:
        print("Sorry, I don't know the answer to that.\n")
    else:
        print("Answer:", faqs[most_similar_index]["answer"], "\n")
