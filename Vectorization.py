                            #BAG OF WORDS EXAMPLE

# from sklearn.feature_extraction.text import CountVectorizer

# documents=[
#     "NLP Stands for Natural Language Processing",
#     "NLP is a subfield of Artificial Intelligence",
# ]

# vectorizer=CountVectorizer(stop_words='english')
# bow_matrix=vectorizer.fit_transform(documents)

# print("Vocabulary:\n", vectorizer.get_feature_names_out())
# print("Bag of Words Matrix:\n", bow_matrix.toarray())

                           #TF-IDF EXAMPLE

# from sklearn.feature_extraction.text import TfidfVectorizer

# documents=[
#     "I love NLP",
#     "I love machine learning",
# ]

# vectorizer=TfidfVectorizer(stop_words='english')
# tfidf_matrix=vectorizer.fit_transform(documents)

# print("Vocabulary:\n", vectorizer.get_feature_names_out())
# print("TF-IDF Matrix:\n", tfidf_matrix.toarray())


                            #WORD EMBEDDING EXAMPLE

# from gensim.models import Word2Vec

# # Sample corpus (already tokenized)
# sentences = [
#     ["i", "love", "nlp"],
#     ["i", "love", "machine", "learning"],
# ]

# # Train Word2Vec model
# model = Word2Vec(
#     sentences=sentences,
#     vector_size=50,
#     window=5,
#     min_count=1,
#     workers=4
# )

# # Get vector for a word
# vector = model.wv["nlp"]
# print("Vector for 'nlp':")
# print(vector)

# # Find similar words
# print("\nWords similar to 'nlp':")
# print(model.wv.most_similar("nlp"))


                              #COSINE SIMILARITY
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "I love natural language processing",
    "I enjoy learning NLP",
    "Machine learning is powerful"
]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)

print(similarity)
