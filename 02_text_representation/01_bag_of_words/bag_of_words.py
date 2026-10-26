from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love machine learning",
    "I love deep learning",
    "I love Python"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())
print(X.toarray())
