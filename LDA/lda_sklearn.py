import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset of text documents
documents = [
    "The cat is on the mat",
    "Dogs and cats are friends",
    "The dog chased the cat",
    "The cat climbed the tree",
    "Cats and dogs live in harmony",
]

# Convert text data into a TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
document_term_matrix = tfidf_vectorizer.fit_transform(documents)

# Set up LDA parameters
n_topics = 2
n_iter = 100
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=n_iter, random_state=42)

# Fit LDA model and collect log-likelihood over iterations for plotting
log_likelihoods = []

for i in range(1, n_iter + 1):
    lda.partial_fit(document_term_matrix)
    log_likelihoods.append(lda.score(document_term_matrix))

# Plotting the log-likelihood curve
plt.plot(range(1, n_iter + 1), log_likelihoods, label="Log-likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.title("LDA Loss Curve (Log-likelihood over iterations)")
plt.legend()
plt.show()


# Extract and print keywords for each topic
def print_top_words(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_features_indices = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_features)}")


# Displaying the top words for each topic
tf_feature_names = tfidf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names)
