"""
Task: Implement TF-IDF (Term Frequency-Inverse Document Frequency)
Your task is to implement a function that computes the TF-IDF scores for a query against a given corpus of documents.

Function Signature
Write a function compute_tf_idf(corpus, query) that takes the following inputs:

corpus: A list of documents, where each document is a list of words.
query: A list of words for which you want to compute the TF-IDF scores.
Output
The function should return a list of lists containing the TF-IDF scores for the query words in each document, rounded to five decimal places.

Important Considerations
Handling Division by Zero:
When implementing the Inverse Document Frequency (IDF) calculation, you must account for cases where a term does not appear in any document (df = 0). This can lead to division by zero in the standard IDF formula. Add smoothing (e.g., adding 1 to both numerator and denominator) to avoid such errors.

Empty Corpus:
Ensure your implementation gracefully handles the case of an empty corpus. If no documents are provided, your function should either raise an appropriate error or return an empty result. This will ensure the program remains robust and predictable.

Edge Cases:

Query terms not present in the corpus.
Documents with no words.
Extremely large or small values for term frequencies or document frequencies.
By addressing these considerations, your implementation will be robust and handle real-world scenarios effectively.

Example:
Input:
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "mat"]
]
query = ["cat"]

print(compute_tf_idf(corpus, query))
Output:
[[0.21461], [0.25754], [0.0]]
Reasoning:
The TF-IDF scores for the word "cat" in each document are computed and rounded to five decimal places.
"""


import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    tf_dict = dict()
    idf_dict = dict()
    for i in range(len(corpus)):
        doc = corpus[i]
        tf_dict[i] = dict()
        for word in doc:
            tf_dict[i][word] = tf_dict[i].get(word, 0) + 1
            if word not in idf_dict:
                idf_dict[word] = set()
            idf_dict[word].add(i)
    scores = []
    for i in range(len(corpus)):
        score = []
        doc = corpus[i]
        for q in query:
            tf = tf_dict[i].get(q, 0) / len(doc)
            idf = np.log((len(corpus) + 1) / (len(idf_dict.get(q, [])) + 1)) + 1
            score.append(round(tf * idf, 5))
        scores.append(score)
    return scores


if __name__ == "__main__":
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["the", "bird", "flew", "over", "the", "mat"]
    ]
    query = ["cat"]
    print(compute_tf_idf(corpus, query))
