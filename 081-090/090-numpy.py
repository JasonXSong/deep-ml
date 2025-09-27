"""
Implement the BM25 ranking function to calculate document scores for a query in an information retrieval context. BM25 is an advanced variation of TF-IDF that incorporates term frequency saturation, document length normalization, and a configurable penalty for document length effects.

Example:
Input:
corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']], query = ['the', 'cat']
Output:
[0.693, 0., 0. ]
Reasoning:
BM25 calculates scores for each document in the corpus by evaluating how well the query terms match each document while considering term frequency saturation and document length normalization.
"""


import numpy as np

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    # Your code here
    count_dq = dict()
    count_q = dict()
    amount = sum([len(doc) for doc in corpus])
    avgd = amount / len(corpus)
    for i in range(len(corpus)):
        count_dq[i] = dict()
    for q in query:
        for i in range(len(corpus)):
            doc = corpus[i]
            flag = False
            for word in doc:
                if q == word:
                    if not flag:
                        count_q[q] = count_q.get(q,0) + 1
                        flag = True
                    count_dq[i][q] = count_dq[i].get(q, 0) + 1

    scores = []
    for i in range(len(corpus)):
        score = 0
        for q in query:
            doc = corpus[i]
            tf = count_dq[i].get(q,0)# / len(corpus[i])
            #idf = np.log((len(corpus)-count_q.get(q, 0)+0.5) / (count_q.get(q, 0)+0.5) + 1)
            idf = np.log((len(corpus)+1) / (count_q.get(q, 0)+1))
            score += idf * (tf*(k1+1)) / (tf+k1*(1-b+b*(len(corpus[i])/avgd)))
        scores.append(score)

    return np.round(scores,3)


if __name__ == '__main__':
    corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']]
    query = ['the', 'cat']
    print(calculate_bm25_scores(corpus, query))