"""
Implement a function that calculates the unigram probability of a given word in a corpus of sentences. Include start <s> and end </s> tokens in the calculation. The probability should be rounded to 4 decimal places.

Example:
Input:
corpus = "<s> Jack I like </s> <s> Jack I do like </s>", word = "Jack"
Output:
0.1818
Reasoning:
The corpus has 11 total tokens. 'Jack' appears twice. So, probability = 2 / 11
"""


def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    arr = corpus.split()
    count = 0
    for item in arr:
        if item == word:
            count += 1
    return round(count / len(arr), 4)


if __name__ == "__main__":
    print(unigram_probability("<s> Jack I like </s> <s> Jack I do like </s>", "Jack"))
