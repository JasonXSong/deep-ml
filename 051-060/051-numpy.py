"""
In this problem, you need to implement a function that calculates the Optimal String Alignment (OSA) distance between two given strings. The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:
Insert a character
Delete a character
Substitute a character
Transpose two adjacent characters
Each of these operations costs 1 unit.
Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).
For example, the OSA distance between the strings caper and acer is 2: one deletion (removing "p") and one transposition (swapping "a" and "c").
Example:
Input:
source = "butterfly"
target = "dragonfly"

distance = OSA(source, target)
print(distance)
Output:
6
Reasoning:
The OSA distance between the strings "butterfly" and "dragonfly" is 6. The minimum number of edits required to transform the source string into the target string is 6.
"""


def OSA(source: str, target: str) -> int:
    # Your code here
    length_source = len(source)
    length_target = len(target)
    dp = [[0]*(length_target+1) for _ in range(length_source+1)]
    for i in range(length_source):
        dp[i+1][0] = i+1
    for j in range(length_target):
        dp[0][j+1] = j+1
    for i in range(length_source):
        for j in range(length_target):
            if source[i] == target[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                if i > 0 and j >0 and source[i] == target[j-1] and source[i-1] == target[j]:
                    dp[i+1][j+1] = min(dp[i-1][j-1], dp[i][j], dp[i][j+1], dp[i+1][j]) + 1
                else:
                    dp[i+1][j+1] = min(dp[i][j], dp[i][j+1], dp[i+1][j]) + 1
    return dp[length_source][length_target]


if __name__ == "__main__":
    source = "butterfly"
    target = "dragonfly"
    distance = OSA(source, target)
    print(distance)
