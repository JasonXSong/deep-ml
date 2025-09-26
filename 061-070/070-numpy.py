"""
Task: Image Brightness Calculator
In this task, you will implement a function calculate_brightness(img) that calculates the average brightness of a grayscale image. The image is represented as a 2D matrix, where each element represents a pixel value between 0 (black) and 255 (white).
Your Task:
Implement the function calculate_brightness(img) to:
Return the average brightness of the image rounded to two decimal places.
Handle edge cases:
If the image matrix is empty.
If the rows in the matrix have inconsistent lengths.
If any pixel values are outside the valid range (0-255).
For any of these edge cases, the function should return -1.
Example:
Input:
img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))
Output:
125.0
Reasoning:
The average brightness is calculated as (100 + 200 + 50 + 150) / 4 = 125.0
"""


import numpy as np


def calculate_brightness(img):
    # Write your code here
    if len(img) == 0:
        return -1
    columns = len(img[0])
    for row in img[1:]:
        if len(row) != columns:
            return -1
    for row in img:
        for pixel in row:
            if pixel < 0 or pixel > 255:
                return -1
    mat = np.array(img)
    return np.mean(mat)


if __name__ == "__main__":
    img = [
        [100, 200],
        [50, 150]
    ]
    print(calculate_brightness(img))
