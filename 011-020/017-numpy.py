"""
Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition n points into k clusters. The goal is to group similar points together and represent each group by its center (called the centroid).
Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.
Example:
Input:
points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
Output:
[(1, 2), (10, 2)]
Reasoning:
Given the initial centroids and a maximum of 10 iterations, the points are clustered around these points, and the centroids are updated to the mean of the assigned points, resulting in the final centroids which approximate the means of the two clusters. The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.
"""


def get_distance(point1, point2):
	return sum([(point1[i] - point2[i])**2 for i in range(len(point1))]) ** 0.5


def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	# Your code here
	for _ in range(max_iterations):
		clusters = [[] for _ in range(k)]
		for point in points:
			min_index = 0
			min_distance = get_distance(point, initial_centroids[0])
			for i in range(1, k):
				distance = get_distance(point, initial_centroids[i])
				if distance < min_distance:
					min_index = i
					min_distance = distance
			clusters[min_index].append(point)
		cur_centroids = []
		for kk in range(k):
			centroid = []
			for i in range(len(points[0])):
				val = round(sum(p[i] for p in clusters[kk]) / len(clusters[kk]), 4)
				centroid.append(val)
			cur_centroids.append(centroid)
		initial_centroids = cur_centroids
	return initial_centroids


if __name__ == '__main__':
	points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
	k = 2
	initial_centroids = [(1, 1), (10, 1)]
	max_iterations = 10
	centroids = k_means_clustering(points, k, initial_centroids, max_iterations)
	print(centroids)
	print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))

