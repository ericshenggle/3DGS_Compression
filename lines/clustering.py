from typing import List

import numpy as np
from lines.segment3D import Segment3D


class CLUniverse:
    def __init__(self, num_nodes):
        self.parent = list(range(num_nodes))  # Each node is initially its own parent
        self.size_ = [1] * num_nodes  # Initial size of each component is 1

    def find(self, node):
        """Find the representative of the set containing the node."""
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def join(self, a, b):
        """Union the sets containing a and b."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.size_[root_a] < self.size_[root_b]:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a
            self.size_[root_a] += self.size_[root_b]

    def size(self, node):
        return self.size_[self.find(node)]


class CLEdge:
    def __init__(self, i, j, w):
        self.i_ = i  # Line segment index 1
        self.j_ = j  # Line segment index 2
        self.w_ = w  # Weight (similarity)


def line_segment_distance(longer_seg, smaller_seg):
    """
    Calculate the minimum distance between two 3D line segments.
    """
    return min(longer_seg.distance_point_to_line(smaller_seg.P1()), longer_seg.distance_point_to_line(smaller_seg.P2()))


def calculate_weight(seg1, seg2, dist_threshold=0.1):
    """
    Calculate the similarity weight between two 3D line segments.

    Parameters:
    - seg1, seg2: Segment3D objects.
    - dist_threshold: The threshold for distance contribution.

    Returns:
    - weight: The calculated weight between two line segments.
    """
    longer_seg = seg1 if seg1.length() > seg2.length() else seg2
    smaller_seg = seg1 if seg1.length() <= seg2.length() else seg2
    # 1. Distance weight
    dist = line_segment_distance(longer_seg, smaller_seg)
    distance_threshold = dist_threshold * longer_seg.length()
    distance_threshold = distance_threshold if distance_threshold > 0.1 else 0.1
    dist_w = 1 / (1 + 2 * (dist / distance_threshold) ** 2)  # Quadratic decay based on distance
    print(f"Distance : {dist} Distance weight : {dist_w}")

    # 2. Angle weight
    cos_theta = np.dot(seg1.dir(), seg2.dir())
    angle_w = abs(np.clip(cos_theta, -1.0,
                          1.0))  # cos(angle) gives weight close to 1 for small angles, close to 0 for large angles
    print(f"Angle : {np.arccos(cos_theta) * 180 / np.pi} Angle weight : {angle_w}")

    # 3. Length ratio weight
    length_ratio = longer_seg.length() / smaller_seg.length()
    # length_w is based on the length ratio and angle. If the length ratio is large, the length_w should be larger.
    # if the angle is small, length ratio is less important.
    # if the angle is large, the larger length ratio is more important.
    length_w = np.tanh(length_ratio ** 2 * angle_w) if angle_w > 0.5 else 0
    print(f"Length ratio : {length_ratio} Length weight : {length_w}")

    # Final weight is a combination of distance, angle, and length ratio
    weight = dist_w * length_w
    return weight


def perform_clustering(segments: List[Segment3D], index: List, weight_threshold=0.5, dist_threshold=0.1, c=1):
    """
    Perform clustering of 3D line segments based on proximity, parallelism, and length ratio.

    Parameters:
    - segments: List of Segment3D objects.
    - index: the index of the line that segments belong to.
    - dist_threshold: Maximum allowed distance between two line segments.
    - c: Clustering constant to adjust merging behavior.

    Returns:
    - universe: The CLUniverse object representing the final clusters.
    """
    num_nodes = len(segments)
    edges = []

    # Step 1: Calculate similarity (edge weight) between each pair of segments
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if index[i] == index[j]:
                continue

            weight = calculate_weight(segments[i], segments[j], dist_threshold)
            print(f"Weight between {i} and {j}: {weight}")
            # If weight is sufficiently large, consider it a valid edge
            if weight >= weight_threshold:  # You can adjust this threshold as needed
                print(f"Adding edge between {i} and {j}")
                edges.append(CLEdge(i, j, weight))

    # Step 2: Sort edges by weight
    edges.sort(key=lambda e: e.w_)

    # Step 3: Initialize CLUniverse
    universe = CLUniverse(num_nodes)

    # Step 4: Perform clustering based on edges
    threshold = [c] * num_nodes
    for edge in edges:
        a = universe.find(edge.i_)
        b = universe.find(edge.j_)
        if a != b:
            if edge.w_ <= threshold[a] and edge.w_ <= threshold[b]:
                universe.join(a, b)
                root = universe.find(a)
                threshold[root] = edge.w_ + c / universe.size(root)

    return universe


def get_clusters(segments: List[Segment3D], universe: CLUniverse):
    """
    Get the final clusters of 3D line segments.

    Parameters:
    - segments: List of Segment3D objects.
    - universe: The CLUniverse object representing the final clusters.

    Returns:
    - clusters: A list of clusters, where each cluster contains a list of Segment3D objects.
    """
    clusters = {}
    for i, seg in enumerate(segments):
        root = universe.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # sort each cluster by calculated weight with respect to the root
    for root, cluster in clusters.items():
        if len(cluster) > 1:
            cluster.sort(key=lambda i: calculate_weight(segments[root], segments[i]))

    return clusters
