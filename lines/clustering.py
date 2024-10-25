from typing import List

import numpy as np

from arguments import SegmentParams
from lines import segment_projection
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
    if segment_projection(longer_seg, smaller_seg):
        return min(longer_seg.distance_point_to_line(smaller_seg.P1()), longer_seg.distance_point_to_line(smaller_seg.P2()))
    else:
        longer_seg_P1 = longer_seg.P1()
        longer_seg_P2 = longer_seg.P2()
        smaller_seg_P1 = smaller_seg.P1()
        smaller_seg_P2 = smaller_seg.P2()
        smaller_seg_P1_proj_to_longer_P1 = np.linalg.norm(longer_seg.project_point_to_line(smaller_seg_P1) - longer_seg_P1)
        smaller_seg_P1_proj_to_longer_P2 = np.linalg.norm(longer_seg.project_point_to_line(smaller_seg_P1) - longer_seg_P2)
        smaller_seg_P2_proj_to_longer_P1 = np.linalg.norm(longer_seg.project_point_to_line(smaller_seg_P2) - longer_seg_P1)
        smaller_seg_P2_proj_to_longer_P2 = np.linalg.norm(longer_seg.project_point_to_line(smaller_seg_P2) - longer_seg_P2)
        return min(smaller_seg_P1_proj_to_longer_P1, smaller_seg_P1_proj_to_longer_P2, smaller_seg_P2_proj_to_longer_P1, smaller_seg_P2_proj_to_longer_P2)


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
    dist_w = 1 / (1 + 2 * (dist / dist_threshold) ** 2)  # Quadratic decay based on distance
    # print(f"Distance : {dist} Distance weight : {dist_w}")

    # 2. Angle weight
    cos_theta = np.dot(seg1.dir(), seg2.dir())
    angle_w = abs(np.clip(cos_theta, -1.0,
                          1.0))  # cos(angle) gives weight close to 1 for small angles, close to 0 for large angles
    # print(f"Angle : {np.arccos(cos_theta) * 180 / np.pi} Angle weight : {angle_w}")

    # 3. Length ratio weight
    length_ratio = longer_seg.length() / smaller_seg.length()
    # length_w is based on the length ratio and angle. If the length ratio is large, the length_w should be larger.
    # if the angle is small, length ratio is less important.
    # if the angle is large, the larger length ratio is more important.
    length_w = np.tanh(length_ratio ** 2 * angle_w) if angle_w > 0.5 else 0
    # print(f"Length ratio : {length_ratio} Length weight : {length_w}")

    # Final weight is a combination of distance, angle, and length ratio
    weight = dist_w * length_w
    return weight


def perform_clustering(segments: List[Segment3D], index: List, args : SegmentParams):
    """
    Perform clustering of 3D line segments based on proximity, parallelism, and length ratio.

    Parameters:
    - segments: List of Segment3D objects.
    - index: the index of the line that segments belong to.
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

            weight = calculate_weight(segments[i], segments[j], args.cluster_dist_threshold)
            # print(f"Weight between {i} and {j}: {weight}")
            # If weight is sufficiently large, consider it a valid edge
            if weight >= args.cluster_weight_threshold:  # You can adjust this threshold as needed
                # print(f"Adding edge between {i} and {j}")
                edges.append(CLEdge(i, j, weight))

    # Step 2: Sort edges by weight
    edges.sort(key=lambda e: e.w_)

    # Step 3: Initialize CLUniverse
    universe = CLUniverse(num_nodes)

    # Step 4: Perform clustering based on edges
    c = args.cluster_c
    threshold = [c] * num_nodes
    for edge in edges:
        a = universe.find(edge.i_)
        b = universe.find(edge.j_)
        if a != b:
            if edge.w_ <= threshold[a] and edge.w_ <= threshold[b]:
                universe.join(a, b)
                root = universe.find(a)
                threshold[root] = edge.w_ + c / universe.size(root)

    clusters = {}
    for i, seg in enumerate(segments):
        root = universe.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # sort each cluster by calculated weight with respect to the root
    for root, cluster in clusters.items():
        if len(cluster) > 1:
            # sort the cluster by the calculated weight of each two segments within the cluster
            # calculate the weight of each two segments within the cluster
            weights = {}
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    weight = calculate_weight(segments[cluster[i]], segments[cluster[j]], args.cluster_dist_threshold)
                    weights[(i, j)] = weight
            # sort the weights
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            # sort the cluster based on the sorted weights
            new_cluster_idx = []
            for idx, _ in sorted_weights:
                i, j = idx
                # if i index and j index are not in the new cluster, add them to the new cluster
                if i not in new_cluster_idx and j not in new_cluster_idx:
                    new_cluster_idx.append(i)
                    new_cluster_idx.append(j)
            # add the remaining segments to the new cluster
            for i in range(len(cluster)):
                if i not in new_cluster_idx:
                    new_cluster_idx.append(i)
            # update the cluster
            clusters[root] = [cluster[i] for i in new_cluster_idx]

    return clusters
