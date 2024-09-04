import torch
from typing import List
import collections
import numpy as np

Segment2D = collections.namedtuple(
    "Segment2D", ["camID", "segID", "coords"])

class Segment3D:
    def __init__(self, P1: np.ndarray = None, P2: np.ndarray = None):
        self.is_cuda = False
        if P1 is None or P2 is None:
            self.P1_ = np.array([0.0, 0.0, 0.0])
            self.P2_ = np.array([0.0, 0.0, 0.0])
            self.dir_ = np.array([0.0, 0.0, 0.0])
            self.length_ = 0.0
            self.valid_ = False
        else:
            # Check if P1 should be before P2
            if np.linalg.norm(P1) > np.linalg.norm(P2):
                P1, P2 = P2, P1
                
            self.length_ = np.linalg.norm(P1 - P2)
            if self.length_ > 1e-12:  
                self.P1_ = P1
                self.P2_ = P2
                self.dir_ = (P2 - P1) / self.length_
                self.valid_ = True
            else:
                self.P1_ = np.array([0.0, 0.0, 0.0])
                self.P2_ = np.array([0.0, 0.0, 0.0])
                self.dir_ = np.array([0.0, 0.0, 0.0])
                self.length_ = 0.0
                self.valid_ = False

    def distance_point_to_line(self, P: np.ndarray) -> float:
        hlp_pt = self.P1_ + self.dir_ * np.dot((P - self.P1_).T, self.dir_)
        return np.linalg.norm(hlp_pt - P)

    def translate(self, t: np.ndarray):
        self.P1_ += t
        self.P2_ += t

    def P1(self) -> np.ndarray:
        return self.P1_

    def P2(self) -> np.ndarray:
        return self.P2_

    def dir(self) -> np.ndarray:
        return self.dir_

    def length(self) -> float:
        return self.length_

    def valid(self) -> bool:
        return self.valid_

    def is_collinear_with(self, other, distance_threshold=1e-2):
        """Check if the current segment is collinear with another segment."""
        cross_dir = np.cross(self.dir_, other.dir_)
        if np.linalg.norm(cross_dir) < 1e-2:  # They are parallel
            P1_to_other = other.P1_ - self.P1_
            projected_distance = np.linalg.norm(np.cross(P1_to_other, self.dir_))
            if projected_distance < distance_threshold:
                return True
        return False
    
    def calculate_density(self, points, gap_threshold=1e-2):
        """Calculate the density of points around the segment within a certain gap threshold."""
        points_in_gap = self.points_in_gap(points, gap_threshold)
        density = len(points_in_gap) / self.length_ if self.length_ > 0 else 0
        return density, len(points_in_gap)
    
    def filter_points_within_segment(self, points, margin=1e-1):
        min_bound = np.minimum(self.P1_, self.P2_) - margin
        max_bound = np.maximum(self.P1_, self.P2_) + margin
        
        filtered_points = [p for p in points if np.all(p >= min_bound) and np.all(p <= max_bound)]
        return filtered_points
    
    def filter_points_within_gap(self, other, points, margin=1e-1):
        candidates = [self.P1_, self.P2_, other.P1_, other.P2_]
        sorted_candidates = sorted(candidates, key=lambda p: np.dot(p - self.P1_, self.dir_))
        min_bound = np.minimum(sorted_candidates[1], sorted_candidates[2]) - margin
        max_bound = np.maximum(sorted_candidates[1], sorted_candidates[2]) + margin
        
        filtered_points = [p for p in points if np.all(p >= min_bound) and np.all(p <= max_bound)]
        return filtered_points, sorted_candidates

    def points_in_gap(self, points, gap_threshold=1e-2):
        """Check if there are enough points within the gap between two segments."""
        gap_points = []
        for point in points:
            # Project the point onto the line defined by the current segment
            dist_to_line = self.distance_point_to_line(point)
            
            # Check if the point is within a certain threshold of the line
            if dist_to_line < gap_threshold:
                gap_points.append(point)
        
        return gap_points

    def try_merge(self, other, points, distance_threshold=1e-2, gap_threshold=1e-1):
        """Try to merge two segments if they are collinear and have enough points in the gap."""
    
        print("Start Step 1")
        # Step 1: Check if segments are collinear or have similar direction
        if not self.is_collinear_with(other, distance_threshold):
            return None
        print("Step 1 successfully")
    
        # Step 2: Identify points in the gap region
        filtered_points_self = self.filter_points_within_segment(points)
        density1, count1 = self.calculate_density(filtered_points_self, gap_threshold)
        print(f"filtered_points_self: {len(filtered_points_self)}, density1: {density1}, count1: {count1}")
        filtered_points_other = other.filter_points_within_segment(points)
        density2, count2 = other.calculate_density(filtered_points_other, gap_threshold)
        print(f"filtered_points_other: {len(filtered_points_other)}, density2: {density2}, count2: {count2}")
        
        filtered_gap_points, sorted_points = self.filter_points_within_gap(other, points)
        gap_points = self.points_in_gap(filtered_gap_points, gap_threshold)
        average_density = (density1 + density2) / 2
        expected_length = np.linalg.norm(sorted_points[1] - sorted_points[2])
        point_threshold = int(average_density * expected_length)
        print(f"Step 2: filtered_gap_points {filtered_gap_points}, number of gap_points {len(gap_points)}, point_threshold {point_threshold}")

        # Step 3: Merge segments if there are enough points in the gap
        if len(gap_points) >= point_threshold:
            new_P1, new_P2 = sorted_points[0], sorted_points[-1]
            return Segment3D(new_P1, new_P2)

        return None

class LineCluster3D:
    def __init__(self, seg3D: Segment3D = None, residuals: List = None, ref_view: int = 0):
        self.seg3D_ = seg3D if seg3D else Segment3D()
        self.residuals_ = residuals if residuals else []
        self.reference_view_ = ref_view

    def seg3D(self) -> Segment3D:
        return self.seg3D_

    def residuals(self) -> List:
        return self.residuals_

    def size(self) -> int:
        return len(self.residuals_)

    def reference_view(self) -> int:
        return self.reference_view_

    def update_3D_line(self, seg3D: Segment3D):
        self.seg3D_ = seg3D

    def translate(self, t: torch.Tensor):
        self.seg3D_.translate(t)


class FinalLine3D:
    def __init__(self):
        self.collinear3Dsegments_ = []
        self.underlyingCluster_ = LineCluster3D()

    def set_segments(self, segments : List):
        self.collinear3Dsegments_ = segments

    def set_cluster(self, cluster : LineCluster3D):
        self.underlyingCluster_ = cluster

    def collinear3Dsegments(self) -> List:
        return self.collinear3Dsegments_

    def underlyingCluster(self) -> LineCluster3D:
        return self.underlyingCluster_
