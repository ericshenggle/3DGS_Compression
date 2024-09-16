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
        sum_dist_to_line = 0.0
        for point in points:
            # Project the point onto the line defined by the current segment
            sum_dist_to_line = sum_dist_to_line + self.distance_point_to_line(point)
            
        gap_threshold = sum_dist_to_line / len(points) if len(points) > 0 else 0
        gap_threshold = gap_threshold * 0.5
            
        for point in points:
            # Check if the point is within a certain threshold of the line
            dist_to_line = self.distance_point_to_line(point)
            if dist_to_line < gap_threshold:
                gap_points.append(point)
        
        return gap_points 
    
    def calculate_density(self, points, gap_threshold=1e-2):
        """Calculate the density of points around the segment within a certain gap threshold."""
        filtered_points = self.filter_points_within_segment(points, 10 * gap_threshold)
        filtered_points = self.points_in_gap(filtered_points, gap_threshold)
        density = len(filtered_points) / self.length_ if self.length_ > 0 else 0
        return density, len(filtered_points)

    def try_collinear_merge(self, other, points, distance_threshold=1e-2, gap_threshold=1e-2):
        """Try to merge two segments if they are collinear and have enough points in the gap."""
    
        print("Start Step 1")
        # Step 1: Check if segments are collinear or have similar direction
        if not self.is_collinear_with(other, distance_threshold):
            return None
        print("Step 1 successfully")
    
        # Step 2: Identify points in the gap region
        density1, count1 = self.calculate_density(points, gap_threshold)
        print(f"density1: {density1}, count1: {count1}")
        density2, count2 = other.calculate_density(points, gap_threshold)
        print(f"density2: {density2}, count2: {count2}")
        
        filtered_gap_points, sorted_points = self.filter_points_within_gap(other, points)
        gap_points = self.points_in_gap(filtered_gap_points, gap_threshold)
        average_density = (density1 + density2) / 2
        expected_length = np.linalg.norm(sorted_points[1] - sorted_points[2])
        point_threshold = int(average_density * expected_length)
        print(f"Step 2: number of gap_points {len(gap_points)}, point_threshold {point_threshold}")

        # Step 3: Merge segments if there are enough points in the gap
        if len(gap_points) >= point_threshold:
            new_P1, new_P2 = sorted_points[0], sorted_points[-1]
            return Segment3D(new_P1, new_P2)

        return None
    
    def binary_search_crop(self, start, end, points, side='P1', margin_ratio=0.02, gap_threshold=1e-2, balance_threshold=0.6):
        """
        Binary search to adjust the segment's endpoint for cropping based on point distribution across a plane.
        
        Parameters:
        - start: The starting point of the segment (either P1 or P2).
        - end: The middle point or the opposite end to crop towards.
        - points: List of 3D points.
        - side: Indicates whether it's P1 or P2 being cropped.
        - margin_ratio: Proportion of the segment length to stop binary search.
        - balance_threshold: The proportion difference threshold to determine if we should stop or continue cropping.
        """
        segment_length = np.linalg.norm(start - end)
        tolerance_length = segment_length * margin_ratio
        
        idx = 0
        while True:
            idx = idx + 1
            if idx > 10:
                break
            
            # Step 1: Compute the midpoint and create a new sub-segment
            midpoint = (start + end) / 2
            sub_segment = Segment3D(midpoint - self.dir_ * tolerance_length / 2, midpoint + self.dir_ * tolerance_length / 2)
            
            filter_points = sub_segment.filter_points_within_segment(points, 10 * gap_threshold)
            
            # Step 2: Define the plane using the direction of the segment (normal vector of the plane)
            # The plane will be orthogonal to the direction vector and pass through the midpoint
            normal_vector = self.dir_  # This is the normal to the plane
            
            # Calculate distances of points from the plane
            def point_plane_distance(point):
                """Calculate the signed distance of a point from the plane defined by the segment direction."""
                point_to_midpoint = point - midpoint
                return np.dot(point_to_midpoint, normal_vector)  # Dot product gives the signed distance

            # Classify points based on which side of the plane they are on
            points_on_positive_side = []
            points_on_negative_side = []

            for point in filter_points:
                distance = point_plane_distance(point)
                if distance > 0:
                    points_on_positive_side.append(point)
                else:
                    points_on_negative_side.append(point)

            # Step 3: Evaluate the distribution of points across the plane
            positive_count = len(points_on_positive_side)
            negative_count = len(points_on_negative_side)

            # Calculate the proportion difference between two sides
            total_points = positive_count + negative_count
            if total_points == 0:
                break  # If there are no points, stop cropping

            proportion_diff = abs(positive_count - negative_count) / total_points
            print(f"positive_count: {positive_count}")
            print(f"negative_count: {negative_count}")
            print(f"proportion_diff: {proportion_diff}")

            # Step 5: If the proportion difference is large, continue cropping
            if proportion_diff > balance_threshold:
                # If the difference is large, move towards the side with fewer points
                if side == 'P1':
                    start = midpoint  # Move the start point closer to the middle
                else:
                    end = midpoint  # Move the end point closer to the middle
            elif proportion_diff < balance_threshold:
                # If the difference is small, move away to extend the segment
                if side == 'P1':
                    end = midpoint  # Move the start point closer to the middle
                else:
                    start = midpoint  # Move the end point closer to the middle
        return end if side == 'P1' else start
    
    def try_cropping(self, points, margin_ratio=0.02, gap_threshold=1e-2, density_threshold_ratio=0.2, balance_threshold=0.6):
        """
        Crop the segment using a binary search approach if the point density near the ends is much lower than in the middle.
        
        Parameters:
        - margin_ratio: The proportion of the segment near each end to evaluate density.
        - density_threshold_ratio: The ratio of density between the ends and the middle for cropping.
        - gap_threshold: Distance threshold for considering points near the segment.
        - tolerance: The threshold for stopping the binary search.
        """

        # Step 1: Define the margin for the ends
        segment_length = self.length()
        margin_length = segment_length * margin_ratio

        # Step 2: Calculate the density near the ends
        P1_end_region = self.P1_ + self.dir_ * margin_length
        P2_end_region = self.P2_ - self.dir_ * margin_length

        # Calculate densities
        p1_segment = Segment3D(self.P1_, P1_end_region)
        P1_density, _ = p1_segment.calculate_density(points, gap_threshold)
        print(f"P1_density: {P1_density}")
        p2_segment = Segment3D(P2_end_region, self.P2_)
        P2_density, _ = p2_segment.calculate_density(points, gap_threshold)
        print(f"P2_density: {P2_density}")

        # Step 3: Calculate the density in the middle region of the segment
        middle_segment = Segment3D(P1_end_region, P2_end_region)
        middle_density, _ = middle_segment.calculate_density(points, gap_threshold)
        print(f"middle_density: {middle_density}")

        # Crop P1 side if needed
        is_cropping = False
        mid_point = (self.P1_ + self.P2_) / 2
        if P1_density < middle_density * density_threshold_ratio:
            self.P1_ = self.binary_search_crop(self.P1_, mid_point, points, 'P1', margin_ratio, gap_threshold, balance_threshold)
            is_cropping = True

        # Crop P2 side if needed
        if P2_density < middle_density * density_threshold_ratio:
            self.P2_ = self.binary_search_crop(mid_point, self.P2_, points, 'P2', margin_ratio, gap_threshold, balance_threshold)
            is_cropping = True
            
        # Update self
        self.length_ = np.linalg.norm(self.P1_ - self.P2_)

        return is_cropping


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
