import torch
from typing import List
import collections
import numpy as np

Segment2D = collections.namedtuple(
    "Segment2D", ["camID", "segID", "coords"])


class Segment3D:
    def __init__(self, P1: np.ndarray = None, P2: np.ndarray = None):
        self.density_ = None
        self.point_count_ = None
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

    def filter_points_within_segment_or_gap(self, points, margin=1e-1, is_gap=False, other=None):
        if is_gap:
            if other is None:
                raise ValueError("Other segment must be provided for gap filtering.")
            candidates = [self.P1_, self.P2_, other.P1_, other.P2_]
            sorted_candidates = sorted(candidates, key=lambda p: np.dot(p - self.P1_, self.dir_))
            target_segment = [sorted_candidates[1], sorted_candidates[2]]
        else:
            sorted_candidates = [self.P1_, self.P2_]
            target_segment = sorted_candidates
        # The main axis (segment direction)
        direction_vector = target_segment[1] - target_segment[0]
        direction_vector /= np.linalg.norm(direction_vector)
        # Step 2: Filter points inside the cylindrical region
        filtered_points = []
        for point in points:
            # Project the point onto the line segment (cylinder axis)
            start_to_point = point - target_segment[0]
            projection_length_1 = np.dot(start_to_point, direction_vector)
            end_to_point = point - target_segment[1]
            projection_length_2 = np.dot(end_to_point, -direction_vector)
            # Check if the projection falls within the segment length
            if 0 <= projection_length_1 and 0 <= projection_length_2:
                # Calculate the perpendicular distance from the point to the cylinder axis
                projection_point = target_segment[0] + projection_length_1 * direction_vector
                radial_distance = np.linalg.norm(point - projection_point)
                # Check if the point is within the cylinder's radius
                if radial_distance <= margin:
                    filtered_points.append(point)
        return filtered_points, sorted_candidates

    def calculate_density(self, points, margin=1e-1):
        """Calculate the density of points around the segment within a certain gap threshold."""
        if self.density_ is not None:
            return self.density_, self.point_count_
        filtered_points, _ = self.filter_points_within_segment_or_gap(points, margin)
        density = len(filtered_points) / self.length_ if self.length_ > 0 else 0
        self.density_ = density
        self.point_count_ = len(filtered_points)
        return density, len(filtered_points)

    def try_collinear_merge(self, other, points, margin=1e-1):
        """Try to merge two segments if they are collinear and have enough points in the gap."""
        # Step 1: Identify points in the gap region
        density1, count1 = self.calculate_density(points, margin)
        print(f"density1: {density1}, count1: {count1}")
        density2, count2 = other.calculate_density(points, margin)
        print(f"density2: {density2}, count2: {count2}")
        filtered_gap_points, sorted_points = self.filter_points_within_segment_or_gap(points, margin, is_gap=True,
                                                                                      other=other)
        average_density = (density1 + density2) / 2
        expected_length = np.linalg.norm(sorted_points[1] - sorted_points[2])
        point_threshold = int(average_density * expected_length) * 0.8
        print(f"number of gap_points {len(filtered_gap_points)}, point_threshold {point_threshold}")
        # Step 2: Merge segments if there are enough points in the gap
        if len(filtered_gap_points) >= point_threshold:
            new_p1, new_p2 = sorted_points[0], sorted_points[-1]
            return Segment3D(new_p1, new_p2)

        return None

    def binary_search_crop(self, start, end, points, p1_flag=True, margin_length=0.02, margin=1e-1,
                           balance_threshold=0.6):
        """
        Binary search to adjust the segment's endpoint for cropping based on point distribution across a plane.
        Parameters:
        - start: The starting point of the segment (either P1 or midpoint).
        - end: The ending point of the segment (either P2 or midpoint).
        - points: List of 3D points.
        - p1_flag: Flag to indicate if the start point is P1 or midpoint.
        - margin_length: The length of the segment to evaluate the density.
        - margin: The radius of the cylinder to filter points around the segment.
        - balance_threshold: The proportion difference threshold to determine if we should stop or continue cropping.
        """
        idx = 0
        p1 = start
        p2 = end
        while True:
            idx = idx + 1
            if idx > 10:
                break
            # Step 1: Compute the midpoint and create a new sub-segment
            midpoint = (start + end) / 2
            sub_segment = Segment3D(midpoint - self.dir_ * margin_length,
                                    midpoint + self.dir_ * margin_length)

            filter_points, _ = sub_segment.filter_points_within_segment_or_gap(points, margin)
            # Step 2: Define the plane using the direction of the segment (normal vector of the plane)
            # The plane will be orthogonal to the direction vector and pass through the midpoint
            normal_vector = self.dir_  # This is the normal to the plane

            # Calculate distances of points from the plane
            def point_plane_distance(p):
                """Calculate the signed distance of a point from the plane defined by the segment direction."""
                point_to_midpoint = p - midpoint
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
            if p1_flag:
                print(f"midpoint from P1: {np.linalg.norm(p1 - midpoint) / np.linalg.norm(p2 - p1) * 100}%")
            else:
                print(f"midpoint from P2: {np.linalg.norm(p2 - midpoint) / np.linalg.norm(p2 - p1) * 100}%")
            # Step 5: If the proportion difference is large, continue cropping
            if proportion_diff > balance_threshold:
                # If the difference is large, move towards the side with fewer points
                if p1_flag:
                    start = midpoint  # Move the start point closer to the middle
                else:
                    end = midpoint  # Move the end point closer to the middle
            elif proportion_diff < balance_threshold:
                # If the difference is small, move away to extend the segment
                if p1_flag:
                    end = midpoint  # Move the start point closer to the middle
                else:
                    start = midpoint  # Move the end point closer to the middle
        return end if p1_flag else start

    def try_cropping(self, points, segment_ratio=0.02, margin=1e-1, density_threshold_ratio=0.5, balance_threshold=0.6):
        """
        Crop the segment using a binary search approach if the point density near the ends is much lower than in the middle.
        Parameters:
        - segment_ratio: The proportion of the segment near each end to evaluate density.
        - margin: The radius of the cylinder to filter points around the segment.
        - density_threshold_ratio: The ratio of density between the ends and the middle for cropping.
        - gap_threshold: Distance threshold for considering points near the segment.
        - balance_threshold: The threshold for stopping the binary search.
        """
        # Step 1: Define the margin for the ends
        segment_length = self.length()
        margin_length = segment_length * segment_ratio
        # Step 2: Calculate the density near the ends
        p1_end_region = self.P1_ + self.dir_ * margin_length
        p2_end_region = self.P2_ - self.dir_ * margin_length
        middle = (self.P1_ + self.P2_) / 2
        middle_start = middle - self.dir_ * margin_length * 20
        middle_end = middle + self.dir_ * margin_length * 20
        # Calculate densities
        p1_segment = Segment3D(self.P1_, p1_end_region)
        p1_density, _ = p1_segment.calculate_density(points, margin)
        print(f"P1_density: {p1_density}")
        p2_segment = Segment3D(p2_end_region, self.P2_)
        p2_density, _ = p2_segment.calculate_density(points, margin)
        print(f"P2_density: {p2_density}")
        # Step 3: Calculate the density in the middle region of the segment
        middle_segment = Segment3D(middle_start, middle_end)
        middle_density, _ = middle_segment.calculate_density(points, margin)
        print(f"middle_density: {middle_density}")
        # Crop P1 side if needed
        is_cropping = False
        mid_point = (self.P1_ + self.P2_) / 2
        if p1_density < middle_density * density_threshold_ratio:
            self.P1_ = self.binary_search_crop(self.P1_, mid_point, points, True, margin_length, margin,
                                               balance_threshold)
            is_cropping = True
        # Crop P2 side if needed
        if p2_density < middle_density * density_threshold_ratio:
            self.P2_ = self.binary_search_crop(mid_point, self.P2_, points, False, margin_length, margin,
                                               balance_threshold)
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

    def translate(self, t: np.ndarray):
        self.seg3D_.translate(t)


class FinalLine3D:
    def __init__(self):
        self.collinear3Dsegments_ = []
        self.underlyingCluster_ = LineCluster3D()

    def set_segments(self, segments: List):
        self.collinear3Dsegments_ = segments

    def set_cluster(self, cluster: LineCluster3D):
        self.underlyingCluster_ = cluster

    def collinear3Dsegments(self) -> List:
        return self.collinear3Dsegments_

    def underlyingCluster(self) -> LineCluster3D:
        return self.underlyingCluster_
