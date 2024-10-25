from typing import List
import collections
import numpy as np
from sklearn.linear_model import LinearRegression

from arguments import SegmentParams
from lines.octree import *

Segment2D = collections.namedtuple(
    "Segment2D", ["camID", "segID", "coords"])


class Segment3D:
    def __init__(self, P1: np.ndarray = None, P2: np.ndarray = None):
        self.filter_points_idx_ = None
        self.filter_points_ = None
        self.density_ = None
        self.point_count_ = None
        self.is_cuda = False
        self.rmse_ = None
        self.residuals_ = None
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

    def project_point_to_line(self, P: np.ndarray) -> np.ndarray:
        return self.P1_ + np.dot((P - self.P1_).T, self.dir_) * self.dir_

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

    def rmse(self) -> float:
        return self.rmse_

    def filter_points_idx(self) -> List:
        return self.filter_points_idx_

    def density(self) -> float:
        return self.density_

    def point_count(self) -> int:
        return self.point_count_

    def filter_points_within_segment_or_gap(self, octree : Octree, margin, is_gap=False, other=None, optimize=False, recalculate=False):
        if is_gap:
            if other is None:
                raise ValueError("Other segment must be provided for gap filtering.")
            candidates = [self.P1_, self.P2_, other.P1_, other.P2_]
            sorted_candidates = sorted(candidates, key=lambda p: np.dot(p - self.P1_, self.dir_))
            origin_segment_direction = sorted_candidates[2] - sorted_candidates[1]
            sorted_candidates[1] = sorted_candidates[1] + origin_segment_direction * 0.02
            sorted_candidates[2] = sorted_candidates[2] - origin_segment_direction * 0.02
            target_segment = sorted_candidates[1:3]
        else:
            if self.filter_points_ is not None and not recalculate and not optimize:
                return self.filter_points_, [self.P1_, self.P2_]
            sorted_candidates = [self.P1_, self.P2_]
            target_segment = sorted_candidates
        # TODO: Also try to filter points based on the features that 3DGS points have
        target = Segment3D(target_segment[0], target_segment[1])
        filtered_points = octree.query(target, radius=margin)
        filtered_idx = target.filter_point(filtered_points, margin=margin)
        filtered_points = filtered_points[filtered_idx]
        # If optimize is True, only return the top 20% of points with the smallest distance
        if optimize:
            filtered_points_dist = [self.distance_point_to_line(p) for p in filtered_points]
            filtered_points_dist = np.array(filtered_points_dist)
            sorted_idx = np.argsort(filtered_points_dist)
            filtered_points = [filtered_points[i] for i in sorted_idx[:int(len(sorted_idx) * 0.4)]]
        elif not is_gap:
            self.filter_points_idx_ = octree.query_indices(target, radius=margin)
            self.filter_points_idx_ = self.filter_points_idx_[filtered_idx]
            self.filter_points_ = filtered_points
            self.point_count_ = len(filtered_points)
        return filtered_points, sorted_candidates

    def filter_point(self, points, margin) -> np.ndarray:
        filtered_points_idx = []
        for i, point in enumerate(points):
            # Project the point onto the line segment (cylinder axis)
            start_to_point = point - self.P1_
            projection_length_1 = np.dot(start_to_point, self.dir_)
            end_to_point = point - self.P2_
            projection_length_2 = np.dot(end_to_point, -self.dir_)
            # Check if the projection falls within the segment length
            if 0 <= projection_length_1 and 0 <= projection_length_2:
                # Calculate the perpendicular distance from the point to the cylinder axis
                projection_point = self.P1_ + projection_length_1 * self.dir_
                radial_distance = np.linalg.norm(point - projection_point)
                # Check if the point is within the cylinder's radius
                if radial_distance <= margin:
                    filtered_points_idx.append(i)
        filtered_points_idx = np.array(filtered_points_idx).astype(int)
        return filtered_points_idx

    def calculate_rmse(self, octree : Octree, margin, recalculate=False):
        """Calculate the root mean squared error (RMSE) of the segment to a set of 3D points."""
        if self.rmse_ is not None and not recalculate:
            return self.rmse_

        filtered_points, _ = self.filter_points_within_segment_or_gap(octree, margin)
        if len(filtered_points) == 0:
            self.rmse_ = np.inf
            return self.rmse_
        distances = [self.distance_point_to_line(p) for p in filtered_points]
        self.rmse_ = np.sqrt(np.mean(np.square(distances)))
        return self.rmse_

    def eval_line(self, octree : Octree, margin) -> (float, np.ndarray):
        """Calculate the root mean squared error (RMSE) of the segment to a set of 3D points."""
        filtered_points = octree.query(self, radius=margin)
        filtered_points_idx = octree.query_indices(self, radius=margin)
        filtered_idx = self.filter_point(filtered_points, margin=margin)
        filtered_points = filtered_points[filtered_idx]
        filtered_points_idx = filtered_points_idx[filtered_idx]
        if len(filtered_points) == 0:
            return np.inf, filtered_points_idx
        distances = [self.distance_point_to_line(p) for p in filtered_points]
        return np.sqrt(np.mean(np.square(distances))), filtered_points_idx

    def compute_gradient(self, octree : Octree, margin, epsilon):
        """Compute numerical gradient of the RMSE w.r.t. the segment's endpoints."""
        grad_p = np.zeros_like(self.P1_)

        initial_rmse = self.calculate_rmse(octree, margin=margin, recalculate=True)
        for i in range(len(self.P1_)):
            # Perturb P1
            self.P1_[i] += epsilon
            self.P2_[i] += epsilon
            rmse_p = self.calculate_rmse(octree, margin=margin, recalculate=True)
            grad_p[i] = (rmse_p - initial_rmse) / epsilon
            self.P1_[i] -= epsilon
            self.P2_[i] -= epsilon

        return grad_p

    def gradient_descent(self, octree : Octree, args : SegmentParams):
        for i in range(args.gd_max_iters):
            grad_p = self.compute_gradient(octree, args.margin, args.gd_epsilon)
            # print(f"Iteration {i}, RMSE: {self.rmse_}")
            self.P1_ -= args.gd_learning_rate * grad_p
            self.P2_ -= args.gd_learning_rate * grad_p
        return self.calculate_rmse(octree, margin=args.margin, recalculate=True)

    def lr_optimization(self, octree : Octree, margin):
        """Linear Regression optimization of the segment endpoints."""
        filtered_points, _ = self.filter_points_within_segment_or_gap(octree, margin=margin, optimize=True)
        # print(f"TLS Optimization: {len(filtered_points)} points")
        if len(filtered_points) != 0:

            # Project points onto the line segment's orthogonal plane
            points_centered = filtered_points - self.P1_
            projections = points_centered - np.dot(points_centered, self.dir_)[:, None] * self.dir_
            # Fit a linear regression model to the projected points
            lr = LinearRegression()
            lr.fit(np.ones((len(projections), 1)), projections)
            translation = lr.intercept_
            # print(f"Translation: {translation}")
            # print(f"Old P1: {self.P1_}, Old P2: {self.P2_}")
            self.P1_ += translation
            self.P2_ += translation
            # print(f"New P1: {self.P1_}, New P2: {self.P2_}")

        return self.calculate_rmse(octree, margin=margin, recalculate=True)

    def optimize_line(self, octree : Octree, args : SegmentParams):
        """Optimize the segment's endpoints to minimize the RMSE to a set of 3D points."""
        if not args.gd_enable:
            # Use TLS to optimize the segment endpoints
            return self.lr_optimization(octree, args.margin)
        else:
            # Use gradient descent to optimize the segment endpoints
            return self.gradient_descent(octree, args)

    def calculate_density(self, octree : Octree, margin, recalculate=False):
        """Calculate the density of points around the segment within a certain gap threshold."""
        if self.density_ is not None and not recalculate:
            return self.density_, self.point_count_
        _, _ = self.filter_points_within_segment_or_gap(octree, margin=margin, recalculate=True)
        density = self.point_count_ / self.length_ if self.length_ > 0 else 0
        self.density_ = density
        return density, self.point_count_

    def try_segments_merge(self, other, octree : Octree, args : SegmentParams, margin):
        """Try to merge two segments if they are collinear and have enough points in the gap."""

        # Step 1: Identify points in the gap region
        density1, count1 = self.calculate_density(octree, margin)
        # print(f"density1: {density1}, count1: {count1}")
        density2, count2 = other.calculate_density(octree, margin)
        # print(f"density2: {density2}, count2: {count2}")
        filtered_gap_points, sorted_points = self.filter_points_within_segment_or_gap(octree, margin, is_gap=True,
                                                                                      other=other)
        num_gap_points = len(filtered_gap_points)
        average_density = (density1 + density2) / 2
        expected_length = np.linalg.norm(sorted_points[1] - sorted_points[2])
        # Step 2: Merge segments if there are enough points in the gap
        # print(f"num_gap_points: {num_gap_points}, expected_length: {expected_length}, point_threshold: {point_threshold}")
        if num_gap_points < int(average_density * expected_length * args.try_merge_threshold):
            return None

        return Segment3D(sorted_points[0], sorted_points[-1])

    def binary_search_crop(self, start, end, octree : Octree, args : SegmentParams, p1_flag=True):
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
        margin_length = self.length() * args.binary_crop_margin / 2
        while True:
            idx = idx + 1
            # Step 1: Compute the midpoint and create a new sub-segment
            midpoint = (start + end) / 2
            if idx > 10 or np.linalg.norm(start - end) < 1e-4:
                return midpoint
            sub_segment = Segment3D(midpoint - self.dir_ * margin_length,
                                    midpoint + self.dir_ * margin_length)
            filter_points, _ = sub_segment.filter_points_within_segment_or_gap(octree, margin=args.margin)
            if len(filter_points) == 0:
                # If there are no points, continue cropping
                if p1_flag:
                    start = midpoint  # Move the start point closer to the middle
                else:
                    end = midpoint  # Move the end point closer to the middle
                continue
            normal_vector = self.dir()

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
            proportion_diff = abs(positive_count - negative_count) / total_points

            # Step 5: If the proportion difference is large, continue cropping
            if proportion_diff > args.binary_crop_threshold:
                # If the difference is large, move towards the side with fewer points
                if p1_flag:
                    start = midpoint  # Move the start point closer to the middle
                else:
                    end = midpoint  # Move the end point closer to the middle
            elif proportion_diff < args.binary_crop_threshold:
                # If the difference is small, move away to extend the segment
                if p1_flag:
                    end = midpoint  # Move the start point closer to the middle
                else:
                    start = midpoint  # Move the end point closer to the middle

    def try_cropping(self, octree : Octree, args : SegmentParams):
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
        if self.length() * args.cropping_endpoint_margin < 2 * args.margin:
            return False
        endpoint_margin_length = self.length() * args.cropping_endpoint_margin
        middle_margin_length = self.length() * args.cropping_mid_margin / 2
        # Step 2: Calculate the density near the ends
        p1_end_region = self.P1_ + self.dir_ * endpoint_margin_length
        p2_end_region = self.P2_ - self.dir_ * endpoint_margin_length
        middle = (self.P1_ + self.P2_) / 2
        middle_start = middle - self.dir_ * middle_margin_length
        middle_end = middle + self.dir_ * middle_margin_length
        # Calculate densities
        p1_segment = Segment3D(self.P1_, p1_end_region)
        p1_density, p1_count = p1_segment.calculate_density(octree, margin=args.margin)
        # print(f"P1_density: {p1_density}")
        p2_segment = Segment3D(p2_end_region, self.P2_)
        p2_density, p2_count = p2_segment.calculate_density(octree, margin=args.margin)
        # print(f"P2_density: {p2_density}")
        # Step 3: Calculate the density in the middle region of the segment
        middle_segment = Segment3D(middle_start, middle_end)
        middle_density, _ = middle_segment.calculate_density(octree, margin=args.margin)
        # print(f"middle_density: {middle_density}")
        # Crop P1 side if needed
        is_cropping = False
        mid_point = (self.P1_ + self.P2_) / 2
        if p1_density < middle_density * args.cropping_den_threshold:
            self.P1_ = self.binary_search_crop(self.P1_, mid_point, octree, args, True)
            is_cropping = True
        # Crop P2 side if needed
        if p2_density < middle_density * args.cropping_den_threshold:
            self.P2_ = self.binary_search_crop(mid_point, self.P2_, octree, args, False)
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


def segment_projection(seg1: Segment3D, seg2: Segment3D) -> bool:
    """Check if one segment can be projected onto another segment."""
    # Check if one endpoint of seg1 can be projected onto seg2 or vice versa
    short_seg = seg1 if seg1.length() < seg2.length() else seg2
    long_seg = seg1 if seg1.length() >= seg2.length() else seg2
    for p in [short_seg.P1(), short_seg.P2()]:
        long_seg_p1_to_p = p - long_seg.P1()
        long_seg_p2_to_p = p - long_seg.P2()
        if np.dot(long_seg_p1_to_p, long_seg.dir()) > 0 and np.dot(long_seg_p2_to_p, -long_seg.dir()) > 0:
            return True
    return False


def join_segments(seg1: Segment3D, seg2: Segment3D, octree : Octree, args : SegmentParams):
    """Join two segments into a single segment if possible."""
    # Check if one endpoint of seg1 which is closer to any endpoint of seg2 is close enough to be joined with seg2
    endpoints = [seg1.P1(), seg1.P2(), seg2.P1(), seg2.P2()]
    endpoints = sorted(endpoints, key=lambda x: np.linalg.norm(x - seg1.P1()))
    gap_segment = Segment3D(endpoints[1], endpoints[2])
    assume_joined_segment = Segment3D(endpoints[0], endpoints[-1])
    # if the angle between the assumed joined segment and the two original segments is not nearly 180 degrees or 0 degrees, return None
    if abs(np.dot(assume_joined_segment.dir(), seg1.dir())) < np.cos(np.radians(10)) or abs(np.dot(assume_joined_segment.dir(), seg2.dir())) < np.cos(np.radians(10)):
        # print("Segments are not nearly collinear.")
        return None
    gap_length = gap_segment.length()
    if gap_length > seg1.length() * args.join_length_threshold or gap_length > seg2.length() * args.join_length_threshold:
        # print("Gap is too large.")
        return None
    new_seg = seg1.try_segments_merge(seg2, octree, args, margin=args.margin)
    if new_seg is not None:
        new_seg.calculate_density(octree, margin=args.margin)
        new_seg.calculate_rmse(octree, margin=args.margin)
        if new_seg.density() < seg1.density() * args.join_den_threshold or new_seg.density() < seg2.density() * args.join_den_threshold:
            # print("Density of the joined segment is too low.")
            return None
        if new_seg.rmse() > seg1.rmse() * args.join_rmse_threshold or new_seg.rmse() > seg2.rmse() * args.join_rmse_threshold:
            # print("RMSE of the joined segment is too high.")
            return None
    return new_seg


def merge_segments(seg1: Segment3D, seg2: Segment3D, octree : Octree, args : SegmentParams):
    """Merge two segments into a single segment if possible."""
    short_seg = seg1 if seg1.length() < seg2.length() else seg2
    long_seg = seg1 if seg1.length() >= seg2.length() else seg2
    short_seg.calculate_rmse(octree, margin=args.margin)
    short_seg.calculate_density(octree, margin=args.margin)
    long_seg.calculate_rmse(octree, margin=args.margin)
    long_seg.calculate_density(octree, margin=args.margin)
    # Check if two segments are collinear or nearly collinear
    if abs(np.dot(short_seg.dir(), long_seg.dir())) < np.cos(np.radians(30)):
        # print("Segments are not nearly collinear.")
        return None
    # Check if this two segments are close enough
    if long_seg.distance_point_to_line(short_seg.P1()) > args.merge_dist_threshold * long_seg.length() and \
            long_seg.distance_point_to_line(short_seg.P2()) > args.merge_dist_threshold * long_seg.length():
        # print("Segments are not close enough.")
        return None

    # translate the long segment to the middle of the two segments
    proportion = short_seg.length() / (short_seg.length() + long_seg.length())
    translate_dir = short_seg.project_point_to_line(long_seg.P1()) - long_seg.P1()

    new_long_seg = Segment3D(long_seg.P1() + translate_dir * proportion, long_seg.P2() + translate_dir * proportion)
    new_long_seg.calculate_rmse(octree, margin=args.margin)
    new_long_seg.calculate_density(octree, margin=args.margin)
    if  new_long_seg.density() < long_seg.density() * args.merge_den_threshold:
        # print("Density of the merged segment is too low.")
        return None

    # return the segment if point_count is the highest among the five segments
    seg_list = [long_seg, new_long_seg]
    seg_list = sorted(seg_list, key=lambda x: x.calculate_rmse(octree, margin=args.margin) / x.point_count() if x.point_count() > 0 else np.inf)
    return seg_list[0]


def get_new_lines(new_segments: List[Segment3D], octree : Octree, args) -> FinalLine3D:
    """Get the new lines from the segments and points."""
    # Create new lines from the new_segments list
    # Two types of creation:
    # 1. Join two segments into a single line which has two endpoints chosen from the two segments,
    #    - the endpoints of two segments are not projected onto another segment, and
    #    - the density of points in the gap region is high enough, and
    #    - the density of the new line is high enough.
    # 2. Merge segments into a single line which is one of the segments or the average segment of the two segments,
    #    - if one segment's endpoint can be projected onto the other segment, and
    #    - the density of each segment is high enough, and
    #    - the average segment of the two segments has a high density.

    # Merge and join segments iteratively
    while len(new_segments) > 0:
        is_merged = False
        for i in range(len(new_segments)):
            for j in range(i + 1, len(new_segments)):
                if segment_projection(new_segments[i], new_segments[j]):
                    # Try to merge two segments
                    new_line = merge_segments(new_segments[i], new_segments[j], octree, args)
                    if new_line is not None:
                        new_segments.pop(j)
                        new_segments[i] = new_line
                        is_merged = True
                        break
            if is_merged:
                break
        if not is_merged:
            break
    while len(new_segments) > 0:
        is_joined = False
        for i in range(len(new_segments)):
            for j in range(i + 1, len(new_segments)):
                if not segment_projection(new_segments[i], new_segments[j]):
                    new_line = join_segments(new_segments[i], new_segments[j], octree, args)
                    if new_line is not None:
                        new_segments.pop(j)
                        new_segments[i] = new_line
                        is_joined = True
                        break
            if is_joined:
                break
        if not is_joined:
            break

    # Create new lines from the remaining segments
    new_line = FinalLine3D()
    new_line.set_segments(new_segments)

    return new_line
