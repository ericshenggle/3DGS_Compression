import torch
from typing import List
import collections
import numpy as np

Segment2D = collections.namedtuple(
    "Segment2D", ["camID", "segID", "coords"])

class Segment3D_Tensor:
    def __init__(self, P1: np.ndarray = None, P2: np.ndarray = None):
        self.is_cuda = True
        if P1 is None or P2 is None:
            self.P1_ = torch.tensor([0.0, 0.0, 0.0])
            self.P2_ = torch.tensor([0.0, 0.0, 0.0])
            self.dir_ = torch.tensor([0.0, 0.0, 0.0])
            self.length_ = 0.0
            self.valid_ = False
        else:
            if isinstance(P1, np.ndarray) and isinstance(P2, np.ndarray):
                P1_tensor = torch.tensor(P1, dtype=torch.float, device="cuda")
                P2_tensor = torch.tensor(P2, dtype=torch.float, device="cuda")
            elif isinstance(P1, torch.Tensor) and isinstance(P2, torch.Tensor):
                P1_tensor = P1
                P2_tensor = P2
            self.length_ = torch.norm(P1_tensor - P2_tensor)
            if self.length_ > 1e-12:
                self.P1_ = P1_tensor
                self.P2_ = P2_tensor
                self.dir_ = (P2_tensor - P1_tensor) / self.length_
                self.valid_ = True
            else:
                self.P1_ = torch.tensor([0.0, 0.0, 0.0])
                self.P2_ = torch.tensor([0.0, 0.0, 0.0])
                self.dir_ = torch.tensor([0.0, 0.0, 0.0])
                self.length_ = 0.0
                self.valid_ = False

    def distance_point_to_line(self, P: torch.Tensor) -> float:
        hlp_pt = self.P1_ + self.dir_ * torch.dot((P - self.P1_), self.dir_)
        return torch.norm(hlp_pt - P).item()

    def translate(self, t: torch.Tensor):
        self.P1_ += t
        self.P2_ += t

    def P1(self) -> torch.Tensor:
        return self.P1_

    def P2(self) -> torch.Tensor:
        return self.P2_

    def dir(self) -> torch.Tensor:
        return self.dir_

    def length(self) -> float:
        return self.length_

    def valid(self) -> bool:
        return self.valid_

    def is_collinear_with(self, other, angle_threshold=1e-3, distance_threshold=1e-2):
        """Check if the current segment is collinear with another segment."""
        angle_diff = torch.acos(torch.clamp(torch.dot(self.dir_, other.dir_), -1.0, 1.0))
        if angle_diff > angle_threshold:
            return False

        return True

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

    def merge_with(self, other):
        """Merge the current segment with another segment."""
        candidates = [self.P1_, self.P2_, other.P1_, other.P2_]
        sorted_candidates = sorted(candidates, key=lambda p: torch.dot(p - self.P1_, self.dir_))
        new_P1, new_P2 = sorted_candidates[0], sorted_candidates[-1]
        if torch.all(new_P1 > new_P2):
            new_P1, new_P2 = new_P2, new_P1
        
        return Segment3D_Tensor(new_P1, new_P2)

    def try_merge(self, other, points, angle_threshold=1e-3, distance_threshold=1e-2, point_threshold=5, gap_threshold=1e-2):
        """Try to merge two segments if they are collinear and have enough points in the gap."""
    
        # Step 1: Check if segments are collinear or have similar direction
        if not self.is_collinear_with(other, angle_threshold, distance_threshold):
            return None
    
        # Step 2: Identify points in the gap region
        gap_points = self.points_in_gap(points, gap_threshold)

        # Step 3: Merge segments if there are enough points in the gap
        if len(gap_points) >= point_threshold:
            return self.merge_with(other)

        return None

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

    def is_collinear_with(self, other, angle_threshold=1e-3, distance_threshold=1e-2):
        """Check if the current segment is collinear with another segment."""
        angle_diff = np.arccos(np.clip(np.dot(self.dir_, other.dir_), -1.0, 1.0))
        if angle_diff > angle_threshold:
            return False

        return True

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

    def merge_with(self, other):
        """Merge the current segment with another segment."""
        candidates = [self.P1_, self.P2_, other.P1_, other.P2_]
        sorted_candidates = sorted(candidates, key=lambda p: np.dot(p - self.P1_, self.dir_))
        new_P1, new_P2 = sorted_candidates[0], sorted_candidates[-1]
        if np.all(new_P1 > new_P2):
            new_P1, new_P2 = new_P2, new_P1
        
        return Segment3D(new_P1, new_P2)

    def try_merge(self, other, points, angle_threshold=1e-3, distance_threshold=1e-2, point_threshold=5, gap_threshold=1e-2):
        """Try to merge two segments if they are collinear and have enough points in the gap."""
    
        # Step 1: Check if segments are collinear or have similar direction
        if not self.is_collinear_with(other, angle_threshold, distance_threshold):
            return None
    
        # Step 2: Identify points in the gap region
        gap_points = self.points_in_gap(points, gap_threshold)

        # Step 3: Merge segments if there are enough points in the gap
        if len(gap_points) >= point_threshold:
            return self.merge_with(other)

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
