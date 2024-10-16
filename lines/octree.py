import numpy as np
from scene.dataset_readers import storePly


class OctreeNode:
    def __init__(self, bounds):
        self.bounds = bounds  # bounding of the node
        self.points = []  # points in the node
        self.points_idx = []  # indices of the points in the node
        self.children = [None] * 8  # children nodes
        self.num_points_ = None  # number of points in the node

    def is_leaf(self):
        """Check if the node is a leaf node"""
        return all(child is None for child in self.children)

    def num_points(self):
        """Get the number of points in the node"""
        if self.num_points_ is None:
            self.num_points_ = len(self.points_idx)
            if not self.is_leaf():
                for child in self.children:
                    self.num_points_ += child.num_points()
        return self.num_points_


class Octree:
    def __init__(self, bounds, max_depth=10, max_points=10):
        """Initialize the Octree"""
        self.root = OctreeNode(bounds)
        self.max_depth = max_depth
        self.max_points = max_points

    def get_num_points(self):
        """Get the number of points in the Octree"""
        return self.root.num_points()

    def insert(self, point, point_idx, node=None, depth=0):
        """Insert a point into the Octree"""
        if node is None:
            node = self.root

        # if the depth is greater than the max depth or the node has fewer points than the threshold, add the point to the node
        if depth >= self.max_depth or (node.is_leaf() and len(node.points) < self.max_points):
            node.points.append(point)
            node.points_idx.append(point_idx)
        else:
            if node.is_leaf():
                self.split(node)

            # recursively insert the point into the appropriate child node
            idx = self.get_child_index(node.bounds, point)
            self.insert(point, point_idx, node.children[idx], depth + 1)

    def split(self, node):
        """Split the node into eight children"""
        x_min, y_min, z_min = node.bounds[0]
        x_max, y_max, z_max = node.bounds[1]
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        # define the bounds for the eight children
        child_bounds = [
            ((x_min, y_min, z_min), (x_mid, y_mid, z_mid)),
            ((x_mid, y_min, z_min), (x_max, y_mid, z_mid)),
            ((x_min, y_mid, z_min), (x_mid, y_max, z_mid)),
            ((x_mid, y_mid, z_min), (x_max, y_max, z_mid)),
            ((x_min, y_min, z_mid), (x_mid, y_mid, z_max)),
            ((x_mid, y_min, z_mid), (x_max, y_mid, z_max)),
            ((x_min, y_mid, z_mid), (x_mid, y_max, z_max)),
            ((x_mid, y_mid, z_mid), (x_max, y_max, z_max)),
        ]

        # create the eight children nodes
        for i in range(8):
            node.children[i] = OctreeNode(child_bounds[i])

        # redistribute the points to the children nodes
        for point, point_idx in zip(node.points, node.points_idx):
            idx = self.get_child_index(node.bounds, point)
            node.children[idx].points.append(point)
            node.children[idx].points_idx.append(point_idx)

        node.points = []  # clear the points in the parent node
        node.points_idx = []  # clear the points indices in the parent node

    def get_child_index(self, bounds, point):
        """Get the index of the child node that the point belongs to"""
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        # determine the index of the child node based on the point's position
        index = 0
        if point[0] > x_mid: index += 1
        if point[1] > y_mid: index += 2
        if point[2] > z_mid: index += 4
        return index

    def intersects(self, bounds, segment3D, radius=None):
        """check if the line segment intersects with the bounding box"""
        # use AABB intersection test
        if radius is None:
            return self.aabb_intersects_line(bounds[0], bounds[1], segment3D)
        else:
            return self.aabb_intersects_cylinder(bounds[0], bounds[1], segment3D, radius)

    def aabb_intersects_line(self, min_bounds, max_bounds, segment3D):
        """simple AABB-line segment intersection test"""
        t_min = 0
        t_max = 1
        line_start = segment3D.P1()
        line_end = segment3D.P2()

        for i in range(3):
            if line_end[i] == line_start[i]:
                if line_start[i] < min_bounds[i] or line_start[i] > max_bounds[i]:
                    return False
            else:
                t1 = (min_bounds[i] - line_start[i]) / (line_end[i] - line_start[i])
                t2 = (max_bounds[i] - line_start[i]) / (line_end[i] - line_start[i])

                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))

                if t_min > t_max:
                    return False

        return True

    def aabb_intersects_cylinder(self, min_bounds, max_bounds, segment3D, radius):
        """AABB-cylinder intersection test with a fixed radius"""
        # check if the line segment intersects with the AABB
        if self.aabb_intersects_line(min_bounds, max_bounds, segment3D):
            return True
        # check if the AABB bounds are within the cylinder
        points = [
            np.array([min_bounds[0], min_bounds[1], min_bounds[2]]),
            np.array([min_bounds[0], min_bounds[1], max_bounds[2]]),
            np.array([min_bounds[0], max_bounds[1], min_bounds[2]]),
            np.array([min_bounds[0], max_bounds[1], max_bounds[2]]),
            np.array([max_bounds[0], min_bounds[1], min_bounds[2]]),
            np.array([max_bounds[0], min_bounds[1], max_bounds[2]]),
            np.array([max_bounds[0], max_bounds[1], min_bounds[2]]),
            np.array([max_bounds[0], max_bounds[1], max_bounds[2]]),
        ]
        for point in points:
            if segment3D.distance_point_to_line(point) <= 2 * radius:
                return True

        return False


    def query(self, segment3D, radius=None):
        """Query the points that intersect with the line segment"""
        found_points = []
        self._query_recursive(self.root, segment3D, found_points, radius)
        return found_points

    def _query_recursive(self, node, segment3D, found_points, radius):
        """recursively query the points that intersect with the line segment"""
        if not node:
            return

        if self.intersects(node.bounds, segment3D, radius):
            if node.is_leaf():
                found_points.extend(node.points)
            else:
                for child in node.children:
                    self._query_recursive(child, segment3D, found_points, radius)

    def query_indices(self, segment3D, radius=None):
        """Query the indices of the points that intersect with the line segment"""
        found_indices = []
        self._query_indices_recursive(self.root, segment3D, found_indices, radius)
        return found_indices

    def _query_indices_recursive(self, node, segment3D, found_indices, radius):
        """recursively query the indices of the points that intersect with the line segment"""
        if not node:
            return

        if self.intersects(node.bounds, segment3D, radius):
            if node.is_leaf():
                found_indices.extend(node.points_idx)
            else:
                for child in node.children:
                    self._query_indices_recursive(child, segment3D, found_indices, radius)

    def print_tree(self):
        """Print the tree structure"""
        self._print_tree_recursive(self.root, 0)

    def _print_tree_recursive(self, node, depth):
        """recursively print the tree structure"""
        if not node:
            return

        print("  " * depth + f"Depth: {depth}, Num points: {len(node.points)}")
        for child in node.children:
            self._print_tree_recursive(child, depth + 1)

    def save_ply(self, path):
        """Save the points in the Octree to a PLY file"""
        points = []
        colors = []
        self._save_ply_recursive(self.root, points, colors)
        points = np.array(points)
        colors = np.array(colors)
        storePly(path, points, colors)

    def _save_ply_recursive(self, node, points, colors):
        """recursively save the points in the Octree to a list"""
        if not node:
            return

        # add the points in the leaf node with different colors
        if node.is_leaf():
            color = np.random.randint(0, 255, 3)
            for point in node.points:
                points.append(point)
                colors.append(color)
        else:
            for child in node.children:
                self._save_ply_recursive(child, points, colors)



