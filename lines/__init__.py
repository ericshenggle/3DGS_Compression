import os
import numpy as np
from lines.segment3D import *
from lines.clustering import *


class Line3D:
    def __init__(self):
        self.prefix_wng_ = "Warning: "
        self.lines3D_ = []
        self.file_name_ = ""

    def lines3D(self) -> List:
        return self.lines3D_

    def load3DLinesFromTXT(self, input_folder):
        # get filename
        for f in os.listdir(input_folder):
            # 检查文件是否以 .txt 结尾
            if f.endswith('.txt'):
                file_name = os.path.basename(f)
                self.file_name_ = os.path.splitext(file_name)[0]

        filename = os.path.join(input_folder, self.file_name_ + ".txt")
        if not os.path.exists(filename):
            print(f"Warning: No 3D lines file found! filename: {filename}")
            return

        with open(filename, 'r') as file:
            for line in file:
                elements = line.strip().split()
                if len(elements) == 0:
                    continue

                # Read 3D segments
                num_segments = int(elements[0])
                segments = []
                idx = 1
                for _ in range(num_segments):
                    P1 = np.array([float(elements[idx]), float(elements[idx + 1]), float(elements[idx + 2])])
                    P2 = np.array([float(elements[idx + 3]), float(elements[idx + 4]), float(elements[idx + 5])])
                    seg3D = Segment3D(P1, P2)
                    segments.append(seg3D)
                    idx += 6

                # Read 2D residuals
                num_residuals = int(elements[idx])
                residuals = []
                idx += 1
                for _ in range(num_residuals):
                    camID = int(elements[idx])
                    segID = int(elements[idx + 1])
                    coords = np.array([float(elements[idx + 2]), float(elements[idx + 3]), float(elements[idx + 4]),
                                       float(elements[idx + 5])])
                    residuals.append(Segment2D(camID, segID, coords))
                    idx += 6

                # Construct and store the final line
                final_line = {
                    "collinear3Dsegments_": segments,
                    "underlyingCluster_": {
                        "residuals": residuals
                    }
                }
                underlyingCluster = LineCluster3D(residuals=residuals)
                final_line = FinalLine3D()
                final_line.set_segments(segments)
                final_line.set_cluster(underlyingCluster)
                self.lines3D_.append(final_line)

    def Write3DlinesToSTL(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        if len(self.lines3D_) == 0:
            print(self.prefix_wng_, "no 3D lines to save!")
            return

        filename = os.path.join(output_folder, self.file_name_ + ".stl")

        with open(filename, 'w') as file:
            file.write("solid lineModel\n")

            for current in self.lines3D_:
                for segment in current.collinear3Dsegments_:
                    P1 = segment.P1()
                    P2 = segment.P2()

                    x1 = f"{P1[0]:.6e}"
                    y1 = f"{P1[1]:.6e}"
                    z1 = f"{P1[2]:.6e}"

                    x2 = f"{P2[0]:.6e}"
                    y2 = f"{P2[1]:.6e}"
                    z2 = f"{P2[2]:.6e}"

                    file.write(" facet normal 1.0e+000 0.0e+000 0.0e+000\n")
                    file.write("  outer loop\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write(f"   vertex {x2} {y2} {z2}\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write("  endloop\n")
                    file.write(" endfacet\n")

            file.write("endsolid lineModel\n")

    def Write3DlinesToSTLEachLine(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # clear the output folder
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

        if len(self.lines3D_) == 0:
            print(self.prefix_wng_, "no 3D lines to save!")
            return

        for i, current in enumerate(self.lines3D_):
            filename = os.path.join(output_folder, self.file_name_ + f"_{i}.stl")

            with open(filename, 'w') as file:
                file.write("solid lineModel\n")

                for segment in current.collinear3Dsegments_:
                    P1 = segment.P1()
                    P2 = segment.P2()

                    x1 = f"{P1[0]:.6e}"
                    y1 = f"{P1[1]:.6e}"
                    z1 = f"{P1[2]:.6e}"

                    x2 = f"{P2[0]:.6e}"
                    y2 = f"{P2[1]:.6e}"
                    z2 = f"{P2[2]:.6e}"

                    file.write(" facet normal 1.0e+000 0.0e+000 0.0e+000\n")
                    file.write("  outer loop\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write(f"   vertex {x2} {y2} {z2}\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write("  endloop\n")
                    file.write(" endfacet\n")

                file.write("endsolid lineModel\n")

    def cluster3Dsegments(self):
        segment_to_line = []
        segments = []
        for i, line in enumerate(self.lines3D_):
            for segment in line.collinear3Dsegments_:
                segment_to_line.append(i)
                segments.append(segment)
        cluUniverse = perform_clustering(segments, segment_to_line)
        clusters = get_clusters(segments, cluUniverse)

        self.lines3D_.clear()
        idx = 0
        for k, v in clusters.items():
            print(f"Cluster {idx}: root={k}, size={len(v)}, segments={v}")
            new_segments = []
            for segID in v:
                new_segments.append(segments[segID])
            final_line = FinalLine3D()
            final_line.set_segments(new_segments)
            self.lines3D_.append(final_line)

        return self.lines3D_
