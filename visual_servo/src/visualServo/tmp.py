# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/clustering.py

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.random.seed(42)


def pointcloud_generator():
    yield "fragment", o3d.io.read_point_cloud(
        "../../data/1.pcd"), 0.02


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.Debug)

    cmap = plt.get_cmap("tab20")
    for pcl_name, pcl, eps in pointcloud_generator():
        print("%s has %d points" % (pcl_name, np.asarray(pcl.points).shape[0]))
        o3d.visualization.draw_geometries([pcl])

        labels = np.array(
            pcl.cluster_dbscan(eps=eps, min_points=20, print_progress=True))
        max_label = labels.max()
        print("%s has %d clusters" % (pcl_name, max_label + 1))

        colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcl])
