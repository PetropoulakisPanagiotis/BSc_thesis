import open3d as o3d
import numpy as np
import copy
import time
import math

for i in range(1, 38):
    #pcd = o3d.io.read_point_cloud("../../data/%d.pcd" % 15)
    pcd = o3d.io.read_point_cloud("../../data/template.pcd")

    pcd.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, -math.pi/4.8, 0]))
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("../../data/template.pcd",pcd)
    break 
    cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.02)
    pcd = pcd.select_down_sample(ind)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)

    pcd = pcd.voxel_down_sample(voxel_size=0.008)
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.023,
                                             ransac_n=3,
                                             num_iterations=150)
    
    pcd = pcd.select_down_sample(inliers, invert=True)

    cl, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.02)
    pcd = pcd.select_down_sample(ind)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.024, max_nn=30))

    o3d.visualization.draw_geometries([pcd])
    
    break
    xyz = []
    cl = []
    points = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)
    print(pcd.points) 
    pcd.translate([0, 0, 1]) 
    for j in range(points.shape[0]):
        if np.array_equal(points[j], [0.0, 0.0, 0.0]):
            continue

        xyz.append(points[j])
        cl.append(color[j])   

    pcd.points = o3d.utility.Vector3dVector(xyz) 
    pcd.colors = o3d.utility.Vector3dVector(cl)

    print(np.asarray(pcd.points))
    o3d.io.write_point_cloud("../../data/%d.pcd" % i,pcd)
