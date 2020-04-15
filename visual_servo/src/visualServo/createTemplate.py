# examples/Python/Advanced/multiway_registration.py

import open3d as o3d
import numpy as np
from math import pi

voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

def load_point_clouds(voxel_size=0.0):
    pcds = []
    pcd = o3d.io.read_point_cloud("../../data/templattt1.pcd")
    pcds.append(pcd)
    pcd = o3d.io.read_point_cloud("../../data/templattt2.pcd")
    pcds.append(pcd)

    '''
    cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.02)
    pcd = pcd.select_down_sample(ind)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)
   
    pcd = pcd.voxel_down_sample(voxel_size=0.008)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.015,
                                             ransac_n=3,
                                             num_iterations=150)
    
    pcd = pcd.select_down_sample(inliers, invert=True)
    
    cl, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.02)
    pcd = pcd.select_down_sample(ind)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.024, max_nn=30))
    #pcds.append(pcd)
    for i in range(27, 38):
        pcd = o3d.io.read_point_cloud("../../data/%d.pcd" % i)
        cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.02)
        pcd = pcd.select_down_sample(ind)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pcd = pcd.select_down_sample(ind)
       
        pcd = pcd.voxel_down_sample(voxel_size=0.008)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.012,
                                                 ransac_n=3,
                                                 num_iterations=150)
        
        pcd = pcd.select_down_sample(inliers, invert=True)
        
        cl, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.02)
        pcd = pcd.select_down_sample(ind)
    
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        pcd = pcd.select_down_sample(ind)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.024, max_nn=30))
        
        pcds.append(pcd)
    
    '''
    print(len(pcds))
    return pcds

def pairwise_registration(source, target):
    radius_feature = 0.024
    maxNN = 30
    distance_threshold = 0.05
    source_fpfh = o3d.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

    target_fpfh = o3d.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))


    '''
    result = o3d.registration.registration_fast_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
   
    '''

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(7000000, 800))


    radius = 0.018
    iter = 140
    fit = 1e-9
    rmse = 1e-9
   
    ''' 
    result = o3d.registration.registration_colored_icp(
            source, target, radius, result.transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=fit,
                                                    relative_rmse=rmse,
                                                    max_iteration=iter))

    '''
    result = o3d.registration.registration_icp(
            source, target, radius, result.transformation,
            o3d.registration.TransformationEstimationPointToPoint(), o3d.registration.ICPConvergenceCriteria(max_iteration=iter))


    information_icp = o3d.registration.get_information_matrix_from_point_clouds(source, target, radius,result.transformation)

    return result.transformation, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.015,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down)

    print("Make a combined point cloud")
    pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    #pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)


    o3d.visualization.draw_geometries([pcd_combined])

    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.008)
    
    cl, ind = pcd_combined.remove_radius_outlier(nb_points=10, radius=0.02)
    pcd_combined = pcd_combined.select_down_sample(ind)

    cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd_combined = pcd_combined.select_down_sample(ind)
   
    o3d.visualization.draw_geometries([pcd_combined])
    o3d.io.write_point_cloud("../../data/template.pcd", pcd_combined)
