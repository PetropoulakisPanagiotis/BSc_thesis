from __future__ import division
import open3d as o3d
import numpy as np
import copy
import time
from numpy import linalg as LA
import math
count = 0

def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    global count     
    if count == 1:
        pcd.paint_uniform_color([1, 0, 0])
        pcd = pcd.voxel_down_sample(voxel_size=0.016)
        return pcd

    count += 1

    pcd.paint_uniform_color([0, 1, 0])
    a = time.time()

    pcd = pcd.voxel_down_sample(voxel_size=0.016)
     
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.016,
                                             ransac_n=3,
                                             num_iterations=100)
    
    pcd = pcd.select_down_sample(inliers, invert=True)

    cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.03)
    pcd = pcd.select_down_sample(ind)
   
    pcd = pcd.voxel_down_sample(voxel_size=0.016)
 
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.024, max_nn=30))
    print(time.time() - a)
 
    return pcd

def execute_fast_global_registration(source, target,voxel_size):
    radius_feature = 0.016 * 2.5
    maxNN = 70

    distance_threshold = 0.016 * 2

    source_fpfh = o3d.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

    target_fpfh = o3d.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

    b = time.time()
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(7000000, 400))
    
    print(time.time() - b)
    draw_registration_result(source, target,
                             result.transformation)
    

    print(result)
    radius = 0.016 * 1.2
    iter = 50
    fit = 1e-3
    rmse = 1e-3


    a = time.time()
    result = o3d.registration.registration_icp(
            source, target, radius, result.transformation,
            o3d.registration.TransformationEstimationPointToPoint(), 
            o3d.registration.ICPConvergenceCriteria(max_iteration=iter,relative_fitness=fit, relative_rmse=rmse))

    print(result)

    print(time.time() - a)

    trans = result.transformation
    sx = LA.norm(trans[:][0])
    sy = LA.norm(trans[:][1])
    sz = LA.norm(trans[:][2])
    r = np.asarray([[trans[0][0]/sx , trans[0][1]/sy, trans[0][2]/sz], 
                    [trans[1][0]/sx , trans[1][1]/sy, trans[1][2]/sz],
                    [trans[2][0]/sx , trans[2][1]/sy, trans[2][2]/sz],
                    ])        
       
    roll, pitch, yaw = euler_angles_from_rotation_matrix(r)
    #print(math.degrees(roll))
    #print(math.degrees(pitch))
    #print(math.degrees(yaw))

    return result

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../data/20.pcd")
    target = o3d.io.read_point_cloud("../../data/template.pcd")
    draw_registration_result(source, target, np.identity(4))
    source = preprocess_point_cloud(source, voxel_size)
    target = preprocess_point_cloud(target, voxel_size)

    draw_registration_result(source, target, np.identity(4))

    return source, target

if __name__ == "__main__":

    voxel_size = 0.05  # means 5cm for the dataset


    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../data/20.pcd")
    target = o3d.io.read_point_cloud("../../data/template.pcd")
    draw_registration_result(source, target, np.identity(4))
    source = preprocess_point_cloud(source, voxel_size)
    target = preprocess_point_cloud(target, voxel_size)

    draw_registration_result(source, target, np.identity(4))

    result_fast = execute_fast_global_registration(source, target,
                                                   voxel_size)
   
    print(result_fast)
    print(np.asarray(result_fast.correspondence_set).shape[0])
 
    draw_registration_result(source, target,
                             result_fast.transformation)
