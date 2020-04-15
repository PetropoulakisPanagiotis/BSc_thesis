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
    return theta

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    transformation = np.linalg.inv(transformation)
    #source_temp.transform(transformation)
    center = target.get_center()
    center[2] += -0.05
   
    '''
    center[0] = abs(transformation[0][3] -center[0]) + center[0]
    center[1] = abs(transformation[1][3] -center[1]) + center[1]
    center[2] = abs(transformation[2][3] -center[2]) + center[2]
    '''

    c = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center)
    c.transform(transformation)
    #x, _ = target.compute_convex_hull()
    #a = target.get_axis_aligned_bounding_box()
    #o3d.visualization.draw_geometries([x, source_temp, target_temp, o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center)])
    o3d.visualization.draw_geometries([source_temp, target_temp, c])
    #o3d.visualization.draw_geometries([target_temp, o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center)])

def preprocess_point_cloud(pcd, voxel_size):
    global count     
    if count == 1:
        pcd.paint_uniform_color([1, 0, 0])
        #pcd.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, -math.pi/4.8, 0]))
        #pcd = pcd.voxel_down_sample(voxel_size=0.008)
        #pcd.normalize_normals()
        #pcd.translate([0, 0, 1])
        return pcd

    xc = -0.11
    yc = -0.929
    zc = -3.07
  
    count += 1

    pcd.paint_uniform_color([0, 1, 0])
    a = time.time()

    pcd = pcd.voxel_down_sample(voxel_size=0.008)
     
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    pcd = pcd.select_down_sample(ind)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.022,
                                             ransac_n=3,
                                             num_iterations=100)
    
    pcd = pcd.select_down_sample(inliers, invert=True)

    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.024)
    pcd = pcd.select_down_sample(ind)
    pcd = pcd.voxel_down_sample(voxel_size=0.008)
    
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.016))
    #pcd.normalize_normals()
    
    print(time.time() - a)
 
    return pcd

def execute_fast_global_registration(source, target,voxel_size):
    radius_feature = 0.008 * 3.0
    maxNN = 150

    distance_threshold = 0.008 * 2.5

    source_fpfh = o3d.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

    target_fpfh = o3d.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

    b = time.time()
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(True), 3, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.87),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(900000, 1000))

    print(time.time() - b)
    draw_registration_result(source, target,
                             result.transformation)
    
    radius = 0.016 * 1.4
    iter = 30
    fit = 1e-9
    rmse = 1e-9

    a = time.time()
  
    target = target.voxel_down_sample(0.016)
    source = source.voxel_down_sample(0.016)
    
    result = o3d.registration.registration_icp(
            source, target, radius, result.transformation,
            o3d.registration.TransformationEstimationPointToPlane(), 
            o3d.registration.ICPConvergenceCriteria(max_iteration=iter,relative_fitness=fit, relative_rmse=rmse))

    print(time.time() - a)

    trans = result.transformation
    print(trans)
    sx = LA.norm(trans[:3, 0])
    sy = LA.norm(trans[:3, 1])
    sz = LA.norm(trans[:3, 2])
    r = np.asarray([[trans[0][0]/sx , trans[0][1]/sy, trans[0][2]/sz], 
                    [trans[1][0]/sx , trans[1][1]/sy, trans[1][2]/sz],
                    [trans[2][0]/sx , trans[2][1]/sy, trans[2][2]/sz],
                    ])        

    # Camera #
    # z out
    # x left 
    # y up 

    # Open3D #
    # x: right
    # y: up 
    # z: back    

    theta = euler_angles_from_rotation_matrix(r)
    
    print(math.degrees(theta))

    return result

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../data/38.pcd")
    target = o3d.io.read_point_cloud("../../data/template.pcd")
    draw_registration_result(source, target, np.identity(4))
    source = preprocess_point_cloud(source, voxel_size)
    target = preprocess_point_cloud(target, voxel_size)

    draw_registration_result(source, target, np.identity(4))

    return source, target

if __name__ == "__main__":

    voxel_size = 0.05  # means 5cm for the dataset

    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../data/27.pcd")
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
