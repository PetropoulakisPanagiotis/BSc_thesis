from __future__ import division
import open3d as o3d
import numpy as np
import math
from numpy import linalg as LA
from helpers import *

# Create point cloud from the detected box  #
# Convert it to open3d format               #
# Camera frame: front Z, right X, down Y    #
# Open3D frame: back Z, right X, up Y       #
def createPCD(color, depth, mapX, mapY, xMin, xMax, yMin, yMax, devDepth):

    # Find center of the box #
    xc = int((xMin + xMax) / 2)
    yc = int((yMin + yMax) / 2)

    # Depth of center #
    centerDepth = helpers.estimateDepthPixel(depth, xc, yc)

    # Initialize point cloud of RoI #
    xyz = []
    
    # Initialize open3d format point cloud #
    pcd = o3d.geometry.PointCloud()
    
    # Create point cloud from depth # 
    for y in range(yMin, yMax):
        for x in range(xMin, xMax):
            
            currDepth = depth[y][x] / 1000.0

            # Cut bad points and points far away from the object #
            if currDepth == 0 or currDepth > centerDepth + devDepth or currDepth < currDepth - devDepth:
                continue

            xyz.append([mapX[x] * currDepth, mapY[y] * currDepth, currDepth])
 
    # Copy pcd #
    xyz = np.asarray(xyz)
    pcd.points = o3d.utility.Vector3dVector(xyz)
   
    # Flip point cloud - Open3D convention #
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Fix pcd #
    pcd = cleanPCD(pcd)

    return pcd 

# Remove floor and bad points #
def cleanPCD(pcd):

    pcd = pcd.voxel_down_sample(voxel_size=0.008)
     
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    pcd = pcd.select_down_sample(ind)

    # Remove floor #
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.022,
                                             ransac_n=3,
                                             num_iterations=100)
    
    pcd = pcd.select_down_sample(inliers, invert=True)

    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.024)
    pcd = pcd.select_down_sample(ind)
    
    pcd = pcd.voxel_down_sample(voxel_size=0.008)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.016))
    
    return pcd

# Find position and orientation of object #
def estimatePos(template, pcd, transformationEstimation, iter=30, fit=1e-9, rmse=1e-9):
    radius = 0.016 * 1.4
  
    template = target.voxel_down_sample(0.016)
    pcd = pcd.voxel_down_sample(0.016)
    
    result = o3d.registration.registration_icp(
            pcd, template, radius, transformationEstimation,
            o3d.registration.TransformationEstimationPointToPlane(), 
            o3d.registration.ICPConvergenceCriteria(max_iteration=iter, relative_fitness=fit, relative_rmse=rmse))

    # Check if registration is accurate #
    if result.fitness < 0.65 or result.correspondence_set.shape[0] < 450:
        return -1, -1, -1, -1
    
    # Pick result #
    t = result.transformation
    
    # Extract rotation matrix #
    sx = LA.norm(t[:3, 0])
    sy = LA.norm(t[:3, 1])
    sz = LA.norm(t[:3, 2])
    r = np.asarray([[t[0][0]/sx , t[0][1]/sy, t[0][2]/sz], 
                    [t[1][0]/sx , t[1][1]/sy, t[1][2]/sz],
                    [t[2][0]/sx , t[2][1]/sy, t[2][2]/sz],
                    ])        
   
    # Find orientation # 
    theta = helpers.rotationToEuler(r)
    
    # Calculate x, y, z(center) in Open3D frame #
    centerT = template.get_center() 
    centerT[2] += -0.05
  
    centerPcd = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=centerT)
    tInv = np.linalg.inv(t)
    centerPcd.transform(tInv)
   
    centerP = createPCD.get_center() 
    x = centerP[0]
    y = centerP[1]
    z = centerP[2]

    # Convert result to camera frame #
    x, y, z = convertOpen3DToCamera(x, y, z)

    return x, y, z, theta

def convertOpen3DToCamera(x, y, z):
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])

    c = np.asarray([[x], [y], [z]])
    r = np.dot(rx, c) 

    x = r[0][0]
    y = r[1][0]
    z = r[2][0]

    return x, y, z

# Petropoulakis Panagiotis
