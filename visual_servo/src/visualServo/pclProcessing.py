from __future__ import division
import open3d as o3d
import numpy as np
from numpy import linalg as LA
import math
import helpers

# Create point cloud from the detected box  #
# Convert it to open3d format               #
# Camera frame: front Z, right X, down Y    #
# Open3D frame: back Z, right X, up Y       #
# xMin: in pixels                           #
def createPCD(color, depth, mapX, mapY, xMin, xMax, yMin, yMax, maxDepth, minNumPoints=1500, maxNumPoints=10000):

    if(color.size == 0 or depth.size == 0 or depth.shape[0] != color.shape[0] or depth.shape[1] != color.shape[1]): 
        return None, -1
    
    if(len(mapX) != color.shape[1] or len(mapY) != color.shape[0]):
        return None, -1

    if(xMin < 0 or xMax >= color.shape[1] or yMin < 0 or yMax >= color.shape[0]):
        return None, -1
 
    if(maxDepth <= 0 or minNumPoints <= 0 or maxNumPoints <= 0 or maxNumPoints < minNumPoints):
        return None, -1
    
    # Initialize point cloud of RoI #
    xyz = []
    pcd = o3d.geometry.PointCloud()

    # Find center of the box #
    xc = int((xMin + xMax) / 2)
    yc = int((yMin + yMax) / 2)

    # Depth of center #
    centerDepth = helpers.estimateDepthPixel(depth, xc, yc)
    
    # Create point cloud from depth # 
    for y in range(yMin, yMax):
        for x in range(xMin, xMax):
            
            currDepth = depth[y][x] / 1000.0

            # Cut bad points and points far away from the object #
            if currDepth == 0 or (centerDepth != 0 and (currDepth > centerDepth + maxDepth or currDepth < currDepth - maxDepth)):
                continue

            currX = mapX[x] * currDepth
            currY = mapY[y] * currDepth

            xyz.append([currX, currY, currDepth])

    # Not enough inliers # 
    if(len(xyz) < minNumPoints):
        return None, -2

    # Copy pcd #
    xyz = np.asarray(xyz)
    pcd.points = o3d.utility.Vector3dVector(xyz)
     
    # Flip point cloud - Open3D convention #
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Fix pcd #
    pcd, code = cleanPCD(pcd)
    
    # Not enough inliers # 
    if(len(pcd.points) < minNumPoints or len(pcd.points) > maxNumPoints):
        return None, -2

    return pcd, code

# Remove floor and bad points             #
# Pcd must not contain none/hidden points #
def cleanPCD(pcd, voxelSize=0.008):
    if(voxelSize <= 0.0 or pcd == None or pcd.has_points() == False):
        return None, -1
    
    pcd = pcd.voxel_down_sample(voxel_size=voxelSize)
     
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    pcd = pcd.select_down_sample(ind)

    # Remove floor #
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.022,
                                             ransac_n=3,
                                             num_iterations=100)
    
    pcd = pcd.select_down_sample(inliers, invert=True)

    cl, ind = pcd.remove_radius_outlier(nb_points=15, radius=voxelSize * 3)
    pcd = pcd.select_down_sample(ind)
    
    pcd = pcd.voxel_down_sample(voxel_size=voxelSize)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(voxelSize * 2))
    
    return pcd, 0

# Find position and orientation of object       #
# Use 3D model and perform local registration   #
# Result is with respect to camera frame        #
def estimatePos(template, pcd, transformationEstimation, voxelSize = 0.008, iter=50, fit=1e-9, rmse=1e-9, minInliers=400, minFitness = 0.65):
    
    if(voxelSize <= 0 or iter <= 0 or fit <= 0 or rmse <= 0 or fit >= 1 or rmse >= 1):
        return -1, -1, -1, -1, np.identity(4), -1

    if(template == None or pcd == None or pcd.has_points() == False or template.has_points() == False):
        return -1, -1, -1, -1, np.identity(4), -1

    if(minInliers <= 0 or minFitness <= 0 or minFitness > 1):
        return -1, -1, -1, -1, np.identity(4), -1

    if(transformationEstimation.shape != (4,4)):
        return -1, -1, -1, -1, np.identity(4), -1

    template = template.voxel_down_sample(voxelSize * 2)
    pcd = pcd.voxel_down_sample(voxelSize * 2)
  
    print(transformationEstimation) 
    pcd.transform(transformationEstimation)
    o3d.visualization.draw_geometries([template, pcd])
    print("a")
    # Initial registration #
    if(np.all(transformationEstimation == np.identity(4))):
        cTemplate = template.get_center()
        cTemplate[2] -= 0.05

        cPcd = pcd.get_center()

        dif =  cTemplate - cPcd

        pcd.translate(dif)
    
        radius_feature = voxelSize * 2 * 2.5
        maxNN = 200

        distance_threshold = voxelSize * 2 * 3.5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

        template_fpfh = o3d.registration.compute_fpfh_feature(
            template,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=maxNN))

        result = o3d.registration.registration_ransac_based_on_feature_matching(
            pcd, template, pcd_fpfh, template_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(), 3, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(1000000, 1700))
        transformationEstimation = result.transformation

    radius = voxelSize * 2 * 1.4

    result = o3d.registration.registration_icp(
            pcd, template, radius, transformationEstimation,
            o3d.registration.TransformationEstimationPointToPlane(), 
            o3d.registration.ICPConvergenceCriteria(max_iteration=iter, relative_fitness=fit, relative_rmse=rmse))

    print(result.transformation) 
    pcd.transform(result.transformation)
    o3d.visualization.draw_geometries([template, pcd])

    print("b")
    # Check if registration is accurate #
    if result.fitness < minFitness or len(result.correspondence_set) < minInliers:
        return -1, -1, -1, -1, np.identity(4), -2
    
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
   
    # Find orientation       #
    # With respect to camera #
    # Theta: y axis rotation # 
    theta = helpers.rotationToEuler(r)
         
    # Get center of the template in Open3D frame #
    centerT = template.get_center() 
    centerT[2] += -0.05 # For this specific template(robot) 
 
    # Calculate position of the current center #
    # Use triangle mesh for calculations       #
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=centerT)
    tInv = np.linalg.inv(t) # Get inverse transformation
    mesh.transform(tInv) # Transform triangle mesh 
   
    centerP = mesh.get_center() 
    x = centerP[0]
    y = centerP[1]
    z = centerP[2]

    # Convert result to camera frame #
    x, y, z = convertOpen3DToCamera(x, y, z)

    return x, y, z, theta, result.transformation, 0

# Open3D to Kinect frame #
def convertOpen3DToCamera(x, y, z):
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])

    c = np.asarray([[x], [y], [z]])
    result = np.dot(rx, c) 

    x = result[0][0]
    y = result[1][0]
    z = result[2][0]

    return x, y, z

# Petropoulakis Panagiotis
