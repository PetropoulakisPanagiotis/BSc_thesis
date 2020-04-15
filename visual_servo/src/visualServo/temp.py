               #mt = np.array([[1059.94655, 0.0, 954.8832], [0.0, 1053.93268, 523.73858],[0.0, 0.0, 0.0]])
            #dist = np.asarray([[0.05626844], [-0.07419914], [0.001425079], [-0.00169517223], [0.02410768]])
            #size_of_marker = 0.02
            #rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker, mt, dist)
            #print(rvecs[-1], tvecs[-1])
            #imaxis = aruco.drawDetectedMarkers(currColor.copy(), corners, ids)    
            #for i in range(len(tvecs)):
            #   imaxis = aruco.drawAxis(imaxis, mt, dist, rvecs[i], tvecs[i], 0.05)
            #plt.imshow(frame_markers)
            #plt.show()        
            # Fix frame for detection #
            #colorFixed = cv2.cvtColor(currColor, cv2.COLOR_BGR2RGB)
            #colorExpand = np.expand_dims(colorFixed, axis=0)

            #type = 0
            # Detect #
            #result = self.pioneerDetector.predict(colorExpand)
            
            #xc, yc = self.pioneerDetector.getCenter(self.width, self.height, result["detection_boxes"][0])

            # Find angle for the center #
            #K = np.array([[self.cameraMatrixColor[0], self.cameraMatrixColor[1],self.cameraMatrixColor[2]],[self.cameraMatrixColor[3], self.cameraMatrixColor[4],self.cameraMatrixColor[5]], [self.cameraMatrixColor[6], self.cameraMatrixColor[7],self.cameraMatrixColor[8]]])
            #Ki = np.linalg.inv(K)
            #r1 = Ki.dot([self.width / 2.0 , self.height / 2.0, 1.0])       
            
            #r2 = Ki.dot([xc, yc, 1.0])       

            #cosAngle = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)) 
            #angleRadians = np.arccos(cosAngle)
           
            # Detect markers # 
            #markers = aruco.Dictionary_get(aruco.DICT_6X6_250)
            #for i in range (2):
            #    for j in range(7):
            #        currColor[int(markerY + i)][int(markerX + j)] = [0, 255, 0] 
            
            #cv2.imshow("image", currColor)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #if sum(score >= 0.5 for score in result["detection_scores"]) > 1:
             #   type = 3

            #elif result["detection_scores"][0] < 0.65:
             #   if result["detection_scores"][0] < 0.5:
              #      type = 1
              # else:
              #    type = 2

            #print(type)
    
            #for i in range(50): 
            #    currColor[yMean][xMean + i] = (0, 255, 0) 
            #plt.imshow(currColor)
            #plt.show()

