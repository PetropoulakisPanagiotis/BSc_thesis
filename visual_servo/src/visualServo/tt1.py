            '''
            if ids == None or len(ids) == 0: 
                self.badDetections += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                self.lostFrames += 1
                continue
 
            
            # Check center # 
            if helpers.checkOutOfBounds(centerX, centerY, self.width, self.height):
                self.success = "Detection out of bounds" 
                break

            # Find 3D coordinates for center with respect to the camera #
            success, cameraX, cameraY, cameraZ = helpers.getXYZ(self.K, centerX, centerY, currDepth, self.width, self.height)
            if success == False:
                self.badPointsTotal += 1
                self.lostFrames += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                continue

            # Fix initial depth #
            if depthDevFlag == 0:
                self.prevDepth = cameraZ
                depthDevFlag = 1

            # Check deviation of depth # 
            if cameraZ > self.prevDepth + self.depthDev or cameraZ < self.prevDepth - self.depthDev:
                self.badDevTotal += 1
                self.lostFrames += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                self.prevDepth = cameraZ
                continue

            #################################################
            # Perform servoing. Publish velocities to robot #
            ################################################
            success = self.servo(cameraX, cameraY, cameraZ)
            if success == False:
                self.success = "Servo out of bounds" 
                break
           
            # Fix stats #
            frameRate += 1

            if self.lostConsecutiveFrames != 0:
                self.lostConsecutiveFrames = 0
       
        '''

