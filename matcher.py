import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import time

class pioneerMatcher():
    def __init__(self):

        self.templates = []

        # Read templates #
        for i in range(1, 65):
            img = Image.open(("data/t%d.jpg" % i))
            self.templates.append(imageToNumpy(img))

        # Set similarity bounds #
        self.distance = 20
        self.minInliers = 8

        ###############################
        # Find features for templates #
        ###############################
        self.kpTO = []
        self.desTO = []

        # Find ORB #
        self.orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.1, edgeThreshold=25, patchSize=25, WTA_K=4)
        for i in range(len(self.templates)):
            kp, des = self.orb.detectAndCompute(self.templates[i], None)
            self.kpTO.append(kp)
            self.desTO.append(des)

        # Init matchers #
        self.bfO = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Check similarity of image #
    def query(self, img):

        # Find features of image #
        kpIO, desIO = self.orb.detectAndCompute(img, None)

        # Check result #
        try:
            if desIO == None or kpIO == None:
                return False
        except:
            pass

        # Compute ORB #
        a = time.time()
        for i in range(len(self.templates)):

            matches = self.bfO.match(self.desTO[i], desIO)

            count = 0
            for m in matches:
                if m.distance < self.distance:
                    count += 1

            if count > self.minInliers:
                return True

            '''
            matches = sorted(matches, key = lambda x:x.distance)
            img3 = None
            img3 = cv2.drawMatches(self.templates[i],self.kpTO[i],img,kpIO,matches[:10], img3, flags=2)
            plt.imshow(img3),plt.show()
            '''

        return False

def imageToNumpy(image):
    (width, height) = image.size

    return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)

if __name__ == "__main__":

    # Init #
    pMatcher = pioneerMatcher()

    # Query image #
    count = 0
    for i in range(1, 20):
        img = Image.open(("data/e%d.jpg" % i))
        q = imageToNumpy(img)

        if not pMatcher.query(q):
            count += 1

    print('False images: {}/19'.format(count))

    count = 0
    for i in range(1, 18):
        img = Image.open(("data/q%d.jpg" % i))
        q = imageToNumpy(img)
        if pMatcher.query(q):
            count += 1

    print('True images: {}/17'.format(count))

# Petropoulakis Panagiotis
