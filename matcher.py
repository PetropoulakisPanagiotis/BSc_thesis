import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

class pioneerMatcher():
    def __init__(self):

        self.templates = []

        # Read templates #
        for i in range(1, 65):
            img = Image.open(("data/t%d.jpg" % i))
            img = imageToNumpy(img)
            img = cv2.resize(img, (170, 170), interpolation=cv2.INTER_LINEAR)
            self.templates.append(img)

        # Set similarity bounds #
        self.distance = 20
        self.minInliers = 13

        ###############################
        # Find features for templates #
        ###############################
        self.kpTO = []
        self.desTO = []

        # Find ORB #
        self.orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.15, edgeThreshold=25, patchSize=25, WTA_K=4)
        for i in range(len(self.templates)):
            kp, des = self.orb.detectAndCompute(self.templates[i], None)
            self.kpTO.append(kp)
            self.desTO.append(des)

        # Init matcher #
        self.bfO = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Check similarity of image #
    def query(self, img):

        if(img.size == 0):
            return False

        img = cv2.resize(img, (170,170), interpolation=cv2.INTER_LINEAR)

        # Find features of image #
        kpIO, desIO = self.orb.detectAndCompute(img, None)

        # Check result #
        try:
            if desIO == None or kpIO == None:
                return False
        except:
            pass

        for i in range(len(self.templates)):

            # Compute ORB #
            matches = self.bfO.match(self.desTO[i], desIO)

            count = 0
            for m in matches:
                if m.distance < self.distance:
                    count += 1

            if count > self.minInliers:
                return True

        return False

    def visualize(self, matches, img1, kp1, img2, kp2):
        matches = sorted(matches, key = lambda x:x.distance)
        img2 = None
        img2 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img3, flags=2)
        plt.imshow(img2),plt.show()

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
