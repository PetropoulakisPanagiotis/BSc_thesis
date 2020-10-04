from __future__ import division
import numpy as np
import xml.etree.ElementTree as ET
import glob

# IoU between a box and all clusters #
def iou(box, clusters):

    intersection = np.minimum(clusters[:, 0], box[0]) * np.minimum(clusters[:, 1], box[1])

    clusterArea = clusters[:, 0] * clusters[:, 1]
    boxArea = box[0] * box[1]

    union = (boxArea + clusterArea) - intersection

    return intersection / union

# Assign each box to the closest cluster and #
# return the average IoU score for all boxes #
def avgIou(boxes, clusters):

    count = 0
    for i in range(boxes.shape[0]):

        iouBox = iou(boxes[i], clusters)
        bestIouBox = np.max(iouBox)

        count += bestIouBox

    return count / boxes.shape[0]

# Perform k-means and return the centroids #
def kmeans(boxes, k=5):
    numBoxes= boxes.shape[0]

    # Choose randomly the initial clusters #
    clusters = boxes[np.random.choice(numBoxes, k, replace=False)]

    # Cluster per box #
    currClusters = np.zeros((numBoxes,))
    prevClusters = np.zeros((numBoxes,))

    while True:
        for box in range(numBoxes):
            distances = 1 - iou(boxes[box], clusters)

            # Assign closest cluster to the current box #
            currClusters[box] = np.argmin(distances)

        # Terminate condition: unchanged clusters #
        if (prevClusters == currClusters).all():
            break

        # Update clusters #
        for cluster in range(k):
            clusters[cluster] = np.median(boxes[currClusters == cluster], axis=0)

        prevClusters = currClusters

    return clusters

# Read xmls #
def parseDataset(path):
	boxes = []

        for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

                height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

                xmin = int(tree.findtext("./object/bndbox/xmin")) / width
                ymin = int(tree.findtext("./object/bndbox/ymin")) / height
                xmax = int(tree.findtext("./object/bndbox/xmax")) / width
                ymax = int(tree.findtext("./object/bndbox/ymax")) / height

                boxes.append([xmax - xmin ,ymax - ymin])

        return boxes

if __name__== "__main__":

    boxes1 = parseDataset("../images/val/")
    boxes2 = parseDataset("../images/train/")
    boxes3 = parseDataset("../images/fine/")

    boxes = np.array(boxes1 + boxes2 + boxes3)

    print("Number of boxes: " + str(boxes.shape[0]))

    bestRatios = None
    bestClusters = None
    bestAvgIou = None

    # Run mutiple kmeans and keep the best partition #
    for i in range(20):
        clusters = kmeans(boxes)
        ratios =  clusters[:, 0] / clusters[:, 1]
        avgIouBoxes = avgIou(boxes, clusters) * 100

        if i == 0:
            bestRatios = ratios
            bestClusters = clusters
            bestAvgIou = avgIouBoxes
        elif avgIouBoxes > bestAvgIou:
            bestRatios = ratios
            bestClusters = clusters
            bestAvgIou = avgIouBoxes

    print("Centroids:\n {}".format(bestClusters))
    print("Ratios:\n {}".format(sorted(bestRatios)))
    print("Average Iou: {:.2f}%".format(bestAvgIou))

# Petropoulakis Panagiotis 
