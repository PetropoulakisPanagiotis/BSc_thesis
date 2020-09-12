from __future__ import division
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
import numpy as np
import time
import copy

# Detect objects #
class objectDetector:
    def __init__(self, modelPath = "../objectDetector/model/frozen_inference_graph.pb", labelPath = "../objectDetector/annotations/label_map.pbtxt", minAreaBox=1100, maxAreaBox=30000, offset=10):

        # Read labels for images # 
        self.labels = label_map_util.create_category_index_from_labelmap(labelPath, use_display_name=True)

        self.numClasses = len(self.labels)
        self.modelPath = modelPath
        self.labelPath = labelPath

        self.graph = tf.Graph()
        self.sess = None
        self.tensorDictCpu = {} # Tensors responsible for detection
        self.tensorDictGpu = {} # Tensors responsible for detection

        self.feedGpu = {} # Input 
        self.feedCpu = {}

        self.runOptions = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.force_gpu_compatible = True

        self.resultGpu = None
        self.result = None
        self.shapeSplit = 5118 # 500 x 500

        # Box bounds #
        self.minAreaBox = minAreaBox # Smaller box is false detection
        self.maxAreaBox = maxAreaBox
        self.offset = offset # Generate bigger box to secure that leader lies inside   
        self.debug = True

        # Tensor name's to split      #
        # Place postproccesing to cpu #
        self.splitNames = ['Postprocessor/Slice',
                             'Postprocessor/convert_scores',
                             'Postprocessor/ExpandDims_1',
                             'Postprocessor/stack_1',
                          ]

        # Add cpu input tensors - for 500x500 model #
        with self.graph.as_default():
            cpuInput = [tf.placeholder(tf.float32, shape=(None, self.shapeSplit, self.numClasses), name=self.splitNames[0]),
                        tf.placeholder(tf.float32, shape=(None, self.shapeSplit, self.numClasses + 1), name=self.splitNames[1]),
                        tf.placeholder(tf.float32, shape=(None, self.shapeSplit, 1, 4), name=self.splitNames[2]),
                        tf.placeholder(tf.float32, shape=(None), name=self.splitNames[3])
                    ]

        # Save graph definition for the previous operations #
        targetDef = self.graph.as_graph_def()

        # Clear graph #
        self.graph = tf.Graph()

        with self.graph.as_default():

            graphDef = tf.GraphDef()

            with tf.gfile.GFile(self.modelPath, 'rb') as fid:

                # Read the given graph(of ssdlite) #
                serializedGraph = fid.read()
                graphDef.ParseFromString(serializedGraph)

                # Parse graph #
                nodes = {}
                inputs = {}
                count = 0
                nodesPos = {}
                for node in graphDef.node:
                    name = self.fixName(node.name)
                    nodes[name] = node
                    inputs[name] = [self.fixName(input) for input in node.input]
                    nodesPos[name] = count
                    count += 1

                #################
                # Make GPU part #
                #################

                # From the split nodes keep all the previous inputs #
                # Start from a node and travel the grpah down       #
                notVisited = self.splitNames
                gpuNodesNames =  set()

                while notVisited:

                    # Get node #
                    name = notVisited[0]
                    del notVisited[0]

                    # Already visited #
                    if name in gpuNodesNames:
                        continue

                    # Add inputs of this node #
                    gpuNodesNames.add(name)
                    notVisited += inputs[name]

                # Sort gpuNodes #
                gpuNodesNames = sorted(list(gpuNodesNames), key=lambda name: nodesPos[name])

                # Create new graph definition for the gpu part #
                gpuNodesDef = tf.GraphDef()
                for gpuNodeName in gpuNodesNames:
                    gpuNodesDef.node.extend([copy.deepcopy(nodes[gpuNodeName])])

                #################
                # Make CPU part #
                #################

                # Find CPU part #
                cpuNodesNames = set()
                for name in nodesPos:
                    if name in gpuNodesNames:
                        continue

                    cpuNodesNames.add(name)

                # Sort cpuNodes #
                cpuNodesNames = sorted(list(cpuNodesNames), key=lambda name: nodesPos[name])

                # Create new graph definition for the cpu part #
                cpuNodesDef = tf.GraphDef()
                for node in targetDef.node:
                    cpuNodesDef.node.extend([node])

                for cpuNodeName in cpuNodesNames:
                    cpuNodesDef.node.extend([copy.deepcopy(nodes[cpuNodeName])])

                # Import graph definitions to devices #
                tf.import_graph_def(gpuNodesDef, name='')

                # Force cpu #
                with tf.device('/cpu:0'):
                    tf.import_graph_def(cpuNodesDef, name='')

            # Init session #
            self.sess = tf.Session()

            # CPU part - output #
            self.tensorDictCpu['num_detections:0'] = tf.get_default_graph().get_tensor_by_name('num_detections:0')
            self.tensorDictCpu['detection_boxes:0'] = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            self.tensorDictCpu['detection_scores:0'] = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            self.tensorDictCpu['detection_classes:0'] = tf.get_default_graph().get_tensor_by_name('detection_classes:0')

            # GPU part - output #
            self.tensorDictGpu['Postprocessor/Slice:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/Slice:0')
            self.tensorDictGpu['Postprocessor/convert_scores:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/convert_scores:0')
            self.tensorDictGpu['Postprocessor/ExpandDims_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            self.tensorDictGpu['Postprocessor/stack_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/stack_1:0')

            # Get image input tensor #
            self.feedGpu['image_tensor:0'] = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # CPU input tensors #
            self.feedCpu['Postprocessor/Slice_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/Slice_1:0')
            self.feedCpu['Postprocessor/convert_scores_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/convert_scores_1:0')
            self.feedCpu['Postprocessor/ExpandDims_1_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            self.feedCpu['Postprocessor/stack_1_1:0'] = tf.get_default_graph().get_tensor_by_name('Postprocessor/stack_1_1:0')

        if(self.debug):
            self.__str__()

    # Predict image - Shape(images, height, width, 3) # 
    def predict(self, image):

        self.resultGpu = self.sess.run(self.tensorDictGpu, feed_dict={self.feedGpu['image_tensor:0']: image}, options=self.runOptions)
        tmpDict = {self.feedCpu['Postprocessor/Slice_1:0']:self.resultGpu['Postprocessor/Slice:0'],
                   self.feedCpu['Postprocessor/convert_scores_1:0']:self.resultGpu['Postprocessor/convert_scores:0'],
                   self.feedCpu['Postprocessor/ExpandDims_1_1:0']:self.resultGpu['Postprocessor/ExpandDims_1:0'],
                   self.feedCpu['Postprocessor/stack_1_1:0']:self.resultGpu['Postprocessor/stack_1:0'],
                   }

        self.result = self.sess.run(self.tensorDictCpu, feed_dict=tmpDict, options=self.runOptions)

        # Fix result #
        self.result['num_detections'] = int(self.result['num_detections:0'][0])
        self.result['detection_classes'] = self.result['detection_classes:0'][0].astype(np.int64)
        self.result['detection_scores'] = self.result['detection_scores:0'][0]
        self.result['detection_boxes'] = self.result['detection_boxes:0'][0]

        return self.result

    # Draw box in image #
    def visualize(self, imageNp, result):
        vis_util.visualize_boxes_and_labels_on_image_array(
        imageNp,
        result['detection_boxes'],
        result['detection_classes'],
        result['detection_scores'],
        self.labels,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Check if box is out of bounds #
    def validBox(self, xMin, xMax, yMin, yMax, width, height):

        if(self.minAreaBox <= 0 or self.maxAreaBox <= 0 or width <= 0 or height <= 0 or xMin < 0 or xMax >= width or yMin < 0 or yMax >= height):
            return False, "Detector: validBox: bad arguments"

        if (xMax - xMin) * (yMax - yMin) < self.minAreaBox or (xMax - xMin) * (yMax - yMin) > self.maxAreaBox:
            return False, "Detector: validBox: bad area box"

        return True, "Detector: success"

    # Create bigger box from the original #
    # Use valid box if needed             #
    # Input(in pixels)                    #
    def getNewBox(self, xMin, xMax, yMin, yMax):
        xMin = xMin - self.offset
        xMax = xMax + self.offset
        yMin = yMin - self.offset
        yMax = yMax + self.offset

        return xMin, xMax, yMin, yMax

    def __str__(self):
        print("")
        print("Detector parameters:")
        print("=============================================================================================================")
        print("Model path: " + self.modelPath)
        print("Label path: " + self.labelPath)
        print("Max area box: " + str(self.maxAreaBox) + " pixel")
        print("Min area box: " + str(self.minAreaBox) + " pixel")
        print("Offset: " + str(self.offset) + " pixel")
        print("=============================================================================================================")
        print("")

    @staticmethod
    def getRealPixelCoordinates(width, height, x, y):
        return (int(x * width), int(y * height))

    @staticmethod
    # Center of box #
    def getCenter(xMin, xMax, yMin, yMax):
        return (int((xMax + xMin)/2), int((yMax + yMin)/2))

    @staticmethod
    # Box normalized coordinates to pixel coordinates #
    def getBox(width, height, box):
        yMin = box[0]
        xMin = box[1]
        yMax = box[2]
        xMax = box[3]

        if width <= 0 or height <= 0:
            return 0, 0, 0, 0, "Detector: getBox: bad arguments"

        if yMin > 1 or yMin < 0:
            return 0, 0, 0, 0, "Detector: getBox: bad arguments"

        if xMin > 1 or xMin < 0:
            return 0, 0, 0, 0, "Detector: getBox: bad arguments"

        if yMax > 1 or yMax < 0:
            return 0, 0, 0, 0, "Detector: getBox: bad arguments"

        if xMax > 1 or xMax < 0:
            return 0, 0, 0, 0, "Detector: getBox: bad arguments"

        xMin, yMin = objectDetector.getRealPixelCoordinates(width, height, xMin, yMin)
        xMax, yMax = objectDetector.getRealPixelCoordinates(width, height, xMax, yMax)

        # Fix values #
        if yMax == height:
            yMax -= 1

        if xMax == width:
            xMax -= 1

        return xMin, xMax, yMin, yMax, "Detector: success"

    @staticmethod
    def imageToNumpy(image):
        (width, height) = image.size

        return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)

    @staticmethod
    def fixName(name):
        if name.startswith("^"):
           return name[1:]
        else:
            return name.split(":")[0]

# Test #
if  __name__ == '__main__':

    pioneerDetector = objectDetector()

    image = Image.open("/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/val/0_robot.jpg")
    #image = Image.open("/home/csl-mini-pc/Desktop/petropoulakis/TensorFlow/workspace/robot_detection/13_robot.jpg")

    imageNp = objectDetector.imageToNumpy(image)
    imageExpand = np.expand_dims(imageNp, axis=0)
    width, height = image.size

    for i in range(2):
        a = time.time()
	result = pioneerDetector.predict(imageExpand)
        if i == 1: # i == 0 is warm up time(slower)
            print(time.time() - a)

    xMin, xMax, yMin, yMax, code = pioneerDetector.getBox(width, height, result["detection_boxes"][0])

    # Check box for safety #
    print(pioneerDetector.validBox(xMin, xMax, yMin, yMax, width, height))

    # Show result #
    pioneerDetector.visualize(imageNp,  result)
    img = Image.fromarray(imageNp, 'RGB')
    img.show()

# Petropoulakis Panagiotis
