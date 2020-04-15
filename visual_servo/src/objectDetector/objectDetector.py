from __future__ import division
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
import time
import copy
import os
import numpy as np

# Detect objects #
class objectDetector:
    def __init__(self, modelPath, labelPath):
        # Read labels for images # 
        self.labels = label_map_util.create_category_index_from_labelmap(labelPath, use_display_name=True)
        self.numClasses = len(self.labels)
        self.modelPath = modelPath
        self.labelPath = labelPath
        self.graph = tf.Graph()
        self.tensorDictCpu = {} # Tensors responsible for detection
        self.tensorDictGpu = {} # Tensors responsible for detection
        self.feedGpu = {}
        self.feedCpu = {}
        self.runOptions = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.force_gpu_compatible = True
        self.resultGpu = None
        self.result = None
        self.sess = None
        self.shapeSplit = 5118 # 1917
        # Tensor name's to split #
        self.splitNames = ['Postprocessor/Slice',
                             'Postprocessor/convert_scores',
                             'Postprocessor/ExpandDims_1',
                             'Postprocessor/stack_1',
                          ]

        # Add cpu input tensors - for 300x300 model #
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
                #tf.import_graph_def(graphDef)
                #tf.summary.FileWriter("./graphs", self.graph)
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

    def visualize(self, imageNp, result):
        vis_util.visualize_boxes_and_labels_on_image_array(
        imageNp,
        result['detection_boxes'],
        result['detection_classes'],
        result['detection_scores'],
        self.labels,
        use_normalized_coordinates=True,
        line_thickness=8)

    @staticmethod
    def getRealPixelCoordinates(width, height, x, y):
        return (int(x * width), int(y * height))
 

    @staticmethod
    def getBox(width, height, box):
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]

        xmin, ymin = objectDetector.getRealPixelCoordinates(width, height, xmin, ymin)
        xmax, ymax = objectDetector.getRealPixelCoordinates(width, height, xmax, ymax)
 
        return xmin, xmax, ymin, ymax

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
    modelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
    labelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"

    pioneerDetector = objectDetector(modelPath, labelPath)

    image = Image.open("/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/val/0_robot.jpg")
    imageNp = objectDetector.imageToNumpy(image)
    imageExpand = np.expand_dims(imageNp, axis=0)

    result = pioneerDetector.predict(imageExpand)
    print(result["detection_boxes"][0])
    
    width, height = image.size
    
    xc, yc = pioneerDetector.getCenter(width, height,  result["detection_boxes"][0])
    pioneerDetector.visualize(imageNp,  result)
    imageNp[int(yc),int(xc)] = [0, 255, 0]
  
    img = Image.fromarray(imageNp, 'RGB')
    img.show()
    
    '''
    imagesPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/val"
    images = []
    lost = 0

    for image in os.listdir(imagesPath):
        if image.endswith(".jpg"):
            images.append(imagesPath + "/" + image)

    count = 0
    t = 0
    for imagePath in images:
        if count == 30:
            break
        image = Image.open(imagePath)
        imageNp = objectDetector.imageToNumpy(image)
        imageExpand = np.expand_dims(imageNp, axis=0)

        a = time.time()
        result = pioneerDetector.predict(imageExpand)
        print(result)
        print(result["detection_boxes"][0])
        print(result["detection_scores"][0])
        exit()
        if(count != 0):
            t += time.time() - a
        count += 1
        if result["detection_scores"][0] < 0.5:
            lost += 1

    print(t / 29)
    print(lost)
'''

# Petropoulakis Panagiotis
