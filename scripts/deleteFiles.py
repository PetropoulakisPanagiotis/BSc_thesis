import os

dir = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/full'
#file = "/home/petropoulakis/Desktop/similarLog.txt"
file = "/home/petropoulakis/Desktop/check.txt"

with open(file, 'r') as f:

    line = f.readline()
    counter = 0
    while line:

        xml = os.path.join(dir, line[:-5] + ".xml")
        name = os.path.join(dir, line[:-5] + ".jpg")

        try:
            os.unlink(name)
        except:
            x = 0
        try:
            os.unlink(xml)
        except:
            x = 0
        line = f.readline()
