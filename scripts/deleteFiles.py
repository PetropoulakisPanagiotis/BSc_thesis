import os

#######################
# Delete inalid files #
#######################

dir = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/fine_aug'
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
            pass

        try:
            os.unlink(xml)
        except:
            pass

        line = f.readline()

# Petropoulakis Panagiotis #
