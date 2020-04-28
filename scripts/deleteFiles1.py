import os
import glob
dir = '/home/petropoulakis/Desktop/lost'

for i in range(228, 276 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0

name = glob.glob(dir + "/" + "469" + "*")

try:
    os.unlink(name[0])
except:
    x = 0

for i in range(471, 488 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0
        
for i in range(509, 511 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0

name = glob.glob(dir + "/" + "536" + "*")

try:
    os.unlink(name[0])
except:
    x = 0
for i in range(539, 554 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0

for i in range(580, 583 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0


for i in range(605, 606 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0


for i in range(621, 679 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0


for i in range(817, 1000 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0

for i in range(1002, 1132 + 1):

    name = glob.glob(dir + "/" + str(i) + "*")

    try:
        os.unlink(name[0])
    except:
        x = 0
