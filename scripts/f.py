import os

dir = '/home/petropoulakis/Desktop/lost'

for i in range(19627, 20393 + 1):
    name = os.path.join(dir, str(i) + "a1.jpg")

    try:
        os.unlink(name)
    except:
        x = 0
