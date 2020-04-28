import numpy as np

f = open("/home/petropoulakis/Desktop/lr.txt", "w+")

initialLr = 0.00001
step = 1.1
for i in range(0,1250):

    mystr =  "schedule{\n\tstep: " + str(i + 1) + "\n\tlearning_rate: "
    mystr +=  str(np.format_float_positional(initialLr*((i + 1) ** step), precision=10)) + "\n}\n"

    f.write(mystr)

f.close()

