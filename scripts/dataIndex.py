import numpy as np
import pandas as pd

# Write a data frame - keep track the data                   #
# Columns: name, background, depth, illumination, angle, pos #

dataInfo = [] # Keep rows of the data frame
fileName = "./data_info.csv"

# Read data #
while True:
    option = raw_input("Q(quit) or S(append data): ")
    if(option != "s"):
        break
    else:
        row = raw_input("Give image name, background, depth, illumination, angle, and pos(seperated with spaces): ")
        dataInfo.append(row.split())

# Create the dataframe and save it #
df = pd.DataFrame(dataInfo, columns = ["Name", "Background", "Depth", "Illumination", "Angle", "Pos"])
df.to_csv(fileName, index=False)
