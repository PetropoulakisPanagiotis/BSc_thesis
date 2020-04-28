import os
import functools
import xml.etree.ElementTree as ET
import os, sys
import shutil

f = open('/home/petropoulakis/Desktop/copy.txt','r')
dir1 = '/home/petropoulakis/Desktop/x_train'
dir2 = '/home/petropoulakis/Desktop/x_val'
mainDir = '/home/petropoulakis/Desktop/lost/'

currDir = dir1

line = f.readline().rstrip()
while line:

    if("val" == line):
        currDir = dir2

    currFile = mainDir + line

    try:
        shutil.copy2(currFile, currDir)
    except:
        print(line)
    line = f.readline().rstrip()
f.close()
