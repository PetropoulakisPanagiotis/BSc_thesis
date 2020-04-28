import os, sys
import functools
import subprocess
import commands
def myCompare(x, y):
    x = int(x)
    y = int(y)

    return (x - y)

# Read checkpoints #
dir = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/training_plans/training_1/'
allFiles = []
idFiles = []

files = os.listdir(dir)
for file in files:
    if(file.startswith("m")) and file.endswith(".index"):
        allFiles.append(file)

        idFiles.append(int(file.partition("-")[2].partition(".")[0]))


idFiles.sort(reverse=True)

for id in idFiles[::4]:

    f = open(dir + "checkpoint" , "w")
    f.write("model_checkpoint_path: " + '"' + "model.ckpt-" + str(id) + '"' + "\n")
    f.write("all_model_checkpoint_paths: " + '"' + "model.ckpt-" + str(id) + '"')

    command = 'python /home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/eval.py --logtostderr --pipeline_config_path=/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/training_plans/training_1/pipeline.config --checkpoint_dir=/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/training_plans/training_1/ --eval_dir=/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/eval/'
    p = "/home/petropoulakis/Desktop/results/" + str(id) + ".txt"
    os.system(command + "2&>1" + " " + p)

    f.close()
    exit()
    sys.stdout.flush()
