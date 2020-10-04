import time
import os, sys

# Read files of dir #
files = os.listdir("training_plans/final/training_5/")

# Keep xml files #
filesAll = []

for file in files:
    if("model" in file):
        file = file[11:]
        pos = file.find('.')
        file = file[:pos]

        file = int(file)
        filesAll.append(file)

filesAll = list(dict.fromkeys(filesAll))
filesAll.sort()
try:
    os.remove("training_plans/final/training_5/checkpoint")
except:
    pass

for i in range(len(filesAll)):
    with open("training_plans/final/training_5/checkpoint", "w") as file:

        file.write('model_checkpoint_path: "model.ckpt-' + str(filesAll[i]) + '"\n')
        file.write('all_model_checkpoint_paths: "model.ckpt-' + str(filesAll[i]) + '"\n')

    os.system("python eval.py --logtostderr --pipeline_config_path=training_plans/final/training_5/pipeline.config --checkpoint_dir=training_plans/final/training_5/ --eval_dir=eval/eval_5/")
    time.sleep(5)

# Petropoulakis Panagiotis
