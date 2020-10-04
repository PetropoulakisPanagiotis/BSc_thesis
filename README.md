[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-270/)

#### <p align="center">A Leader-Follower Mobile Robot Scheme using an RGBD-Camera and MobileNets</p>

<p align="center">
<img src="experiments.png" width="490px" height="320px"> <br /> <br />
</p>

Additional resources such videos and the Pioneer -3AT dataset can be found at my personal [Google Drive](https://drive.google.com/drive/folders/1FQmJPG-sj2xHcH3shPUANIkJkorwxRfR?usp=sharing).

#### Requirements
* ROS distribution: (melodic)[http://wiki.ros.org/melodic/Installation/Ubuntu].
* Tools and libraries for a ROS Interface to the Kinect: (iai_kinect2)[https://github.com/code-iai/iai_kinect2].
* Object Detection: (tensorflow 1.15)[https://www.tensorflow.org/install/pip], CUDA 9.0 and (TensorFlow Official Models 1.13.0)[https://github.com/tensorflow/models/releases].
* The required python packages can be found inside the BSc\_thesis/code folder at requirements2.txt and requirements3.txt files.
* ROS package to establish and manage the communication between the Desktop and the Follower's computer: (multimaster_fkie)[http://wiki.ros.org/multimaster_fkie].
* Simulation to tune the controller: (summit_xl_sim)[https://github.com/RobotnikAutomation/summit_xl_sim].    

*The code has been tested on Ubuntu 18.04.5 LTS in Python 2.7.17 and Python 3.6.9.


#### Citation
If you find my thesis useful in your research, please consider citing:

```bib
@thesis{Petropoulakis2020,
    author    = {Petropoulakis Panagiotis},
    title     = {A Leader-Follower Mobile Robot Scheme using an RGBD-Camera and MobileNets},
    type = {bscthesis}
    url   = {https://github.com/PetropoulakisPanagiotis/BSc_thesis},
    institution = {Department of Informatics and Telecommunications of the University of Athens},
    year      = {2020},
}
```
#### Acknowledgements 
Supervisor: [Prof. Kostas J. Kyriakopoulos](http://www.controlsystemslab.gr/kkyria/)<br />
Advisors: [Dr. George Karras](https://scholar.google.gr/citations?user=VxIC7-cAAAAJ&hl=el), [Michalis Logothetis](https://scholar.google.com/citations?user=fFLmpWsAAAAJ&hl=en), [PhD student Kostas Alevizos](http://www.controlsystemslab.gr/main/members/kostas-alevizos/),
[PhD student Sotiris Aspragkathos](http://www.controlsystemslab.gr/main/members/sotiris-aspragkathos/)
