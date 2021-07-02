[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-270/)

#### <p align="center">A Leader-Follower Mobile Robot Scheme using an RGBD-Camera and MobileNets</p>

<p align="center">
<img src="experiments.png" width="490px" height="320px"> <br /> <br />
</p>

:zap:<b>You can receive Pioneer-3AT dataset upon request at petropoulakispanagiotis[at]gmail.com</b> 

#### Requirements
* ROS distribution: [melodic](http://wiki.ros.org/melodic/Installation/Ubuntu).
* Tools and libraries for a ROS Interface to the Kinect: [iai_kinect2](https://github.com/code-iai/iai_kinect2).
* Object Detection: [tensorflow 1.15](https://www.tensorflow.org/install/pip), CUDA 9.0 and [TensorFlow Official Models 1.13.0](https://github.com/tensorflow/models/releases).
* The required python packages can be found inside the BSc_thesis/code folder into the requirements2.txt and requirements3.txt files.
* ROS package to establish and manage the communication between the Desktop and the Follower's computer: [multimaster_fkie](http://wiki.ros.org/multimaster_fkie).
* Simulation to tune the controller: [summit_xl_sim](https://github.com/RobotnikAutomation/summit_xl_sim).    

:zap:The code has been tested on Ubuntu 18.04.5 LTS in Python 2.7.17 and Python 3.6.9.

#### Citation
If you find my thesis useful in your research, please consider citing:

```bib
@thesis{Petropoulakis2020,
    author      = {Petropoulakis Panagiotis, Konstantinos J. Kyriakopoulos},
    title       = {A Leader-Follower Mobile Robot Scheme using an RGBD-Camera and MobileNets},
    type        = {bscthesis}
    url         = {https://github.com/PetropoulakisPanagiotis/BSc_thesis},
    institution = {Department of Informatics and Telecommunications of the University of Athens},
    year        = {2020},
}
```
#### Acknowledgements 
Supervisor: [Prof. Konstantinos J. Kyriakopoulos](http://www.controlsystemslab.gr/kkyria/)<br />
Advisors: [Dr. George Karras](https://scholar.google.gr/citations?user=VxIC7-cAAAAJ&hl=el), [Michalis Logothetis](https://scholar.google.com/citations?user=fFLmpWsAAAAJ&hl=en), [PhD student Kostas Alevizos](http://www.controlsystemslab.gr/main/members/kostas-alevizos/),
[PhD student Sotiris Aspragkathos](http://www.controlsystemslab.gr/main/members/sotiris-aspragkathos/)
