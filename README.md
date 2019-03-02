# AutonomousVehiclePaper
无人驾驶相关论文速递。论文速递信息见[Issues](https://github.com/DeepTecher/AutonomousVehiclePaper/issues)  

以下论文将大致按照下图无人驾驶系统系统架构来整理。
> 注：以下统计的时间为在Arxiv提交的时间 

![img](imgs/ARCHITECTURE%20OF%20SELF-DRIVING%20CARS%20.svg)

## 感知系统|Precision
### 其他|Other
#### 物体检测|Object Detection
* [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784),
:trophy: SOTA for Birds Eye View Object Detection on KITTI Cyclists Moderate  
作者：Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom  
日期： 2018-12-14   
代码：[traveller59/second.pytorch](https://github.com/traveller59/second.pytorch)   
摘要：Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this 
paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline.
 Recent literature suggests two types of encoders; fixed encoders tend to be fast but sacrifice accuracy, while encoders 
 that are learned from data are more accurate, but slower. In this work we propose PointPillars, a novel encoder which 
 utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). While the encoded 
 features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. 
 Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy
  by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art,
  even among fusion methods, with respect to both the 3D and bird's eye view KITTI benchmarks. This detection performance 
  is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state
  of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.  

* [Road Damage Detection And Classification In Smartphone Captured Images Using Mask R-CNN](https://arxiv.org/abs/1811.04535),
IEEE International Conference On Big Data Cup 2018(2018年IEEE国际大数据杯会议的道路损伤检测和分类挑战)    
作者：Janpreet Singh, Shashank Shekhar    
代码：2018-11-12  
代码：[sshkhr/BigDataCup18_Submission](https://github.com/sshkhr/BigDataCup18_Submission)  
摘要: This paper summarizes the design, experiments and results of our solution to the Road Damage Detection and Classification 
Challenge held as part of the 2018 IEEE International Conference On Big Data Cup. Automatic detection and classification 
of damage in roads is an essential problem for multiple applications like maintenance and autonomous driving. We demonstrate
 that convolutional neural net based instance detection and classfication approaches can be used to solve this problem. 
 In particular we show that Mask-RCNN, one of the state-of-the-art algorithms for object detection, localization and instance 
 segmentation of natural images, can be used to perform this task in a fast manner with effective results. We achieve a mean 
 F1 score of 0.528 at an IoU of 50% on the task of detection and classification of different types of damages in real-world 
 road images acquired using a smartphone camera and our average inference time for each image is 0.105 seconds on an NVIDIA 
 GeForce 1080Ti graphic card. The code and saved models for our approach can be found here : this https URL Submission
#### 语义分割|Semantic Segmentation
* [A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes](https://arxiv.org/abs/1812.09953),
 :trophy: SOTA for Image-to-Image Translation on SYNTHIA-to-Cityscapes.  
 作者:Yang Zhang, Philip David, Hassan Foroosh, Boqing Gong  
 日期：2019-01-10（2018-12-24 v1版本）  
 代码：[YangZhang4065/AdaptationSeg](https://github.com/YangZhang4065/AdaptationSeg)  
 摘要:During the last half decade, convolutional neural networks (CNNs) have triumphed over semantic segmentation, which is 
 one of the core tasks in many applications such as autonomous driving and augmented reality. However, to train CNNs 
 requires a considerable amount of data, which is difficult to collect and laborious to annotate. Recent advances in 
 computer graphics make it possible to train CNNs on photo-realistic synthetic imagery with computer-generated annotations. 
 Despite this, the domain mismatch between the real images and the synthetic data hinders the models' performance. Hence,
  we propose a curriculum-style learning approach to minimizing the domain gap in urban scene semantic segmentation. The 
  curriculum domain adaptation solves easy tasks first to infer necessary properties about the target domain; in particular, 
  the first task is to learn global label distributions over images and local distributions over landmark superpixels. 
  These are easy to estimate because images of urban scenes have strong idiosyncrasies (e.g., the size and spatial relations 
  of buildings, streets, cars, etc.). We then train a segmentation network, while regularizing its predictions in the 
  target domain to follow those inferred properties. In experiments, our method outperforms the baselines on two datasets 
  and two backbone networks. We also report extensive ablation studies about our approach.  

#### 传感器融合|Sensor Fusion  
* [Online Temporal Calibration for Monocular Visual-Inertial Systems](https://arxiv.org/abs/1808.00692),
SOTA，IROS 2018,IMU和（单目）摄像头融合的校正方法，用来校准IMU和相机之间的时间偏移。  
作者：[Tong Qin](http://www.qintong.xyz/), [Shaojie Shen(沈劭劼)](http://uav.ust.hk/group/)   
机构： [香港科技大学航空机器人](http://uav.ust.hk/)  
代码：[HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
摘要：Accurate state estimation is a fundamental module for various intelligent applications, such as robot navigation, 
autonomous driving, virtual and augmented reality. Visual and inertial fusion is a popular technology for 6-DOF state 
estimation in recent years. Time instants at which different sensors' measurements are recorded are of crucial importance 
to the system's robustness and accuracy. In practice, timestamps of each sensor typically suffer from triggering and 
transmission delays, leading to temporal misalignment (time offsets) among different sensors. Such temporal offset 
dramatically influences the performance of sensor fusion. To this end, we propose an online approach for calibrating 
temporal offset between visual and inertial measurements. Our approach achieves temporal offset calibration by jointly 
optimizing time offset, camera and IMU states, as well as feature locations in a SLAM system. Furthermore, the approach 
is a general model, which can be easily employed in several feature-based optimization frameworks. Simulation and
experimental results demonstrate the high accuracy of our calibration approach even compared with other state-of-art 
offline tools. The VIO comparison against other methods proves that the online temporal calibration significantly benefits 
visual-inertial systems. The source code of temporal calibration is integrated into our public project, VINS-Mono.

  

## 决策系统|Decision Making
### 运动规划|Motion Planer
* [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079),
:+1: :+1: :+1: :+1: :+1: Waymo出品，通过模仿学习对无人车进行运动规划，全文中文翻译:[知乎|每周一篇 & 无人驾驶](https://zhuanlan.zhihu.com/p/57275593)  
作者：Mayank Bansal, Alex Krizhevsky, Abhijit Ogale  
机构：Waymo Research  
日期:2018-12-07  
代码：[aidriver/ChauffeurNet](https://github.com/aidriver/ChauffeurNet) 非官方   
摘要: Our goal is to train a policy for autonomous driving via imitation learning that is robust enough to drive a real vehicle. 
We find that standard behavior cloning is insufficient for handling complex driving scenarios, even when we leverage a 
perception system for preprocessing the input and a controller for executing the output on the car: 30 million examples 
are still not enough. We propose exposing the learner to synthesized data in the form of perturbations to the expert's 
driving, which creates interesting situations such as collisions and/or going off the road. Rather than purely imitating 
all data, we augment the imitation loss with additional losses that penalize undesirable events and encourage progress 
-- the perturbations then provide an important signal for these losses and lead to robustness of the learned model. 
We show that the ChauffeurNet model can handle complex situations in simulation, and present ablation experiments that
 emphasize the importance of each of our proposed changes and show that the model is responding to the appropriate causal 
 factors. Finally, we demonstrate the model driving a car in the real world.

 







