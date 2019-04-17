# AutonomousVehiclePaper

* `无人驾驶相关论文速递`。论文速递信息见[Issues](https://github.com/DeepTecher/AutonomousVehiclePaper/issues),可在issue下进行讨论、交流 、学习:smile: :smile: :smile:  
* `无人驾驶相关学者信息` 见 [scholars in Autonomous Vehicle](scholars%20in%20Autonomous%20Vehicle.md)    
* `CVPR2019 无人驾驶相关论文` 见[CVPR2019](CVPR2019.md) :blush: :blush: :blush: 

## 如何贡献（Be Free）
看到相关无人驾驶最新论文，欢迎大伙儿提交到Issues上，当然也可以在对应Issues下进行讨论论文和补充相关材料～～   
`Again,Be Free！`一切都是为了更好的学术！！！  
#### 论文速递遵从以下模板：

* Issue 标题为论文题目
* 描述模板：
> * [Paper Title](Paper link)
> * 提交日期; 年份-月-日  
> * 团队: XXXX  
> * 作者: XXXX  
> * 摘要: 中文摘要  



------
* 以下论文主要收录相关方向SOTA代码的论文，框架大致按照下图无人驾驶系统系统架构来整理,其中相同类别论文按照时间排序。  
> 注：以下统计的时间为在Arxiv提交的时间 

![img](imgs/ARCHITECTURE%20OF%20SELF-DRIVING%20CARS%20.svg)

## 感知系统|Precision
### 其他|Other
#### 物体检测|Object Detection  
* [SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition](https://arxiv.org/abs/1903.05831)
:trophy: :+1: SOTA on consumer grade hardware at large scale       
作者： Yuntao Chen, Chenxia Han, Yanghao Li, Zehao Huang, Yi Jiang, Naiyan Wang, Zhaoxiang Zhang     
机构：图森      
日期：2019-03-14   
代码：[tusimple/simpledet](https://github.com/tusimple/simpledet) :star: 1340  
摘要:Object detection and instance recognition play a central role in many AI applications like autonomous driving, 
video surveillance and medical image analysis. However, training object detection models on large scale datasets remains 
computationally expensive and time consuming. This paper presents an efficient and open source object detection framework 
called SimpleDet which enables the training of state-of-the-art detection models on consumer grade hardware at large scale.
 SimpleDet supports up-to-date detection models with best practice. SimpleDet also supports distributed training with near 
 linear scaling out of box. Codes, examples and documents of SimpleDet can be found at [this https URL](https://github.com/tusimple/simpledet).     
  
* [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784),
:trophy: SOTA for Birds Eye View Object Detection on KITTI Cyclists Moderate  
作者：Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom  
机构：nuTonomy（安波福下的公司）      
日期： 2018-12-14   
代码：[traveller59/second.pytorch](https://github.com/traveller59/second.pytorch):star: 228    
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
时间：2018-11-12  
代码：[sshkhr/BigDataCup18_Submission](https://github.com/sshkhr/BigDataCup18_Submission)  
摘要: This paper summarizes the design, experiments and results of our solution to the Road Damage Detection and Classification 
Challenge held as part of the 2018 IEEE International Conference On Big Data Cup. Automatic detection and classification 
of damage in roads is an essential problem for multiple applications like maintenance and autonomous driving. We demonstrate
 that convolutional neural net based instance detection and classfication approaches can be used to solve this problem. 
 In particular we show that Mask-RCNN, one of the state-of-the-art algorithms for object detection, localization and instance 
 segmentation of natural images, can be used to perform this task in a fast manner with effective results. We achieve a mean 
 F1 score of 0.528 at an IoU of 50% on the task of detection and classification of different types of damages in real-world 
 road images acquired using a smartphone camera and our average inference time for each image is 0.105 seconds on an NVIDIA 
 GeForce 1080Ti graphic card. The code and saved models for our approach can be found here : this [https URL](https://github.com/sshkhr/BigDataCup18_Submission) Submission
* [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/abs/1712.02294) 
:+1: 2nd best model for 3D Object Detection on KITTI Cars Hard   
作者：Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh, Steven Waslander  
机构：滑铁卢大学工程学院机械与机电工程系   
日期：2018-06-12 (2017-12-06 v1)     
代码：[kujason/avod](https://github.com/kujason/avod) :star: 429   
摘要：We present AVOD, an Aggregate View Object Detection network for autonomous driving scenarios. The proposed neural
 network architecture uses LIDAR point clouds and RGB images to generate features that are shared by two subnetworks:
a region proposal network (RPN) and a second stage detector network. The proposed RPN uses a novel architecture capable
of performing multimodal feature fusion on high resolution feature maps to generate reliable 3D object proposals for
multiple object classes in road scenes. Using these proposals, the second stage detection network performs accurate
oriented 3D bounding box regression and category classification to predict the extents, orientation, and classification
of objects in 3D space. Our proposed architecture is shown to produce state of the art results on the KITTI 3D object
detection benchmark while running in real time with a low memory footprint, making it a suitable candidate for 
deployment on autonomous vehicles. Code is at: this https URL

 * [SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving](https://arxiv.org/abs/1612.01051),
 :trophy: SOTA for KITTI(2016)  
 作者：Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer  
 机构：伯克利、[DeepScale](http://deepscale.ai/)（专注于自动驾驶感知技术）  
 日期：2017-11-29（2016-12-04 v1版本）  
 代码: TensorFLow:[BichenWuUCB/squeezeDet](https://github.com/BichenWuUCB/squeezeDet) :star:631  
 摘要:Object detection is a crucial task for autonomous driving. In addition to requiring high accuracy to ensure safety, 
 object detection for autonomous driving also requires real-time inference speed to guarantee prompt vehicle control, 
 as well as small model size and energy efficiency to enable embedded system deployment. In this work, we propose SqueezeDet, 
 a fully convolutional neural network for object detection that aims to simultaneously satisfy all of the above constraints. 
 In our network we use convolutional layers not only to extract feature maps, but also as the output layer to compute bounding 
 boxes and class probabilities. The detection pipeline of our model only contains a single forward pass of a neural network, 
 thus it is extremely fast. Our model is fully-convolutional, which leads to small model size and better energy efficiency. 
 Finally, our experiments show that our model is very accurate, achieving state-of-the-art accuracy on the KITTI benchmark.
 
#### 语义分割|Semantic Segmentation
* [A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes](https://arxiv.org/abs/1812.09953),
 :trophy: SOTA for Image-to-Image Translation on SYNTHIA-to-Cityscapes.  
 作者:Yang Zhang, Philip David, Hassan Foroosh, Boqing Gong  
 日期：2019-01-10（2018-12-24 v1版本）  
 代码：[YangZhang4065/AdaptationSeg](https://github.com/YangZhang4065/AdaptationSeg):star:68    
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
  
* [RTSeg: Real-time Semantic Segmentation Comparative Study](https://arxiv.org/abs/1803.02758)
Benchmarking Framework（Cityscapes dataset for urban scenes）      
作者：Mennatullah Siam, Mostafa Gamal, Moemen Abdel-Razek, Senthil Yogamani, Martin Jagersand  
机构：阿尔伯塔大学、开罗大学    
日期：2018-06-10 （2018-03-07 v1）  
代码：[MSiam/TFSegmentation](https://github.com/MSiam/TFSegmentation) :star: 437  
摘要：Semantic segmentation benefits robotics related applications especially autonomous driving. Most of the research 
on semantic segmentation is only on increasing the accuracy of segmentation models with little attention to computationally
efficient solutions. The few work conducted in this direction does not provide principled methods to evaluate the different
design choices for segmentation. In this paper, we address this gap by presenting a real-time semantic segmentation 
benchmarking framework with a decoupled design for feature extraction and decoding methods. The framework is comprised 
of different network architectures for feature extraction such as VGG16, Resnet18, MobileNet, and ShuffleNet. It is also 
comprised of multiple meta-architectures for segmentation that define the decoding methodology. These include SkipNet, UNet, 
and Dilation Frontend. Experimental results are presented on the Cityscapes dataset for urban scenes. The modular design 
allows novel architectures to emerge, that lead to 143x GFLOPs reduction in comparison to SegNet. This benchmarking framework 
is publicly available at "[this https URL](https://github.com/MSiam/TFSegmentation) ".  

* [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://arxiv.org/abs/1612.07695)  
:trophy: SOTA for KITTI(Road Segmentation)   
作者: Marvin Teichmann, Michael Weber, Marius Zoellner, Roberto Cipolla, Raquel Urtasun    
机构：多伦多大学计算机科学、剑桥大学工程系、[FZI研究中心](https://www.fzi.de/en/about-us/)、Uber ATG    
日期：2018-05-08 （2016-12-22）   
代码：[MarvinTeichmann/MultiNet](https://github.com/MarvinTeichmann/MultiNet) :star: 367     
摘要：While most approaches to semantic reasoning have focused on improving performance, in this paper we argue that 
computational times are very important in order to enable real time applications such as autonomous driving. Towards this goal,
 we present an approach to joint classification, detection and semantic segmentation via a unified architecture where 
 the encoder is shared amongst the three tasks. Our approach is very simple, can be trained end-toend and performs 
 extremely well in the challenging KITTI dataset, outperforming the state-of-the-art in the road segmentation task. 
 Our approach is also very efficient, allowing us to perform inference at more then 23 frames per second. Training scripts 
 and trained weights to reproduce our results can be found here: [MarvinTeichmann/MultiNet](https://github.com/MarvinTeichmann/MultiNet)    
 
* [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 :trophy: :+1: SOTA in (Semantic Segmentation & Real-Time Semantic Segmentation)，[more detail](https://paperswithcode.com/paper/pyramid-scene-parsing-network) 、
 CVPR 2017     
 作者：Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia    
 机构：香港中文大学、商汤    
 日期：2017-04-27   
 代码：[tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab)、[hszhao/PSPNet](https://github.com/hszhao/PSPNet) :star: 1004  
 摘要：Scene parsing is challenging for unrestricted open vocabulary and diverse scenes. In this paper, we exploit the 
 capability of global context information by different-region-based context aggregation through our pyramid pooling module 
 together with the proposed pyramid scene parsing network (PSPNet). Our global prior representation is effective to produce 
 good quality results on the scene parsing task, while PSPNet provides a superior framework for pixel-level prediction tasks. 
 The proposed approach achieves state-of-the-art performance on various datasets. It came first in ImageNet scene parsing 
 challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields new record of mIoU accuracy 85.4% 
 on PASCAL VOC 2012 and accuracy 80.2% on Cityscapes. 



#### 传感器融合|Sensor Fusion  
* [Online Temporal Calibration for Monocular Visual-Inertial Systems](https://arxiv.org/abs/1808.00692),
SOTA，IROS 2018,IMU和（单目）摄像头融合的校正方法，用来校准IMU和相机之间的时间偏移。  
作者：[Tong Qin](http://www.qintong.xyz/), [Shaojie Shen(沈劭劼)](http://uav.ust.hk/group/)   
机构： [香港科技大学航空机器人](http://uav.ust.hk/)  
代码：[HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono):star:1345    
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
* [Virtual to Real Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1704.03952),在虚拟环境通过强化学习来训练无人驾驶   
作者：Xinlei Pan, Yurong You, Ziyan Wang, Cewu Lu  
机构：Berkley、清华大学、上海交通大学  
日期：2017-09-26 （2017-04-13 v1）  
代码：[xinleipan/VirtualtoReal-RL](https://github.com/xinleipan/VirtualtoReal-RL) :star: 20        
摘要：Reinforcement learning is considered as a promising direction for driving policy learning. However, training autonomous
 driving vehicle with reinforcement learning in real environment involves non-affordable trial-and-error. It is more desirable 
 to first train in a virtual environment and then transfer to the real environment. In this paper, we propose a novel realistic 
 translation network to make model trained in virtual environment be workable in real world. The proposed network can convert 
 non-realistic virtual image input into a realistic one with similar scene structure. Given realistic frames as input, 
 driving policy trained by reinforcement learning can nicely adapt to real world driving. Experiments show that our proposed 
 virtual to real (VR) reinforcement learning (RL) works pretty well. To our knowledge, this is the first successful case of
  driving policy trained by reinforcement learning that can adapt to real world driving data.  

* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)  
作者：Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski 等  
机构：英伟达  
日期：2016-09-25  
代码：[marsauto/europilot](https://github.com/marsauto/europilot):star:1237、[SullyChen/Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow):star: 956 非官方  
摘要：We trained a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands. This end-to-end approach proved surprisingly powerful. With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways. It also operates in areas with unclear visual guidance such as in parking lots and on unpaved roads. 
The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal. We never explicitly trained it to detect, for example, the outline of roads. 
Compared to explicit decomposition of the problem, such as lane marking detection, path planning, and control, our end-to-end system optimizes all processing steps simultaneously. We argue that this will eventually lead to better performance and smaller systems. Better performance will result because the internal components self-optimize to maximize overall system performance, instead of optimizing human-selected intermediate criteria, e.g., lane detection. Such criteria understandably are selected for ease of human interpretation which doesn't automatically guarantee maximum system performance. Smaller networks are possible because the system learns to solve the problem with the minimal number of processing steps. 
We used an NVIDIA DevBox and Torch 7 for training and an NVIDIA DRIVE(TM) PX self-driving car computer also running Torch 7 for determining where to drive. The system operates at 30 frames per second (FPS).

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

 







