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
* [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1902.09738),CVPR 2019  
作者:Peiliang Li, Xiaozhi Chen, Shaojie Shen  
机构：香港科技大学、大疆   
日期：2019-04-10（2019-02-26 v1） 
代码：[HKUST-Aerial-Robotics/Stereo-RCNN](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)  :star:241   
摘要：We propose a 3D object detection method for autonomous driving by fully exploiting the sparse and dense, semantic 
and geometry information in stereo imagery. Our method, called Stereo R-CNN, extends Faster R-CNN for stereo inputs to 
simultaneously detect and associate object in left and right images. We add extra branches after stereo Region Proposal 
Network (RPN) to predict sparse keypoints, viewpoints, and object dimensions, which are combined with 2D left-right boxes 
to calculate a coarse 3D object bounding box. We then recover the accurate 3D bounding box by a region-based photometric 
alignment using left and right RoIs. Our method does not require depth input and 3D position supervision, however, 
outperforms all existing fully supervised image-based methods. Experiments on the challenging KITTI dataset show that 
our method outperforms the state-of-the-art stereo-based method by around 30% AP on both 3D detection and 3D localization 
tasks. Code has been released at this https URL.

* [SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition](https://arxiv.org/abs/1903.05831)
:trophy: :+1: SOTA on consumer grade hardware at large scale       
作者： Yuntao Chen, Chenxia Han, Yanghao Li, Zehao Huang, Yi Jiang, Naiyan Wang, Zhaoxiang Zhang     
机构：图森      
日期：2019-03-14   
代码：[tusimple/simpledet](https://github.com/tusimple/simpledet) :star: 1546  
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

* [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244),
:trophy: KITTI for 3D Object Detection (Cars) : #2,Cars-Easy(AP:84.32%);#1,Cars-Moderate(AP:75.42%);#1,Cars-Hard(AP:67.86%)  
作者：Shaoshuai Shi, Xiaogang Wang, Hongsheng Li  
机构：香港中文大学   
日期：2018-12-11  
代码：[sshaoshuai/PointRCNN](https://github.com/sshaoshuai/PointRCNN) :star:262
摘要：In this paper, we propose PointRCNN for 3D object detection from raw point cloud. The whole framework is composed 
of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical 
coordinates to obtain the final detection results. Instead of generating proposals from RGB image or projecting point 
cloud to bird's view or voxels as previous methods do, our stage-1 sub-network directly generates a small number of 
high-quality 3D proposals from point cloud in a bottom-up manner via segmenting the point cloud of whole scene into 
foreground points and background. The stage-2 sub-network transforms the pooled points of each proposal to canonical 
coordinates to learn better local spatial features, which is combined with global semantic features of each point learned 
in stage-1 for accurate box refinement and confidence prediction. Extensive experiments on the 3D detection benchmark of 
KITTI dataset show that our proposed architecture outperforms state-of-the-art methods with remarkable margins by using 
only point cloud as input.

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
 
* [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199)
作者：Martin Simon, Stefan Milz, Karl Amende, Horst-Michael Gross
机构：法里奥、伊尔默瑙理工大学  
日期：2018-09-24 (2018-03-16 v1)
代码：[AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO) :star: 191
摘要：Lidar based 3D object detection is inevitable for autonomous driving, because it directly links to environmental 
understanding and therefore builds the base for prediction and motion planning. The capacity of inferencing highly sparse 
3D data in real-time is an ill-posed problem for lots of other application areas besides automated vehicles, e.g. augmented 
reality, personal robotics or industrial automation. We introduce Complex-YOLO, a state of the art real-time 3D object 
detection network on point clouds only. In this work, we describe a network that expands YOLOv2, a fast 2D standard object 
detector for RGB images, by a specific complex regression strategy to estimate multi-class 3D boxes in Cartesian space. 
Thus, we propose a specific Euler-Region-Proposal Network (E-RPN) to estimate the pose of the object by adding an imaginary 
and a real fraction to the regression network. This ends up in a closed complex space and avoids singularities, which occur 
by single angle estimations. The E-RPN supports to generalize well during training. Our experiments on the KITTI benchmark 
suite show that we outperform current leading methods for 3D object detection specifically in terms of efficiency. We 
achieve state of the art results for cars, pedestrians and cyclists by being more than five times faster than the fastest 
competitor. Further, our model is capable of estimating all eight KITTI-classes, including Vans, Trucks or sitting pedestrians 
simultaneously with high accuracy.  

* [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/abs/1712.02294) 
:+1: #2 for KITTI 3D Object Detection for cars: #2 Cars-Hard(AP:66.38%)  
作者：Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh, Steven Waslander  
机构：滑铁卢大学工程学院机械与机电工程系   
日期：2018-06-12 (2017-12-06 v1)     
代码：[kujason/avod](https://github.com/kujason/avod) :star: 468   
摘要：We present AVOD, an Aggregate View Object Detection network for autonomous driving scenarios. The proposed neural
 network architecture uses LIDAR point clouds and RGB images to generate features that are shared by two subnetworks:
a region proposal network (RPN) and a second stage detector network. The proposed RPN uses a novel architecture capable
of performing multimodal feature fusion on high resolution feature maps to generate reliable 3D object proposals for
multiple object classes in road scenes. Using these proposals, the second stage detection network performs accurate
oriented 3D bounding box regression and category classification to predict the extents, orientation, and classification
of objects in 3D space. Our proposed architecture is shown to produce state of the art results on the KITTI 3D object
detection benchmark while running in real time with a low memory footprint, making it a suitable candidate for 
deployment on autonomous vehicles. Code is at: this https URL

* [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/abs/1711.08488)
:trophy: :+1:  SOTA（Object Localization & 3D Object Detection）:Cars、Cyclists、Pedestrian 
作者：Charles R. Qi, Wei Liu, Chenxia Wu, Hao Su, Leonidas J. Guibas
机构：斯坦福大学、Nuro公司、加州大学圣地亚哥分校   
日期：2018-04-13（2017-11-22 v1）
代码：[charlesq34/pointnet](https://github.com/charlesq34/pointnet) :star: 1846 
摘要：In this work, we study 3D object detection from RGB-D data in both indoor and outdoor scenes. While previous methods 
focus on images or 3D voxels, often obscuring natural 3D patterns and invariances of 3D data, we directly operate on raw 
point clouds by popping up RGB-D scans. However, a key challenge of this approach is how to efficiently localize objects 
in point clouds of large-scale scenes (region proposal). Instead of solely relying on 3D proposals, our method leverages 
both mature 2D object detectors and advanced 3D deep learning for object localization, achieving efficiency as well as 
high recall for even small objects. Benefited from learning directly in raw point clouds, our method is also able to 
precisely estimate 3D bounding boxes even under strong occlusion or with very sparse points. Evaluated on KITTI and 
SUN RGB-D 3D detection benchmarks, our method outperforms the state of the art by remarkable margins while having real-time 
capability.

 * [SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving](https://arxiv.org/abs/1612.01051),
 :trophy: SOTA for KITTI(2016)  
 作者：Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer  
 机构：伯克利、[DeepScale](http://deepscale.ai/)（专注于自动驾驶感知技术）  
 日期：2017-11-29（2016-12-04 v1版本）  
 代码: TensorFLow:[BichenWuUCB/squeezeDet](https://github.com/BichenWuUCB/squeezeDet) :star:650  
 摘要:Object detection is a crucial task for autonomous driving. In addition to requiring high accuracy to ensure safety, 
 object detection for autonomous driving also requires real-time inference speed to guarantee prompt vehicle control, 
 as well as small model size and energy efficiency to enable embedded system deployment. In this work, we propose SqueezeDet, 
 a fully convolutional neural network for object detection that aims to simultaneously satisfy all of the above constraints. 
 In our network we use convolutional layers not only to extract feature maps, but also as the output layer to compute bounding 
 boxes and class probabilities. The detection pipeline of our model only contains a single forward pass of a neural network, 
 thus it is extremely fast. Our model is fully-convolutional, which leads to small model size and better energy efficiency. 
 Finally, our experiments show that our model is very accurate, achieving state-of-the-art accuracy on the KITTI benchmark.
 
 * [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396), :trophy: :+1: 
 SOTA（Object Localization & 3D Object Detection）:Cars、Cyclists、Pedestrian 
 作者：Yin Zhou, Oncel Tuzel  
 机构：苹果公司
 日期：2017-11-17  
 代码：[charlesq34/pointnet](https://github.com/charlesq34/pointnet) :star: 1846 
 摘要：Accurate detection of objects in 3D point clouds is a central problem in many applications, such as autonomous 
 navigation, housekeeping robots, and augmented/virtual reality. To interface a highly sparse LiDAR point cloud with a 
 region proposal network (RPN), most existing efforts have focused on hand-crafted feature representations, for example, 
 a bird's eye view projection. In this work, we remove the need of manual feature engineering for 3D point clouds and propose 
 VoxelNet, a generic 3D detection network that unifies feature extraction and bounding box prediction into a single stage, 
 end-to-end trainable deep network. Specifically, VoxelNet divides a point cloud into equally spaced 3D voxels and 
 transforms a group of points within each voxel into a unified feature representation through the newly introduced 
 voxel feature encoding (VFE) layer. In this way, the point cloud is encoded as a descriptive volumetric representation, 
 which is then connected to a RPN to generate detections. Experiments on the KITTI car detection benchmark show that 
 VoxelNet outperforms the state-of-the-art LiDAR based 3D detection methods by a large margin. Furthermore, our network 
 learns an effective discriminative representation of objects with various geometries, leading to encouraging results in 
 3D detection of pedestrians and cyclists, based on only LiDAR.
 
 
#### 分割|Segmentation  
* [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)，
:trophy: Semantic Segmentation: Real-time(71FPS)、Semantic Segmentation(Mean IoU 70.6%)，ICIP 2019  
作者：Yu Wang, Quan Zhou, Jia Liu, Jian Xiong, Guangwei Gao, Xiaofu Wu, Longin Jan Latecki  
机构：南京邮电大学、天普大学 
日期：2019-05-07  
代码：[xiaoyufenfei/LEDNet](https://github.com/xiaoyufenfei/LEDNet) :star:26 (暂未release)
摘要：The extensive computational burden limits the usage of CNNs in mobile devices for dense estimation tasks. 
In this paper, we present a lightweight network to address this problem,namely LEDNet, which employs an asymmetric 
encoder-decoder architecture for the task of real-time semantic segmentation.More specifically, the encoder adopts a 
ResNet as backbone network, where two new operations, channel split and shuffle, are utilized in each residual block 
to greatly reduce computation cost while maintaining higher segmentation accuracy. On the other hand, an attention 
pyramid network (APN) is employed in the decoder to further lighten the entire network complexity. Our model has 
less than 1M parameters,and is able to run at over 71 FPS in a single GTX 1080Ti GPU. The comprehensive experiments 
demonstrate that our approach achieves state-of-the-art results in terms of speed and accuracy trade-off on CityScapes dataset.
* [In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images](https://arxiv.org/abs/1903.08469) 
:trophy: #2 for Real-Time Semantic Segmentation on Cityscapes: #9,Semantic Segmentation(`Mean IoU:75.5%`);#2,Real-time(`Mean IoU:75.5%`)
;#3,Real-time(`Frame:39.9 fps`),CVPR2019  
作者：Marin Oršić, Ivan Krešo, Petra Bevandić, Siniša Šegvić  
机构：萨格勒布大学 电气工程与计算学院  
日期: 2019-04-12
代码：[orsic/swiftnet](https://github.com/orsic/swiftnet) :star: 61
摘要：Recent success of semantic segmentation approaches on demanding road driving datasets has spurred interest in many 
related application fields. Many of these applications involve real-time prediction on mobile platforms such as cars, 
drones and various kinds of robots. Real-time setup is challenging due to extraordinary computational complexity involved. 
Many previous works address the challenge with custom lightweight architectures which decrease computational complexity by 
reducing depth, width and layer capacity with respect to general purpose architectures. We propose an alternative approach 
which achieves a significantly better performance across a wide range of computing budgets. First, we rely on a light-weight 
general purpose architecture as the main recognition engine. Then, we leverage light-weight upsampling with lateral connections 
as the most cost-effective solution to restore the prediction resolution. Finally, we propose to enlarge the receptive field by 
fusing shared features at multiple resolutions in a novel fashion. Experiments on several road driving datasets show a 
substantial advantage of the proposed approach, either with ImageNet pre-trained parameters or when we learn from scratch. 
Our Cityscapes test submission entitled SwiftNetRN-18 delivers 75.5% MIoU and achieves 39.9 Hz on 1024x2048 images on GTX1080Ti.

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
  
* [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897),
Cityscapes：#2，Real-time（`65.5 Fps`）;#8 (`Mean IoU 78.9%`)、CamVid：#2，Mean IoU 68.7%;ECCV 2018  
作者：Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang  
机构：多谱信息处理技术国家级重点实验室、华中科技大学自动化学院、北大、旷视  
日期: 
代码：[ycszen/TorchSeg](https://github.com/ycszen/TorchSeg)  :star:660
摘要：Semantic segmentation requires both rich spatial information and sizeable receptive field. However, modern 
approaches usually compromise spatial resolution to achieve real-time inference speed, which leads to poor performance. 
In this paper, we address this dilemma with a novel Bilateral Segmentation Network (BiSeNet). We first design a Spatial 
Path with a small stride to preserve the spatial information and generate high-resolution features. Meanwhile, a Context 
Path with a fast downsampling strategy is employed to obtain sufficient receptive field. On top of the two paths, we 
introduce a new Feature Fusion Module to combine features efficiently. The proposed architecture makes a right balance 
between the speed and segmentation performance on Cityscapes, CamVid, and COCO-Stuff datasets. Specifically, for a 
2048x1024 input, we achieve 68.4% Mean IOU on the Cityscapes test dataset with speed of 105 FPS on one NVIDIA Titan XP 
card, which is significantly faster than the existing methods with comparable performance.

* [RTSeg: Real-time Semantic Segmentation Comparative Study](https://arxiv.org/abs/1803.02758)
Benchmarking Framework（Cityscapes dataset for urban scenes）      
作者：Mennatullah Siam, Mostafa Gamal, Moemen Abdel-Razek, Senthil Yogamani, Martin Jagersand  
机构：阿尔伯塔大学、开罗大学    
日期：2018-06-10 （2018-03-07 v1）  
代码：[MSiam/TFSegmentation](https://github.com/MSiam/TFSegmentation) :star: 459  
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
代码：[MarvinTeichmann/MultiNet](https://github.com/MarvinTeichmann/MultiNet) :star: 765     
摘要：While most approaches to semantic reasoning have focused on improving performance, in this paper we argue that 
computational times are very important in order to enable real time applications such as autonomous driving. Towards this goal,
 we present an approach to joint classification, detection and semantic segmentation via a unified architecture where 
 the encoder is shared amongst the three tasks. Our approach is very simple, can be trained end-toend and performs 
 extremely well in the challenging KITTI dataset, outperforming the state-of-the-art in the road segmentation task. 
 Our approach is also very efficient, allowing us to perform inference at more then 23 frames per second. Training scripts 
 and trained weights to reproduce our results can be found here: [MarvinTeichmann/MultiNet](https://github.com/MarvinTeichmann/MultiNet)    

* [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790),
:trophy: Cityscapes:#1 for Real-Time（76.9 fps）、#16 for Mean IoU(63.06%),CVPR 2018   
作者：Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko   
机构：鲁汶大学  
日期：2018-04-09（2017-05-24 v1）
代码：[bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax) :star: 591  
摘要：The Jaccard index, also referred to as the intersection-over-union score, is commonly employed in the evaluation of 
image segmentation results given its perceptual qualities, scale invariance - which lends appropriate relevance to small 
objects, and appropriate counting of false negatives, in comparison to per-pixel losses. We present a method for direct 
optimization of the mean intersection-over-union loss in neural networks, in the context of semantic image segmentation, 
based on the convex Lovász extension of submodular losses. The loss is shown to perform better with respect to the Jaccard 
index measure than the traditionally used cross-entropy loss. We show quantitative and qualitative differences between 
optimizing the Jaccard index per image versus optimizing the Jaccard index taken over an entire dataset. We evaluate the 
impact of our method in a semantic segmentation pipeline and show substantially improved intersection-over-union segmentation 
scores on the Pascal VOC and Cityscapes datasets using state-of-the-art deep learning segmentation architectures.
 
* [SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud](https://arxiv.org/abs/1710.07368),  
作者：Bichen Wu, Alvin Wan, Xiangyu Yue, Kurt Keutzer      
机构：伯克利    
日期：2017-10-19      
代码：[BichenWuUCB/SqueezeSeg](https://github.com/BichenWuUCB/SqueezeSeg) :star: 307   
摘要：In this paper, we address semantic segmentation of road-objects from 3D LiDAR point clouds. In particular, we wish to 
detect and categorize instances of interest, such as cars, pedestrians and cyclists. We formulate this problem as a point- wise 
classification problem, and propose an end-to-end pipeline called SqueezeSeg based on convolutional neural networks (CNN): 
the CNN takes a transformed LiDAR point cloud as input and directly outputs a point-wise label map, which is then refined
by a conditional random field (CRF) implemented as a recurrent layer. Instance-level labels are then obtained by conventional 
clustering algorithms. Our CNN model is trained on LiDAR point clouds from the KITTI dataset, and our point-wise segmentation 
labels are derived from 3D bounding boxes from KITTI. To obtain extra training data, we built a LiDAR simulator into Grand 
Theft Auto V (GTA-V), a popular video game, to synthesize large amounts of realistic training data. Our experiments show 
that SqueezeSeg achieves high accuracy with astonishingly fast and stable runtime (8.7 ms per frame), highly desirable for 
autonomous driving applications. Furthermore, additionally training on synthesized data boosts validation accuracy on real-world 
data. Our source code and synthesized data will be open-sourced.

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

* [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323),#4 for Real-time Semantic segmentation on 
Cityscapes(`mIOU:71.8 % ,Time:469ms`),#10 for Semantic segmentation for Cityscapes(`Mean IoU :71.8 % `) ,CVPR 2017   
作者：Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe   
机构：亚琛工业大学视觉计算研究所     
日期：2016-12-6   
摘要：Semantic image segmentation is an essential component of modern autonomous driving systems, as an accurate understanding 
of the surrounding scene is crucial to navigation and action planning. Current state-of-the-art approaches in semantic image 
segmentation rely on pre-trained networks that were initially developed for classifying images as a whole. While these networks 
exhibit outstanding recognition performance (i.e., what is visible?), they lack localization accuracy (i.e., where precisely is 
something located?). Therefore, additional processing steps have to be performed in order to obtain pixel-accurate segmentation 
masks at the full image resolution. To alleviate this problem we propose a novel ResNet-like architecture that exhibits strong 
localization and recognition performance. We combine multi-scale context with pixel-level accuracy by using two processing 
streams within our network: One stream carries information at the full image resolution, enabling precise adherence to 
segment boundaries. The other stream undergoes a sequence of pooling operations to obtain robust features for recognition. 
The two streams are coupled at the full image resolution using residuals. Without additional processing steps and without 
pre-training, our approach achieves an intersection-over-union score of 71.8% on the Cityscapes dataset.   
  
 


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
代码：[xinleipan/VirtualtoReal-RL](https://github.com/xinleipan/VirtualtoReal-RL) :star: 979        
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

 







