# SceneSegmentation
This project is a part of the Pawsey Summer Internship where I will do test multiple semantic segmentation algorithms and models on their training and inference time. To do this TensorFlow 2.0 (TF 2.0) will be used benchmark them against one another primarily on the Cityscapes Dataset. The main goal is to asceertain the usefulness of these models for Critical Infrastructure like railways, buildings roads etc. Their inference times will likely be assessed on a smaller machine like the NVIDIA Jetson AGX to assess their real-time practical use. This makes Cityscapes a perfect dataset to benchmark on. Reimplementing certain models will be easier than others, while some will be difficult as they were made in PyTorch or much older versions of TF.

Furthermore, with the rise of Panoptic Segmentation in 2019 if time allows, some investigation of Panoptic Segmentation models will also be attempted to assess their real-time practical use.

## Transfer Learning
Some models require the use of ImageNet or PASCAL VOC 2012 pretrained models to initialize their weights

## Requirements for Project
* TensorFlow 2.0 (To enable use of TensorRT and the latest Keras library and smoother integration)
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores (Project will use different GPUs)
* Python >= 3.7

## Semantic Segmentation
The models that will be trained and tested against will be:
### U-Net 
Following this github repo here: https://github.com/zhixuhao/unet <br>
Which was inspired by the paper found here:https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ <br>
I will reimplement U-Net in TF 2.0 which will be easier than others due to its use of TF and Keras already, it's just going to be a matter of using the same functions in TF 2.0. Looking at the repository, no pretrained weights were used in the calling of the unet function. Some pretrained weights were used for the github, but this model isn't supplied in the github repo, as such will likely train from scratch. The model was chosen due to it being an old model that won the 2015 ISIBI Challenge, it's also the most common example of Semantic Segmentation due to its small size compared to its modern day ResNet competitors.
### Bayesian SegNet
Following this github repo here: https://github.com/toimcio/SegNet-tensorflow <br>
Which was inspired by the paper found here: https://arxiv.org/abs/1511.02680 <br>
Reimplementing this in TF2.0 should be ok, the author has done it in pure TF, redoing it in Keras will likely make it easier to understand. This model was chosen due to its use of VGG16 trained on ImageNet, this model was also the last hurrah of the standard Encoder-Decoder architecture which essentially mirrored the VGG16 model topping benchmarks in 2017. Trying this on the Cityscapes dataset and comparing it to newer models should show har far we have come.
### DeepLabV3+
Following this github repo here: https://github.com/rishizek/tensorflow-deeplab-v3-plus <br>
Which was inspired by this paper found here: https://arxiv.org/abs/1802.02611 <br>
One of the many models which used atrous convolution on a ResNet-101 pre-trained from ImageNet, it beat the previous DeepLabV3 and PSPNet which employed very similar architectures for semantic segmentation. It was chosen as it seems like a good representation of this type of model. Using this on Cityscapes will be more like reporducing the results from the original paper. This will show if TF 2.0 and the use of TensorRT makes any difference to the accuracy of the model (it shouldn't really, just make training faster). Overall this will add to the validity of this benchmark.
### Fast-SCNN
Following this github repo here: https://github.com/kshitizrimal/Fast-SCNN <br>
Which was inspired by this paper found here: https://arxiv.org/abs/1902.04502 <br>
First this one is already using TF 2.0 and Keras, so my job here is making the Cityscapes work with it, which shouldn't be an issue. It's one of the most recent semantic segmentation articles being from 2019. It was designed to has very fast inference and training times. In fact it can run at 120+ fps when using 1024x2048 images which is *impressive*. Also, it was designed in mind that embedded systems would use this model for inference. It's done this using the power of Depthwise Convolution and very efficient pyramid modules. While its accuracy isn't as high as others, it's able to achieve 68% on Cityscapes while infering in *real time*. Ultimately will be interesting to test against others when not in real time as well.
### Fast-FCN
Following this github repo here: https://github.com/wuhuikai/FastFCN <br>
Which was inspired by this paper found here: http://wuhuikai.me/FastFCNProject/fast_fcn.pdf
This architecture is similar to DeepLabV3+ however it uses Joint Pyramid Upsampling (JPU) for its decoding of images which is more efficient in its upsampling compared to atrous convolution. It also showed that using the JPU had no real performance loss alongside other reductions in computational complexity make it another great contender for using on embedded systems like the Jetson AGX. The ResNet-50 and ResNet-101 are used as the backbones here, meaning the weights will start from a pretrained ImageNet ResNet as most do in 2019. 

## Panoptic Segmentation
This will be explored if there is time, Panoptic Segmentation is the idea of combining semantic and instance segmentation together to produce an output with a class and an instance ID for some object. This seems like a novel idea, and in theory will be more useful than semantic segmentation alone especially for automotive applications and critical infrastructure projects. In fact, some of the latest Panotpic Segmentation models actually top the charts for semantic segmentation benchmarks also, such as the Panoptic-DeepLab, so looking into this would be very interesting.

## Datasets
* Cityscapes - will require an account. https://www.cityscapes-dataset.com/downloads/
