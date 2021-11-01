# r3d_TensorRT
An [r3d network](https://arxiv.org/abs/1711.11248) implemented with TensorRT8.x, The weight of the model comes from PyTorch. A description of the models in Pytroch can be found [here](https://pytorch.org/vision/stable/models.html#video-classification)

## Enviroments

Make sure the following tools are installed before running:

```
cmake
opencv 3.4
TensorRT 8.x
```

## How to run

First you need to export the .wts file from the PyTorch weight .pth file. We use torchvision to do it.

```
git clone https://github.com/tinchee/r3d_TensorRT.git
python gen_wts.py
```

Then compile the project file that contains the tensorRT code.

```
mkdir build
cd build
cmake ..
make
```

Models are usually saved before inferencing. ```./r3d -g```  saves the network structure and ```./r3d -r [video file path]```inferences using the saved network structure, for example:

```
./r3d -g
./r3d -r ../video/applying\ cream/OEGarMnMljQ_000007_000017.mp4
```

## Docker

Using Docker is also a good choice. Make sure you have successfully installed the docker and nvidia-docker. Once the environment is ready, you can using following commands to download and boot the docker image:

```
docker pull cheer7/r3d_tensorrt:latest
docker run -it --gpus all cheer7/r3d_tensorrt:latest
```

The same content in the Docker, the file path is ```/root/r3d_TensorRT```. It doesn't need to be compiled and can be used directly:

```
cd
cd r3d_TensorRT/build/
./r3d -r ../video/applying\ cream/OEGarMnMljQ_000007_000017.mp4
```



## More Information

r3d does not have the last softmax function in pyTorch's implementation version. To facilitate classification, the last layer of softmax function is added in tensorRT's implementation, which is also the operation in the experimental part of the paper.

As for video segmentation, every 32 frames is regarded as one clip, and there is no interval between each clip, that is, the number of clips is ```framesOfVideo/32```. All the clips of a video are made up of one input(N * 3 * 32 * 112 * 112, N represents the number of clips). Each clip predicts a label separately, and the label that appears the most times in all clips becomes the prediction label of the video. The following information appear at the end of the inferencing.

```
the class probabilities / each clip:
16.7224% 10.2105% 12.0446% 7.01359% 37.3826% 6.69616% 4.17084% 
label of the video is " applying cream "
```

The meaning of the first two lines of output is the predicted probability of the video tag in each clip, and the last line represents the video tag.

If you run ```./r3d -r ../video/laying\ bricks/N3byaY-0E3E_000042_000052.mp4```, you will found the output is different from the true label, It may be that the prediction result of r3d is a wrong answer :). I put the segmented video data into tensorRT and pytroch, they can get the same data after softmax function...
