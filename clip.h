#ifndef CLIP
#define CLIPfloat
#include <opencv2/opencv.hpp>

// every video frame will be scaled to the size of 128*171
// then center crops of the scaled frame
void scaleAndCropFrames(const cv::Mat frame, float *data, const int i, const int j);

float *clipVideo(const std::string &filePath, int &clipNUm);

#endif