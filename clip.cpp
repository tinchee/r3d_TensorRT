#include "clip.h"

const int perClip = 3 * 32 * 112 * 112;
const int perChannel = 32 * 112 * 112;
const int perPictures = 112 * 112;

void scaleAndCropFrames(const cv::Mat frame, float *data, const int i, const int j)
{

    cv::Mat newframe(128, 171, CV_8UC3);
    cv::resize(frame, newframe, newframe.size());
    cv::Mat crop = newframe(cv::Rect(30, 8, 112, 112));
    for (int k = 0; k < perPictures; k++)
    {
        data[i * perClip + 0 * perChannel + j * perPictures + k] = crop.at<cv::Vec3b>(k)[2] / 255.0;
        data[i * perClip + 1 * perChannel + j * perPictures + k] = crop.at<cv::Vec3b>(k)[1] / 255.0;
        data[i * perClip + 2 * perChannel + j * perPictures + k] = crop.at<cv::Vec3b>(k)[0] / 255.0;
    }
}
float *clipVideo(const std::string &filePath, int &clipNum)
{

    cv::VideoCapture inputVideo(filePath);
    if (!inputVideo.isOpened())
    {
        std::cout << "Could not open the input video: " << filePath << std::endl;
        return nullptr;
    }

    int framesNum = inputVideo.get(CV_CAP_PROP_FRAME_COUNT);
    int framesPerClip = 32;
    clipNum = framesNum / framesPerClip;
    int framesPerBlock = framesNum / clipNum;
    float *data = new float[clipNum * perClip];
    if (data == nullptr)
    {
        std::cout << "new memory fail!" << std::endl;
        return nullptr;
    }

    int count = 0;
    cv::Mat frame;
    while (count < framesNum)
    {
        if (!inputVideo.read(frame))
            break;

        if ((count % framesPerBlock) < framesPerClip && (count / framesPerBlock) < clipNum)
        {
            scaleAndCropFrames(frame, data, count / framesPerBlock, count % framesPerBlock);
            ++count;
        }
        else
        {
            ++count;
            continue;
        }
    }
    return data;
}
