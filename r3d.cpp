#include <NvInfer.h>
#include <cstdio>

#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "clip.h"
#include "logger.h"

using namespace nvinfer1;
const double eps = 1e-5;
#undef ASSERT
#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            gLogError << "Assertion failure: " << #condition << std::endl;                                             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            gLogError << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

class r3d
{
  public:
    r3d() : mEngine(nullptr)
    {
    }
    bool build();
    bool infer(const char *trtModel, const int size, const std::string &filePath);

  private:
    ICudaEngine *mEngine;
    std::map<std::string, Weights> mWeightMap;
    bool loadWeights(const std::string &filePath);
    bool constructNetwork(IBuilder *builder, INetworkDefinition *network, IBuilderConfig *config);
};
bool r3d::loadWeights(const std::string &filePath)
{
    std::ifstream input(filePath, std::ios::binary);
    int32_t count=0;
    input >> count;
    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        std::string name;
        uint32_t size;
        input >> name >> std::dec >> size;
        uint32_t *mem = new uint32_t[size];
        for (uint32_t i = 0; i < size; i++)
            input >> std::hex >> mem[i];
        wt.values = mem;
        wt.count = size;
        mWeightMap[name] = wt;
    }
    return true;
}

IScaleLayer *addBatchNorm3d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                            const std::string &name)
{
    // std::cout<<name<<"****"<<std::endl;
    float *gamma = (float *)weightMap[name + ".weight"].values;
    float *beta = (float *)weightMap[name + ".bias"].values;
    float *mean = (float *)weightMap[name + ".running_mean"].values;
    float *var = (float *)weightMap[name + ".running_var"].values;
    int len = weightMap[name + ".running_var"].count;

    float *scval = new float[len];
    for (int i = 0; i < len; i++)
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = new float[len];
    for (int i = 0; i < len; i++)
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = new float[len];
    for (int i = 0; i < len; i++)
        pval[i] = 1.0;
    Weights power{DataType::kFLOAT, pval, len};

    IScaleLayer *batchN = network->addScaleNd(input, ScaleMode::kCHANNEL, shift, scale, power, 0);
    return batchN;
}
IActivationLayer *addBasicBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                                const std::string &name, const int outChannel, const int kernelSize, int stride,
                                const int padding, bool isStrideHalf, bool haveDown)
{
    Weights empty{DataType::kFLOAT, nullptr, 0};
    // std::cout << name << std::endl;
    IConvolutionLayer *conv10 = network->addConvolutionNd(input, outChannel, Dims3{kernelSize, kernelSize, kernelSize},
                                                          weightMap[name + ".conv1.0.weight"], empty);
    ASSERT(conv10);
    conv10->setPaddingNd(Dims3{padding, padding, padding});
    conv10->setStrideNd(Dims3{stride, stride, stride});
    IScaleLayer *conv11 = addBatchNorm3d(network, weightMap, *conv10->getOutput(0), name + ".conv1.1");
    ASSERT(conv11);
    IActivationLayer *conv12 = network->addActivation(*conv11->getOutput(0), ActivationType::kRELU);
    ASSERT(conv12);

    if (isStrideHalf)
        stride = stride / 2;
    IConvolutionLayer *conv20 =
        network->addConvolutionNd(*conv12->getOutput(0), outChannel, Dims3{kernelSize, kernelSize, kernelSize},
                                  weightMap[name + ".conv2.0.weight"], empty);
    ASSERT(conv20);
    conv20->setStrideNd(Dims3{padding, padding, padding});
    conv20->setPaddingNd(Dims3{stride, stride, stride});
    IScaleLayer *conv21 = addBatchNorm3d(network, weightMap, *conv20->getOutput(0), name + ".conv2.1");
    ASSERT(conv21);

    IElementWiseLayer *ele = nullptr;
    if (!haveDown)
        ele = network->addElementWise(input, *conv21->getOutput(0), ElementWiseOperation::kSUM);
    else
    {
        IConvolutionLayer *downConv = network->addConvolutionNd(input, outChannel, Dims3{1, 1, 1},
                                                                weightMap[name + ".downsample.0.weight"], empty);
        downConv->setStrideNd(Dims3{2, 2, 2});
        IScaleLayer *downScale = addBatchNorm3d(network, weightMap, *downConv->getOutput(0), name + ".downsample.1");
        ele = network->addElementWise(*conv21->getOutput(0), *downScale->getOutput(0), ElementWiseOperation::kSUM);
    }
    ASSERT(ele);
    IActivationLayer *relu = network->addActivation(*ele->getOutput(0), ActivationType::kRELU);
    ASSERT(relu);

    return relu;
}
bool r3d::constructNetwork(IBuilder *builder, INetworkDefinition *network, IBuilderConfig *config)
{
    Weights emptyWeight;
    emptyWeight.count = 0;
    emptyWeight.type = DataType::kFLOAT;
    emptyWeight.values = nullptr;

    ITensor *inputData = network->addInput("input", DataType::kFLOAT, Dims4{3, 32, 112, 112});
    ASSERT(inputData);

    // stem
    IConvolutionLayer *stemConv =
        network->addConvolutionNd(*inputData, 64, Dims3{3, 7, 7}, mWeightMap["stem.0.weight"], emptyWeight);
    stemConv->setPaddingNd(Dims3{1, 3, 3});
    stemConv->setStrideNd(Dims3{1, 2, 2});
    ASSERT(stemConv);
    IScaleLayer *stemBatch = addBatchNorm3d(network, mWeightMap, *stemConv->getOutput(0), "stem.1");
    ASSERT(stemBatch);
    IActivationLayer *stemrelu = network->addActivation(*stemBatch->getOutput(0), ActivationType::kRELU);
    ASSERT(stemrelu);

    // std::cout << "stemrelu done" << std::endl;

    // layer1
    IActivationLayer *layer10 =
        addBasicBlock(network, mWeightMap, *stemrelu->getOutput(0), "layer1.0", 64, 3, 1, 1, false, false);
    ASSERT(layer10);
    IActivationLayer *layer11 =
        addBasicBlock(network, mWeightMap, *layer10->getOutput(0), "layer1.1", 64, 3, 1, 1, false, false);
    ASSERT(layer11);

    // layer2
    IActivationLayer *layer20 =
        addBasicBlock(network, mWeightMap, *layer11->getOutput(0), "layer2.0", 128, 3, 2, 1, true, true);
    ASSERT(layer20);
    IActivationLayer *layer21 =
        addBasicBlock(network, mWeightMap, *layer20->getOutput(0), "layer2.1", 128, 3, 1, 1, false, false);
    ASSERT(layer21);

    // layer3
    IActivationLayer *layer30 =
        addBasicBlock(network, mWeightMap, *layer21->getOutput(0), "layer3.0", 256, 3, 2, 1, true, true);
    ASSERT(layer30);
    IActivationLayer *layer31 =
        addBasicBlock(network, mWeightMap, *layer30->getOutput(0), "layer3.1", 256, 3, 1, 1, false, false);
    ASSERT(layer31);

    // layer4
    IActivationLayer *layer40 =
        addBasicBlock(network, mWeightMap, *layer31->getOutput(0), "layer4.0", 512, 3, 2, 1, true, true);
    ASSERT(layer40);
    IActivationLayer *layer41 =
        addBasicBlock(network, mWeightMap, *layer40->getOutput(0), "layer4.1", 512, 3, 1, 1, false, false);
    ASSERT(layer41);

    // avgpool
    IPoolingLayer *pool = network->addPoolingNd(*layer41->getOutput(0), PoolingType::kAVERAGE, Dims3{4, 7, 7});
    ASSERT(pool);

    IShuffleLayer *shuff = network->addShuffle(*pool->getOutput(0));
    shuff->setReshapeDimensions(Dims3{1, 1, 512});
    ASSERT(shuff);

    // fc
    IFullyConnectedLayer *fc =
        network->addFullyConnected(*shuff->getOutput(0), 400, mWeightMap["fc.weight"], mWeightMap["fc.bias"]);
    ASSERT(fc);

    // softmax
    ISoftMaxLayer *soft = network->addSoftMax(*fc->getOutput(0));
    ASSERT(soft);

    soft->getOutput(0)->setName("output");
    network->markOutput(*soft->getOutput(0));
    builder->setMaxBatchSize(20);
    config->setMaxWorkspaceSize(16 * (1 << 20));

    IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan)
        return false;

    std::ofstream planWriter("r3d.engine", std::ios::binary);
    if (!planWriter)
    {
        std::cout << "file open error" << std::endl;
        return false;
    }
    planWriter.write(reinterpret_cast<const char *>(plan->data()), plan->size());

    delete plan;
    return true;
}
bool r3d::build()
{
    bool isLoad = loadWeights("../r3d.wts");
    if (!isLoad)
    {
        std::cout << "Load Error!" << std::endl;
        return false;
    }
    IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
    if (!builder)
        return false;
    INetworkDefinition *network = builder->createNetworkV2(0);
    if (!network)
        return false;
    IBuilderConfig *config = builder->createBuilderConfig();
    if (!config)
        return false;
    bool constructed = constructNetwork(builder, network, config);
    if (!constructed)
        return false;

    delete builder;
    delete network;
    delete config;

    return true;
}

std::string dealOut(float *data, const int outputSize, const int clipNum)
{
    int label[400] = {0};

    for (int i = 0; i < clipNum; i++)
    {
        double maxx = -1;
        int index = -1;
        for (int j = 0; j < 400; j++)
        {
            if (data[i * 400 + j] > maxx)
                maxx = data[i * 400 + j], index = j;
        }

        label[index]++;
    }

    int ans = 0;
    int maxx = -1;
    for (int i = 0; i < 400; i++)
    {
        if (label[i] > maxx)
            maxx = label[i], ans = i;
    }
    std::cout << "the class probabilities / each clip:" << std::endl;
    for (int i = 0; i < clipNum; i++)
        std::cout << data[ans + 400 * i] * 100 << "% ";
    std::cout << std::endl;

    std::ifstream labelInput("../label.txt", std::ios::binary);
    std::string labelTxt;
    std::vector<std::string> labelName;
    while (getline(labelInput, labelTxt))
        labelName.push_back(labelTxt);
    labelInput.close();

    return labelName[ans];
}
bool r3d::infer(const char *trtModel, const int size, const std::string &filePath)
{

    IRuntime *runtime = createInferRuntime(gLogger.getTRTLogger());
    mEngine = runtime->deserializeCudaEngine(trtModel, size);

    IExecutionContext *context = mEngine->createExecutionContext();

    int clipNum = 0;
    float *input = clipVideo(filePath, clipNum);
    const int inputSize = clipNum * 3 * 32 * 112 * 112;

    const int outputSize = clipNum * 400;
    float *ouput = new float[outputSize];

    void *buffers[2];
    const int inputIndex = mEngine->getBindingIndex("input");
    const int outputIndex = mEngine->getBindingIndex("output");
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float)));

    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    context->execute(clipNum, buffers);
    CHECK(cudaMemcpy(ouput, buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    std::string label = dealOut(ouput, outputSize, clipNum);
    std::cout << "label of the video is \" " << label << " \"" << std::endl;

    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    delete runtime;
    delete context;
    return true;
}

int main(int argc, char **argv)
{
    r3d _r3d;
    if (argc == 2 && std::string(argv[1]) == "-g")
    {
        if (!_r3d.build())
            std::cout << "build error" << std::endl;
    }
    else if (argc == 3 && std::string(argv[1]) == "-r")
    {
        std::string filePath = std::string(argv[2]);
        std::ifstream file("r3d.engine", std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size_t size = file.tellg();
            file.seekg(0, file.beg);
            char *trtModel = new char[size];
            ASSERT(trtModel);
            file.read(trtModel, size);
            file.close();
            if (!_r3d.infer(trtModel, size, filePath))
                std::cout << "infer error" << std::endl;
            delete[] trtModel;
        }
    }
    else
    {
        std::cout << "Parameter Error!" << std::endl;
    }
    return 0;
}
