#include <RandomForest.h>
#include <algorithm> // For std::shuffle
#include <iostream>  // For std::cout, std::endl
#include <random>    // For std::mt19937, std::random_device
#include <vector>    // For std::vector
#include <iterator>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

RandomForest::RandomForest(/* args */)
{
}

cv::Ptr<RandomForest> RandomForest::create(int numberOfClasses,
                                           int numberOfDTrees,
                                           Size winSize)
{
    cv::Ptr<RandomForest> randomForest = new RandomForest();
    randomForest->m_numberOfClasses = numberOfClasses;
    randomForest->m_numberOfDTrees = numberOfDTrees;
    randomForest->m_winSize = winSize;
    randomForest->m_models.reserve(numberOfDTrees);
    randomForest->m_hog = randomForest->createHogDescriptor();
    long unsigned int timestamp = static_cast<long unsigned int>(time(0));
    cout << timestamp << endl;
    randomForest->m_randomGenerator = std::mt19937(timestamp);
    return randomForest;
}

vector<int> RandomForest::getRandomUniqueIndices(int start, int end, int numOfSamples)
{
    std::vector<int> indices;
    indices.reserve(end - start);
    for (size_t i = start; i < end; i++)
        indices.push_back(i);

    std::shuffle(indices.begin(), indices.end(), m_randomGenerator);
    // copy(indices.begin(), indices.begin() + numOfSamples, std::ostream_iterator<int>(std::cout, ", "));
    // cout << endl;
    return std::vector<int>(indices.begin(), indices.begin() + numOfSamples);
}

vector<pair<int, cv::Mat>> RandomForest::generateTrainingImagesLabelSubsetVector(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                                                                 float subsetPercentage,
                                                                                 bool undersampling)
{

    vector<pair<int, cv::Mat>> trainingImagesLabelSubsetVector;

    // Compute minimum number of samples a class label has.
    int minimumSample = trainingImagesLabelVector.size(); // A high enough value

    if (undersampling)
    {
        int minimumClassSamples[m_numberOfClasses];
        for (size_t i = 0; i < m_numberOfClasses; i++)
            minimumClassSamples[i] = 0;
        for (auto &&trainingSample : trainingImagesLabelVector)
            minimumClassSamples[trainingSample.first]++;
        for (size_t i = 1; i < m_numberOfClasses; i++)
            if (minimumClassSamples[i] < minimumSample)
                minimumSample = minimumClassSamples[i];
    }

    int count[m_numberOfClasses];
    for (size_t label = 0; label < m_numberOfClasses; label++)
    {
        count[label] = 0;

        // Create a subset vector for all the samples with class label.
        vector<pair<int, cv::Mat>> temp;
        temp.reserve(100);
        for (auto &&sample : trainingImagesLabelVector)
            if (sample.first == label)
                temp.push_back(sample);

        // Compute how many samples to choose for each label for random subset.
        int numOfElements;
        if (undersampling)
        {
            numOfElements = (subsetPercentage * minimumSample) / 100;
            trainingImagesLabelSubsetVector.reserve(numOfElements * 6);
        }
        else
        {
            numOfElements = (temp.size() * subsetPercentage) / 100;
        }

        // Filter numOfElements elements from temp and append to trainingImagesLabelSubsetVector
        vector<int> randomUniqueIndices = getRandomUniqueIndices(0, temp.size(), numOfElements);
        for (size_t j = 0; j < randomUniqueIndices.size(); j++)
        {
            trainingImagesLabelSubsetVector.push_back(temp.at(randomUniqueIndices.at(j)));
            count[temp.at(randomUniqueIndices.at(j)).first]++;
        }
    }

    return trainingImagesLabelSubsetVector;
}

void RandomForest::train(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                         float subsetPercentage,
                         Size winStride,
                         Size padding)
{
    bool undersampling = true;
    // Train each decision tree
    for (size_t i = 0; i < m_numberOfDTrees; i++)
    {
        // cout << "Training decision tree: " << i+1 << " of " << m_numberOfDTrees << ".\n";
        vector<pair<int, cv::Mat>> trainingImagesLabelSubsetVector = generateTrainingImagesLabelSubsetVector(trainingImagesLabelVector,
                                                                                                             subsetPercentage,
                                                                                                             undersampling);
        cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingImagesLabelSubsetVector,
                                                          winStride,
                                                          padding);
        m_models.push_back(model);
    }
}

Prediction RandomForest::predict(cv::Mat testImage,
                                 Size winStride,
                                 Size padding)
{
    cv::Mat resizedInputImage = resizeToBoundingBox(testImage);

    // Compute Hog only of center crop of grayscale image
    vector<float> descriptors;
    vector<Point> foundLocations;
    vector<double> weights;
    m_hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

    // Store the features and labels for model training.
    // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
    // if(testImagesLabelVector.at(i).first == randomForest.at(0)->predict(cv::Mat(descriptors)))
    //     accuracy += 1;
    std::map<int, int> labelCounts;
    int maxCountLabel = -1;
    for (auto &&model : m_models)
    {
        int label = model->predict(cv::Mat(descriptors));
        if (labelCounts.count(label) > 0)
            labelCounts[label]++;
        else
            labelCounts[label] = 1;

        if (maxCountLabel == -1)
            maxCountLabel = label;
        else if (labelCounts[label] > labelCounts[maxCountLabel])
            maxCountLabel = label;
    }

    return Prediction{.label = maxCountLabel, .confidence = (labelCounts[maxCountLabel] * 1.0f) / m_numberOfDTrees};
}

cv::Ptr<cv::ml::DTrees> RandomForest::trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                                        Size winStride,
                                                        Size padding)
{
    // Create the model
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    // See https://docs.opencv.org/3.0-beta/modules/ml/doc/decision_trees.html#dtrees-params
    model->setCVFolds(0);        // set num cross validation folds - Not implemented in OpenCV
    model->setMaxCategories(10); // set max number of categories
    model->setMaxDepth(20);      // set max tree depth
    model->setMinSampleCount(2); // set min sample count
    // ToDo - Tweak this
    // cout << "Number of cross validation folds are: " << model->getCVFolds() << endl;
    // cout << "Max Categories are: " << model->getMaxCategories() << endl;
    // cout << "Max depth is: " << model->getMaxDepth() << endl;
    // cout << "Minimum Sample Count: " << model->getMinSampleCount() << endl;

    // Compute Hog Features for all the training images
    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;
        m_hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << "=====================================" << endl;
        // cout << "Number of descriptors are: " << descriptors.size() << endl;
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        // cout << "New size of training features" << feats.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
        // cout << "New size of training labels" << labels.size() << endl;
    }

    cv::Ptr<cv::ml::TrainData> trainData = ml::TrainData::create(feats, ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

HOGDescriptor RandomForest::createHogDescriptor()
{
    // Create Hog Descriptor
    Size blockSize(16, 16);
    Size blockStride(8, 8);
    Size cellSize(8, 8);
    int nbins(18);
    int derivAperture(1);
    double winSigma(-1);
    int histogramNormType(HOGDescriptor::L2Hys);
    double L2HysThreshold(0.2);
    bool gammaCorrection(true);
    float free_coef(-1.f);
    //! Maximum number of detection window increases. Default value is 64
    int nlevels(HOGDescriptor::DEFAULT_NLEVELS);
    //! Indicates signed gradient will be used or not
    bool signedGradient(true);
    HOGDescriptor hog(m_winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, signedGradient);
    return hog;
}

cv::Mat RandomForest::resizeToBoundingBox(cv::Mat &inputImage)
{
    cv::Mat resizedInputImage;
    if (inputImage.rows < m_winSize.height || inputImage.cols < m_winSize.width)
    {
        float scaleFactor = fmax((m_winSize.height * 1.0f) / inputImage.rows, (m_winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    Rect r = Rect((resizedInputImage.cols - m_winSize.width) / 2, (resizedInputImage.rows - m_winSize.height) / 2,
                  m_winSize.width, m_winSize.height);
    // cv::imshow("Resized", resizedInputImage(r));
    // cv::imshow("Original", inputImage);
    // cv::waitKey(0);
    return resizedInputImage(r);
}