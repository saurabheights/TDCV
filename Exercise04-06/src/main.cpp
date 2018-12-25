#include <opencv2/opencv.hpp>
#include <HogVisualization.h>
#include <iomanip>
#include <sstream>
using namespace cv;
using namespace std;

HOGDescriptor createHogDescriptor(Size winSize)
{
    // Create Hog Descriptor
    Size blockSize(16, 16);
    Size blockStride(8, 8);
    Size cellSize(8, 8);
    int nbins(9);
    int derivAperture(1);
    double winSigma(-1);
    int histogramNormType(HOGDescriptor::L2Hys);
    double L2HysThreshold(0.2);
    bool gammaCorrection(true);
    float free_coef(-1.f);
    //! Maximum number of detection window increases. Default value is 64
    int nlevels(HOGDescriptor::DEFAULT_NLEVELS);
    //! Indicates signed gradient will be used or not
    bool signedGradient(false);
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, signedGradient);
    return hog;
}

cv::Mat resizeToBoundingBox(cv::Mat& inputImage, Size& winSize)
{
    cv::Mat resizedInputImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    Rect r = Rect((resizedInputImage.cols - winSize.width) / 2, (resizedInputImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);
    return resizedInputImage(r);
}

void task1()
{
    // Read image and display
    string imagePath = string(PROJ_DIR) + "/data/task1/obj1000.jpg";
    cout << imagePath << endl;
    Mat inputImage = imread(imagePath, cv::IMREAD_UNCHANGED);
    imshow("task1 - Input Image", inputImage);
    cv::waitKey(200);

    // Resize image if very small while maintaining aspect ratio till its bigger than winSize
    Size winSize(128, 128);
    cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

    HOGDescriptor hog = createHogDescriptor(winSize);

    cv::Mat grayImage;
    cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Compute Hog only of center crop of grayscale image
    vector<float> descriptors;
    vector<Point> foundLocations;
    vector<double> weights;
    Size winStride(8, 8);
    Size padding(0, 0);
    hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);
    visualizeHOG(resizedInputImage, descriptors, hog, 6);
}

vector<vector<pair<int, cv::Mat>>> loadTask2Dataset()
{
    vector<pair<int, cv::Mat>> labelImagesTrain;
    vector<pair<int, cv::Mat>> labelImagesTest;
    labelImagesTrain.reserve(49 + 67 + 42 + 53 + 67 + 110);
    labelImagesTest.reserve(60);
    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << string(PROJ_DIR) << "/data/task2/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            // cout << imagePathStr << endl;
            pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;    
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTrain.push_back(labelImagesTrainPair);
        }


        for (size_t j = 0; j < numberOfTestImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << string(PROJ_DIR) << "/data/task2/test/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j + numberOfTrainImages[i] << ".jpg";
            string imagePathStr = imagePath.str();
            // cout << imagePathStr << endl;
            pair<int, cv::Mat> labelImagesTestPair;
            labelImagesTestPair.first = i;
            labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTest.push_back(labelImagesTestPair);
        }
    }

    vector<vector<pair<int, cv::Mat>>> fullDataset;
    fullDataset.push_back(labelImagesTrain);
    fullDataset.push_back(labelImagesTest);
    return fullDataset;
}

void task2()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Dataset();

    // Create the model
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    model->setCVFolds(0);  // set num cross validation folds - Not implemented in OpenCV
    // model->setMaxCategories();  // set max number of categories
    model->setMaxDepth(6);  // set max tree depth
    model->setMinSampleCount(2);  // set min sample count
    cout << "Number of cross validation folds are: " << model->getCVFolds() << endl;
    cout << "Max Categories are: " << model->getMaxCategories() << endl;
    cout << "Max depth is: " << model->getMaxDepth() << endl;
    cout << "Minimum Sample Count: " << model->getMinSampleCount() << endl;

    // Compute Hog Features for all the training images
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);
    Size winSize(128, 128);
    HOGDescriptor hog = createHogDescriptor(winSize);
    Size winStride(8, 8);
    Size padding(0, 0);
    
    cv::Mat feats, labels;
    for(size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << "=====================================" << endl;
        // cout << "Number of descriptors are: " << descriptors.size() << endl;
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        // cout << "New size of training features" << feats.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
        // cout << "New size of training labels" << labels.size() << endl;
    }

    // cout << "Features Rows is: " << feats.rows << endl;
    // cout << "Labels Size is: " << labels.size() << endl;

    // Train model
    cv::Ptr<cv::ml::TrainData> trainData = ml::TrainData::create(feats, ml::ROW_SAMPLE, labels);
    model->train(trainData);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    float accuracy = 0;
    for(size_t i = 0; i < testImagesLabelVector.size(); i++) {
        cv::Mat inputImage = testImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
        if(testImagesLabelVector.at(i).first == model->predict(cv::Mat(descriptors)))
            accuracy += 1;
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Single Decision Tree Accuracy is: [" << accuracy/testImagesLabelVector.size() <<"]." << endl;
    cout << "==================================================" << endl;
}

int main()
{
    // Task 1
    task1();
    // waitKey(0);
    destroyAllWindows();

    // Task 2
    task2();
    waitKey(0);
    destroyAllWindows();

    // Task 3

    waitKey(0);
    destroyAllWindows();
    return 0;
}