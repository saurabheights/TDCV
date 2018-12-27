#include <opencv2/opencv.hpp>
#include <HogVisualization.h>
#include <RandomForest.h>
#include <iomanip>
#include <sstream>
#include <random>
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

cv::Mat resizeToBoundingBox(cv::Mat &inputImage, Size &winSize)
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

cv::Ptr<cv::ml::DTrees> trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector)
{
    // Create the model
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    model->setCVFolds(0); // set num cross validation folds - Not implemented in OpenCV
    // model->setMaxCategories();  // set max number of categories
    model->setMaxDepth(6);       // set max tree depth
    model->setMinSampleCount(2); // set min sample count
    cout << "Number of cross validation folds are: " << model->getCVFolds() << endl;
    cout << "Max Categories are: " << model->getMaxCategories() << endl;
    cout << "Max depth is: " << model->getMaxDepth() << endl;
    cout << "Minimum Sample Count: " << model->getMinSampleCount() << endl;

    // Compute Hog Features for all the training images
    Size winSize(128, 128);
    HOGDescriptor hog = createHogDescriptor(winSize);
    Size winStride(8, 8);
    Size padding(0, 0);

    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
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

    cv::Ptr<cv::ml::TrainData> trainData = ml::TrainData::create(feats, ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

void task2Basic()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Dataset();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Train model
    cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingImagesLabelVector);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    float accuracy = 0;
    Size winSize(128, 128);
    HOGDescriptor hog = createHogDescriptor(winSize);
    Size winStride(8, 8);
    Size padding(0, 0);

    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = testImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
        if (testImagesLabelVector.at(i).first == model->predict(cv::Mat(descriptors)))
            accuracy += 1;
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Single Decision Tree Accuracy is: [" << accuracy / testImagesLabelVector.size() << "]." << endl;
    cout << "==================================================" << endl;
}

void task2()
{
    cout << "Task 2 started.\n";
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Dataset();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Create model
    int numberOfClasses = 6;
    int numberOfDTrees = 40;
    Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    Size winStride(8, 8);
    Size padding(0, 0);
    float subsetPercentage = 50.0f;
    bool undersampling = true;
    bool augment = false;
    randomForest->train(trainingImagesLabelVector, subsetPercentage, winStride, padding, undersampling, augment);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    float accuracy = 0;
    float accuracyPerClass[6] = {0};
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        Prediction prediction = randomForest->predict(testImage, winStride, padding);
        if (testImagesLabelVector.at(i).first == prediction.label)
        {
            accuracy += 1;
            accuracyPerClass[prediction.label] += 1;
        }
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Random Forest Accuracy is: [" << accuracy / testImagesLabelVector.size() << "]." << endl;

    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};
    for (size_t i = 0; i < numberOfClasses; i++)
    {
        cout << "Class " << i << " accuracy: [" << accuracyPerClass[i] / numberOfTestImages[i] << "]." << endl;
    }
    cout << "==================================================" << endl;
}

vector<vector<pair<int, cv::Mat>>> loadTask3Dataset()
{
    vector<pair<int, cv::Mat>> labelImagesTrain;
    vector<pair<int, cv::Mat>> labelImagesTest;
    labelImagesTrain.reserve(53 + 81 + 51 + 290);
    labelImagesTest.reserve(44);
    int numberOfTrainImages[6] = {53, 81, 51, 290};
    int numberOfTestImages[1] = {44};

    for (int i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << string(PROJ_DIR) << "/data/task3/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            // cout << imagePathStr << endl;
            pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTrain.push_back(labelImagesTrainPair);
        }
    }

    for (size_t j = 0; j < numberOfTestImages[0]; j++)
    {
        stringstream imagePath;
        imagePath << string(PROJ_DIR) << "/data/task3/test/" << setfill('0') << setw(4) << j << ".jpg";
        string imagePathStr = imagePath.str();
        // cout << imagePathStr << endl;
        pair<int, cv::Mat> labelImagesTestPair;
        labelImagesTestPair.first = -1; // These test images have no label
        labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        labelImagesTest.push_back(labelImagesTestPair);
    }

    vector<vector<pair<int, cv::Mat>>> fullDataset;
    fullDataset.push_back(labelImagesTrain);
    fullDataset.push_back(labelImagesTest);
    return fullDataset;
}

vector<vector<vector<int>>> getLabelAndBoundingBoxes()
{
    int numberOfTestImages = 44;
    vector<vector<vector<int>>> groundTruthBoundingBoxes;
    for (size_t j = 0; j < numberOfTestImages; j++)
    {
        stringstream gtFilePath;
        gtFilePath << string(PROJ_DIR) << "/data/task3/gt/" << setfill('0') << setw(4) << j << ".gt.txt";
        string gtFilePathStr = gtFilePath.str();

        fstream gtFile;
        gtFile.open(gtFilePathStr);
        if (!gtFile.is_open())
        {
            cout << "Failed to open file: " << gtFilePathStr << endl;
            exit(-1);
        }

        std::string line;
        vector<vector<int>> groundTruthBoundingBoxesPerImage;
        while (std::getline(gtFile, line))
        {
            std::istringstream in(line);
            vector<int> groundTruthLabelAndBoundingBox(5);
            int temp;
            for (size_t i = 0; i < 5; i++)
            {
                in >> temp;
                groundTruthLabelAndBoundingBox.at(i) = temp;
            }
            groundTruthBoundingBoxesPerImage.push_back(groundTruthLabelAndBoundingBox);
        }
        groundTruthBoundingBoxes.push_back(groundTruthBoundingBoxesPerImage);
    }
    return groundTruthBoundingBoxes;
}

void task3()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask3Dataset();
    // Load the ground truth bounding boxes values
    vector<vector<vector<int>>> labelAndBoundingBoxes = getLabelAndBoundingBoxes();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Create model
    int numberOfClasses = 4;
    int numberOfDTrees = 50;
    Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    Size winStride(8, 8);
    Size padding(0, 0);
    float subsetPercentage = 50.0f;
    bool undersampling = false;
    bool augment = true;
    randomForest->train(trainingImagesLabelVector, subsetPercentage, winStride, padding, undersampling, augment);

    // For each test image
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    cv::Scalar gtColors[4];
    gtColors[0] = cv::Scalar(255, 0, 0);
    gtColors[1] = cv::Scalar(0, 255, 0);
    gtColors[2] = cv::Scalar(0, 0, 255);
    gtColors[3] = cv::Scalar(255, 255, 0);
    int minSize = 1000, maxSize = -1; // To compute the variation of bounding box sizes in ground truth
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cout << "Test Image: " << i << endl;
        cv::Mat testImage = testImagesLabelVector.at(i).second;

        // Run testing on various bounding boxes of different scales
        // int minBoundingBoxSideLength = 70, maxBoundingBoxSideLength = 230;
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        vector<vector<int>> imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        cout << "Ground truth" << endl;
        cout << imageLabelsAndBoundingBoxes.size() << endl;
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            cout << imageLabelsAndBoundingBoxes.at(j).at(0) << " " << rect.x << " " << rect.y << " " << rect.height << " " << rect.width << endl;
            minBoundingBoxSideLength = min(minBoundingBoxSideLength, min(rect.width, rect.height));
            maxBoundingBoxSideLength = max(maxBoundingBoxSideLength, max(rect.width, rect.height));
        }
        minBoundingBoxSideLength -= 10;
        maxBoundingBoxSideLength += 10;

        int boundingBoxSideLength = minBoundingBoxSideLength;
        float scaleFactor = 1.20;
        int strideX = 2;
        int strideY = 2;
        vector<Prediction> predictionsVector; // Output of Hog Detection
        float NMS_CONFIDENCE_THRESHOLD = 0.6f;
        while (true)
        {
            cout << "Processing at bounding box side length: " << boundingBoxSideLength << '\n';
            
            // Sliding window with stride
            for (size_t row = 0; row < testImage.rows - boundingBoxSideLength; row += strideY)
            {
                for (size_t col = 0; col < testImage.cols - boundingBoxSideLength; col += strideX)
                {
                    cv::Rect rect(col, row, boundingBoxSideLength, boundingBoxSideLength);
                    cv::Mat rectImage = testImage(rect);

                    // Predict on subimage
                    Prediction prediction = randomForest->predict(rectImage, winStride, padding);
                    if (prediction.label != 3 && prediction.confidence > NMS_CONFIDENCE_THRESHOLD) // Ignore Background class.
                    {
                        prediction.bbox = rect;
                        predictionsVector.push_back(prediction);
                    }
                }
            }

            if (boundingBoxSideLength == maxBoundingBoxSideLength) // Maximum Bounding Box Size from ground truth
                break;
            boundingBoxSideLength = (boundingBoxSideLength * scaleFactor + 0.5);
            if (boundingBoxSideLength > maxBoundingBoxSideLength)
                boundingBoxSideLength = maxBoundingBoxSideLength;
        }

        // Display all the bounding boxes before NonMaximal Suppression
        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsVector)
        {
            cv::rectangle(testImageClone, prediction.bbox, gtColors[prediction.label]);
        }
        cv::imshow("TestImageOutput", testImageClone);
        cv::waitKey(500);

        // Apply NonMaximal Suppression
        cv::Mat testImageNmsClone = testImage.clone(); // For drawing bbox
        float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
        float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
        {
            vector<Prediction> predictionsNMSVector;
            predictionsNMSVector.reserve(100); // 100 should be enough.
            for (auto &&prediction : predictionsVector)
            {
                // Check if NMS already has a cluster which shares NMS_IOU_THRESHOLD area with current prediction.bbox and both have same label.
                bool clusterFound = false;
                for (auto &&nmsCluster : predictionsNMSVector)
                {
                    if (nmsCluster.label == prediction.label)
                    { // Only if same label
                        Rect &rect1 = prediction.bbox;
                        Rect &rect2 = nmsCluster.bbox;
                        float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
                        if (iouScore > NMS_MAX_IOU_THRESHOLD) // Merge the two bounding boxes
                        {
                            // Update cluster bbox using weighted mean
                            // Point topLeft;
                            // topLeft.x = rect1.x * prediction.confidence + rect2.x * nmsCluster.confidence;
                            // topLeft.y = rect1.y * prediction.confidence + rect2.y * nmsCluster.confidence;
                            // Point bottomRight;
                            // bottomRight.x = (rect1.x + rect1.width) * prediction.confidence + (rect2.x + rect2.width) * nmsCluster.confidence;
                            // bottomRight.y = (rect1.y + rect1.height) * prediction.confidence + (rect2.y + rect2.height) * nmsCluster.confidence;
                            // nmsCluster.bbox = cv::Rect(topLeft, bottomRight);
                            nmsCluster.bbox = rect1 | rect2;
                            nmsCluster.confidence = max(prediction.confidence, nmsCluster.confidence);
                            clusterFound = true;
                            break;
                        } else if(iouScore > NMS_MIN_IOU_THRESHOLD) { // Drop the bounding box with lower confidence
                            if(nmsCluster.confidence < prediction.confidence) {
                                nmsCluster = prediction; 
                            }
                            clusterFound = true;
                            break;
                        }
                    }
                }

                // If no NMS cluster found, add the prediction as a new cluster
                if (!clusterFound)
                {
                    predictionsNMSVector.push_back(prediction);
                }
            }

            // Display all the bounding boxes before NonMaximal Suppression
            cout << "NMS Prediction: " << endl;
            cout << predictionsNMSVector.size() << endl;
            for (auto &&prediction : predictionsNMSVector)
            {
                cv::rectangle(testImageNmsClone, prediction.bbox, gtColors[prediction.label]);
                cout << prediction.label << " " << prediction.bbox.x << " " << prediction.bbox.y << " " << prediction.bbox.height << " " << prediction.bbox.width << endl;
            }
            // cv::imshow("TestImage NMS Output", testImageNmsClone);
            // cv::waitKey(500);
            cout << "Boxes count: " << predictionsVector.size();
            cout << "\nNMS boxes count: " << predictionsNMSVector.size() << '\n';
        } 

        // Draw bounding box on the test image using ground truth
        imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        cv::Mat testImageGtClone = testImage.clone(); // For drawing bbox
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            minSize = min(minSize, min(rect.width, rect.height)); // - For Analysis of scale to be used
            maxSize = max(maxSize, max(rect.width, rect.height)); // - For Analysis of scale to be used
            cv::rectangle(testImageGtClone, rect, gtColors[bbox[0]]);
        }

        // cv::imshow("GrountTruth", testImageGtClone);
        // cv::waitKey(0);

        stringstream modelOutputFilePath;
        modelOutputFilePath << string(PROJ_DIR) << "/output-augmentTrue-undersamplingFalse/" << setfill('0') << setw(4) << i << "-ModelOutput.png";
        string modelOutputFilePathStr = modelOutputFilePath.str();
        cv::imwrite(modelOutputFilePathStr, testImageClone);

        stringstream nmsOutputFilePath;
        nmsOutputFilePath << string(PROJ_DIR) << "/output-augmentTrue-undersamplingFalse/" << setfill('0') << setw(4) << i << "-NMSOutput.png";
        string nmsOutputFilePathStr = nmsOutputFilePath.str();
        cv::imwrite(nmsOutputFilePathStr, testImageNmsClone);

        stringstream gtFilePath;
        gtFilePath << string(PROJ_DIR) << "/output-augmentTrue-undersamplingFalse/" << setfill('0') << setw(4) << i << "-GrountTruth.png";
        string gtFilePathStr = gtFilePath.str();
        cv::imwrite(gtFilePathStr, testImageGtClone);
    }
    cout << minSize << ", " << maxSize << endl;
}

int main()
{
    // // Task 1
    // task1();
    // waitKey(1000);
    // destroyAllWindows();

    // // Task 2
    // task2Basic();
    // waitKey(1000);
    // destroyAllWindows();

    // task2()
    // waitKey(1000);
    // destroyAllWindows();

    // Task 3
    task3();
    waitKey(0);
    int a;
    cin >> a;
    return 0;
}