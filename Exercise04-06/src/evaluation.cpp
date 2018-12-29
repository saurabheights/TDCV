#include <opencv2/opencv.hpp>
#include <HogVisualization.h>
#include <RandomForest.h>
#include <iomanip>
#include <sstream>
#include <random>
#include <opencv2/core/utils/filesystem.hpp>
using namespace cv;
using namespace std;

// #define DISPLAY

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

vector<float> computeTpFpFn(vector<Prediction> predictionsNMSVector,
                            vector<Prediction> groundTruthPredictions)
{
    float tp = 0, fp = 0, fn = 0;
    float matchThresholdIou = 0.5f;

    for (auto &&myPrediction : predictionsNMSVector)
    {
        bool matchesWithAnyGroundTruth = false;
        Rect myRect = myPrediction.bbox;

        for (auto &&groundTruth : groundTruthPredictions)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            Rect gtRect = groundTruth.bbox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                matchesWithAnyGroundTruth = true;
                break;
            }
        }

        if (matchesWithAnyGroundTruth)
            tp++;
        else
            fp++;
    }

    for (auto &&groundTruth : groundTruthPredictions)
    {
        bool isGtBboxMissed = true;
        Rect gtRect = groundTruth.bbox;
        for (auto &&myPrediction : predictionsNMSVector)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            Rect myRect = myPrediction.bbox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                isGtBboxMissed = false;
                break;
            }
        }

        if (isGtBboxMissed)
            fn++;
    }

    vector<float> results;
    results.push_back(tp);
    results.push_back(fp);
    results.push_back(fn);
    return results;
}

vector<float> task3_core(string outputDir,
                         vector<pair<int, cv::Mat>> &testImagesLabelVector,
                         vector<vector<vector<int>>> &labelAndBoundingBoxes,
                         cv::Scalar *gtColors,
                         float NMS_MIN_IOU_THRESHOLD,
                         float NMS_MAX_IOU_THRESHOLD,
                         float NMS_CONFIDENCE_THRESHOLD)
{
    ifstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open())
    {
        cout << "Failed to open" << outputDir + "predictions.txt" << endl;
        exit(-1);
    }

    float tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        int fileNumber;
        predictionsFile >> fileNumber; // Prediction file format: Starts with File number
        assert(fileNumber == i);

        // Ignore the ground truth data in predictions.txt. we already have it.
        int tmp, tmp1;
        predictionsFile >> tmp; // Ignore - number of ground truth
        vector<Prediction> groundTruthPredictions;
        for (size_t j = 0; j < tmp; j++)
        {
            Prediction groundTruthPrediction;
            groundTruthPrediction.label = labelAndBoundingBoxes.at(i).at(j).at(0);
            groundTruthPrediction.bbox.x = labelAndBoundingBoxes.at(i).at(j).at(1);
            groundTruthPrediction.bbox.y = labelAndBoundingBoxes.at(i).at(j).at(2);
            groundTruthPrediction.bbox.height = labelAndBoundingBoxes.at(i).at(j).at(3);
            groundTruthPrediction.bbox.height -= groundTruthPrediction.bbox.x;
            groundTruthPrediction.bbox.width = labelAndBoundingBoxes.at(i).at(j).at(4);
            groundTruthPrediction.bbox.width -= groundTruthPrediction.bbox.y;
            groundTruthPredictions.push_back(groundTruthPrediction);

            predictionsFile >> tmp1; // Ignore - label;
            for (size_t k = 0; k < 4; k++)
            {
                predictionsFile >> tmp1; // Ignore - rectangle
            }
        }

        // Read prediction data
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        vector<Prediction> predictionsVector; // Output of Hog Detection on ith test image
        int numOfPredictions;
        predictionsFile >> numOfPredictions;
        predictionsVector.reserve(numOfPredictions);
        for (size_t i = 0; i < numOfPredictions; i++)
        {
            Prediction prediction;
            predictionsFile >> prediction.label;
            predictionsFile >> prediction.bbox.x >> prediction.bbox.y >> prediction.bbox.height >> prediction.bbox.width;
            predictionsFile >> prediction.confidence;
            predictionsVector.push_back(prediction);
        }

        // Display all the bounding boxes before NonMaximal Suppression
#ifdef DISPLAY
        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsVector)
        {
            cv::rectangle(testImageClone, prediction.bbox, gtColors[prediction.label]);
        }
        cv::imshow("TestImageOutput", testImageClone);
        cv::waitKey(100);
#endif

        // Apply NonMaximal Suppression
        cv::Mat testImageNms1Clone = testImage.clone(); // For drawing bbox
        cv::Mat testImageNmsClone = testImage.clone();  // For drawing bbox
        vector<Prediction> predictionsNMSVector;
        predictionsNMSVector.reserve(20); // 20 should be enough.
        for (auto &&prediction : predictionsVector)
        {
            // Ignore boxes with low threshold.
            if (prediction.confidence < NMS_CONFIDENCE_THRESHOLD)
                continue;
            cv::rectangle(testImageNms1Clone, prediction.bbox, gtColors[prediction.label]);

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
                        nmsCluster.bbox = rect1 | rect2;
                        nmsCluster.confidence = max(prediction.confidence, nmsCluster.confidence);
                        clusterFound = true;
                        break;
                    }
                    else if (iouScore > NMS_MIN_IOU_THRESHOLD) // ToDo: Improve this.
                    {
                        // Drop the bounding box with lower confidence
                        if (nmsCluster.confidence < prediction.confidence)
                        {
                            nmsCluster = prediction;
                        }
                        clusterFound = true;
                        break;
                    }
                }
            }

            // If no NMS cluster found, add the prediction as a new cluster
            if (!clusterFound)
                predictionsNMSVector.push_back(prediction);
        }

        // Prediction file format: Next is N Lines of Labels and cv::Rect
        for (auto &&prediction : predictionsNMSVector)
            cv::rectangle(testImageNmsClone, prediction.bbox, gtColors[prediction.label]);

#ifdef DISPLAY
        // Display all the bounding boxes before NonMaximal Suppression
        cv::imshow("TestImage NMS BBox Filter", testImageNms1Clone);
        cv::imshow("TestImage NMS Output", testImageNmsClone);
        cv::waitKey(500);
#endif

        // cout << "Boxes count: " << predictionsVector.size();
        // cout << "\nNMS boxes count: " << predictionsNMSVector.size() << '\n';

#ifdef DISPLAY
        // Display all ground truth boxes
        cv::Mat testImageGtClone = testImage.clone(); // For drawing bbox
        for (size_t j = 0; j < groundTruthPredictions.size(); j++)
            cv::rectangle(testImageGtClone, groundTruthPredictions.at(j).bbox, gtColors[groundTruthPredictions.at(j).label]);
        cv::imshow("Ground Truth", testImageGtClone);
        cv::waitKey(500);
#endif

        // Write NMS output image
        stringstream nmsOutputFilePath;
        nmsOutputFilePath << outputDir << setfill('0') << setw(4) << i << "-Confidence-" << NMS_CONFIDENCE_THRESHOLD << "-NMSOutput.png";
        string nmsOutputFilePathStr = nmsOutputFilePath.str();
        cv::imwrite(nmsOutputFilePathStr, testImageNmsClone);

        // Compute precision and recall using groundTruthPredictions and predictionsNMSVector
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        vector<float> tpFpFn = computeTpFpFn(predictionsNMSVector, groundTruthPredictions);
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        tp += tpFpFn[0];
        fp += tpFpFn[1];
        fn += tpFpFn[2];
    }

    float precision = tp / (tp + fp);
    float recall = tp / (tp + fn);
    predictionsFile.close();
    vector<float> precisionRecallValue;
    precisionRecallValue.push_back(precision);
    precisionRecallValue.push_back(recall);
    return precisionRecallValue;
}

int main()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask3Dataset();
    // Load the ground truth bounding boxes values
    vector<vector<vector<int>>> labelAndBoundingBoxes = getLabelAndBoundingBoxes();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);

    cv::Scalar gtColors[4];
    gtColors[0] = cv::Scalar(255, 0, 0);
    gtColors[1] = cv::Scalar(0, 255, 0);
    gtColors[2] = cv::Scalar(0, 0, 255);
    gtColors[3] = cv::Scalar(255, 255, 0);

    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
    // float NMS_CONFIDENCE_THRESHOLD = 0.75f;

    // Loop over multiple values.
    string outputDir;
    cout << "Enter path to dir of predictions.txt: ";
    // cin >> outputDir;
    outputDir = "/media/sk/6a4a41c4-a920-46db-84c5-69e0450c2dd0/mega/TUM-Study/TrackingAndDetectionInComputerVision/Exercises/Exercise04-06/output/Trees-50_subsetPercent-50-undersampling_0-augment_1-strideX_2-strideY_2-NMS_MIN_0.1-NMS_Max_0.5-NMS_CONF_0.6";
    if (outputDir.at(outputDir.length() - 1) != '/')
    {
        outputDir += '/';
    }

#ifdef DISPLAY
    cv::namedWindow("TestImageOutput");
    cv::namedWindow("TestImage NMS Output");
    cv::namedWindow("Ground Truth");
    cv::namedWindow("TestImage NMS BBox Filter");
    cv::waitKey(0);
#endif

cout << "\n";
    ofstream outputFile;
    outputFile.open(outputDir+"predictionRecallValues.csv");
    if (!outputFile.is_open())
    {
        cout << "Failed to open" << outputDir+"predictionRecallValues.csv" << endl;
        exit(-1);
    }
    outputFile << "Precision,Recall"<< endl;
    for (float NMS_CONFIDENCE_THRESHOLD = 0; NMS_CONFIDENCE_THRESHOLD <= 1; NMS_CONFIDENCE_THRESHOLD += 0.05)
    {
        vector<float> precisionRecallValue = task3_core(outputDir, testImagesLabelVector, labelAndBoundingBoxes, gtColors, NMS_MIN_IOU_THRESHOLD, NMS_MAX_IOU_THRESHOLD, NMS_CONFIDENCE_THRESHOLD);
        cout << "NMS_CONFIDENCE_THRESHOLD: " << NMS_CONFIDENCE_THRESHOLD << ", Precision: " << precisionRecallValue[0] << ", Recall: " << precisionRecallValue[1] << endl;
        outputFile << precisionRecallValue[0] << "," << precisionRecallValue[1] << endl;
    }
    outputFile.close();

cout << "\n";
}