#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load Haar Cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load("C:/Users/madha/Downloads/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading Haar Cascade for face detection!" << endl;
        return -1;
    }

    // Load YOLO model for person detection
    dnn::Net personDetectionModel = dnn::readNetFromDarknet("C:/Users/madha/Downloads/yolov3.cfg", "C:/Users/madha/Downloads/yolov3.weights");
    if (personDetectionModel.empty()) {
        cerr << "Error loading YOLO model!" << endl;
        return -1;
    }

    // Set preferable backend and target for YOLO
    personDetectionModel.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    personDetectionModel.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // Open video capture
    VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    while (videoCapture.read(frame)) {
        vector<Rect> detectedFaces;
        // Detect faces using Haar Cascade
        faceCascade.detectMultiScale(frame, detectedFaces, 1.1, 3, 0, Size(30, 30));

        int faceCount = 0;

        // Draw blue rectangles around detected faces
        for (const auto& face : detectedFaces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);  // Blue rectangle for faces
            faceCount++;
        }

        // Prepare input for YOLO
        Mat inputBlob;
        dnn::blobFromImage(frame, inputBlob, 0.00392, Size(416, 416), Scalar(0, 0, 0), true, false);
        personDetectionModel.setInput(inputBlob);
        
        vector<Mat> detections;
        personDetectionModel.forward(detections, personDetectionModel.getUnconnectedOutLayersNames());

        const float confidenceThreshold = 0.5;

        // Process YOLO detections and draw green boxes for persons
        for (size_t i = 0; i < detections.size(); ++i) {
            Mat detection = detections[i];
            for (int j = 0; j < detection.rows; ++j) {
                const int confidenceIndex = 5;
                Mat classScores = detection.row(j).colRange(confidenceIndex, detection.cols);
                
                Point classIdPoint;
                double confidence;
                minMaxLoc(classScores, nullptr, &confidence, nullptr, &classIdPoint);

                if (confidence > confidenceThreshold) {
                    int centerX = static_cast<int>(detection.at<float>(j, 0) * frame.cols);
                    int centerY = static_cast<int>(detection.at<float>(j, 1) * frame.rows);
                    int width = static_cast<int>(detection.at<float>(j, 2) * frame.cols);
                    int height = static_cast<int>(detection.at<float>(j, 3) * frame.rows);
                    Rect personBox(centerX - width / 2, centerY - height / 2, width, height);

                    rectangle(frame, personBox, Scalar(0, 255, 0), 2);  // Green rectangle for persons
                }
            }
        }

        // Display face count on the frame
        string faceCountText = "Faces Detected: " + to_string(faceCount);
        putText(frame, faceCountText, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);

        // Show the final frame with detections
        imshow("Person Detection", frame);

        // Print the number of faces detected to the terminal
        cout << "Number of Persons = " << faceCount << endl;

        // Exit on 'Esc' key
        if (waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
