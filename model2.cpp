#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <map>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Function to encode faces using DNN-based face embeddings
map<string, vector<Mat>> trainDataset(const string& datasetPath, dnn::Net& faceNet) {
    map<string, vector<Mat>> faceEncodings;

    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (fs::is_directory(entry)) {
            string label = entry.path().filename().string(); // Get person's name
            for (const auto& imgFile : fs::directory_iterator(entry.path())) {
                Mat img = imread(imgFile.path().string());
                if (img.empty()) continue;

                // Preprocess the image for DNN model
                Mat blob = dnn::blobFromImage(img, 1.0, Size(224, 224), Scalar(104.0, 177.0, 123.0), false, false);

                // Pass the image through the DNN model
                faceNet.setInput(blob);
                Mat feature = faceNet.forward();

                faceEncodings[label].push_back(feature); // Store the embedding
            }
        }
    }

    cout << "Training completed for " << faceEncodings.size() << " person(s)." << endl;
    return faceEncodings;
}

// Function to find the best match for a detected face
string matchFace(const Mat& detectedFace, const map<string, vector<Mat>>& faceEncodings) {
    double minDistance = DBL_MAX;
    string bestMatch = "Unknown Person";

    // Preprocess the detected face for the model
    Mat blob = dnn::blobFromImage(detectedFace, 1.0, Size(224, 224), Scalar(104.0, 177.0, 123.0), false, false);

    // Compare to all stored face encodings
    for (const auto& [label, encodings] : faceEncodings) {
        for (const auto& encoding : encodings) {
            double distance = norm(encoding, blob, NORM_L2);
            if (distance < minDistance && distance < 0.6) { // Adjust threshold if needed
                minDistance = distance;
                bestMatch = label;
            }
        }
    }

    return bestMatch;
}

int main() {
    // Load the pre-trained DNN model for face feature extraction
    string modelPath = "M:/cpp/FaceONNX/face_model.onnx"; // Path to your DNN model
    dnn::Net faceNet = dnn::readNetFromONNX(modelPath);

    if (faceNet.empty()) {
        cerr << "Error loading DNN model!" << endl;
        return -1;
    }

    string datasetPath = "M:/cpp/Dataset";

    // Train the dataset using the DNN model
    map<string, vector<Mat>> faceEncodings = trainDataset(datasetPath, faceNet);

    // Open video capture
    VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    while (videoCapture.read(frame)) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Use Haar Cascade to detect faces
        CascadeClassifier faceCascade;
        faceCascade.load("C:/Users/madha/Downloads/haarcascade_frontalface_default.xml");

        if (faceCascade.empty()) {
            cerr << "Error loading Haar Cascade!" << endl;
            return -1;
        }

        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(gray, detectedFaces, 1.1, 3, 0, Size(30, 30));

        for (const auto& face : detectedFaces) {
            Mat faceROI = frame(face);

            // Match the detected face with the trained dataset
            string detectedName = matchFace(faceROI, faceEncodings);

            // Draw bounding box and label
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            putText(frame, detectedName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            // Print to terminal
            if (detectedName == "Unknown Person") {
                cout << "Unknown person detected!" << endl;
            } else {
                cout << detectedName << " detected!" << endl;
            }
        }

        imshow("Face Recognition", frame);

        // Exit on 'Esc' key
        if (waitKey(1) == 27) break;
    }

    videoCapture.release();
    destroyAllWindows();
    return 0;
}
