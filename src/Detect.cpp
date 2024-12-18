#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <map>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Function to encode faces from the dataset
map<string, vector<Mat>> trainDataset(const string& datasetPath, CascadeClassifier& faceCascade) {
    map<string, vector<Mat>> faceEncodings;

    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (fs::is_directory(entry)) {
            string label = entry.path().filename().string(); // Get person's name
            for (const auto& imgFile : fs::directory_iterator(entry.path())) {
                Mat img = imread(imgFile.path().string());
                if (img.empty()) continue;

                vector<Rect> faces;
                faceCascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));
                if (!faces.empty()) {
                    Mat faceROI = img(faces[0]); // Use the first detected face
                    Mat faceEncoding;
                    resize(faceROI, faceEncoding, Size(128, 128)); // Resize for encoding
                    faceEncoding = faceEncoding.reshape(1, 1);     // Flatten for comparison
                    faceEncodings[label].push_back(faceEncoding); // Store encoding
                }
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

    for (const auto& [label, encodings] : faceEncodings) {
        for (const auto& encoding : encodings) {
            double distance = norm(detectedFace, encoding, NORM_L2);
            if (distance < minDistance && distance < 80) { // Adjust threshold if needed
                minDistance = distance;
                bestMatch = label;
            }
        }
    }

    return bestMatch;
}

int main() {
    string datasetPath = "M:/cpp/Dataset";
    string faceCascadePath = "C:/Users/madha/Downloads/haarcascade_frontalface_default.xml";

    // Load Haar Cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load(faceCascadePath)) {
        cerr << "Error loading Haar Cascade!" << endl;
        return -1;
    }

    // Train the dataset
    map<string, vector<Mat>> faceEncodings = trainDataset(datasetPath, faceCascade);

    // Open video capture
    VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    while (videoCapture.read(frame)) {
        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(frame, detectedFaces, 1.1, 3, 0, Size(30, 30));

        for (const auto& face : detectedFaces) {
            Mat faceROI = frame(face);
            Mat resizedFace;
            resize(faceROI, resizedFace, Size(128, 128));
            resizedFace = resizedFace.reshape(1, 1); // Flatten for comparison

            // Match the face
            string detectedName = matchFace(resizedFace, faceEncodings);

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
