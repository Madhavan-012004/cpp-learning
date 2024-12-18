#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    string datasetPath = "M:/cpp/Dataset";  // Path to your dataset
    string faceCascadePath = "C:/Users/madha/Downloads/haarcascade_frontalface_default.xml";  // Path to Haar Cascade

    // Load Haar Cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load(faceCascadePath)) {
        cerr << "Error loading Haar Cascade!" << endl;
        return -1;
    }

    // Create the LBPH face recognizer
    Ptr<face::LBPHFaceRecognizer> faceRecognizer = face::LBPHFaceRecognizer::create();

    // Prepare training data
    vector<Mat> images;  // Store face images
    vector<int> labels;  // Store corresponding labels

    // Iterate through the dataset
    int label = 0;  // Start assigning labels from 0
    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (fs::is_directory(entry)) {
            string labelName = entry.path().filename().string();  // Get folder name (person's name)
            for (const auto& imgFile : fs::directory_iterator(entry.path())) {
                Mat img = imread(imgFile.path().string(), IMREAD_GRAYSCALE);  // Read image in grayscale
                if (img.empty()) continue;

                vector<Rect> faces;
                faceCascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));

                if (!faces.empty()) {
                    Mat faceROI = img(faces[0]);  // Take the first detected face
                    images.push_back(faceROI);
                    labels.push_back(label);  // Assign label for this face
                }
            }
            label++;  // Increment label for next person
        }
    }

    // Train the face recognizer
    if (images.empty()) {
        cerr << "No faces found in the dataset!" << endl;
        return -1;
    }

    faceRecognizer->train(images, labels);
    cout << "Training completed!" << endl;

    // Save the trained model (optional)
    faceRecognizer->save("trained_model.xml");

    // Open video capture
    VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    while (videoCapture.read(frame)) {
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);  // Convert to grayscale

        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(grayFrame, detectedFaces, 1.1, 3, 0, Size(30, 30));

        for (const auto& face : detectedFaces) {
            Mat faceROI = grayFrame(face);  // Get the face region of interest

            // Recognize the face using the trained model
            int predictedLabel = -1;
            double confidence = 0.0;
            faceRecognizer->predict(faceROI, predictedLabel, confidence);

            string detectedName = "Unknown Person";
            if (predictedLabel != -1) {
                // Map label to person's name (based on folder name)
                vector<string> labelsList = {"Madhavan", "John", "Alice"};  // Update with your person names
                detectedName = labelsList[predictedLabel];
            }

            // Draw bounding box and label
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            putText(frame, detectedName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            // Print to terminal
            cout << detectedName << " detected!" << endl;
        }

        imshow("Face Recognition", frame);

        // Exit on 'Esc' key
        if (waitKey(1) == 27) break;
    }

    videoCapture.release();
    destroyAllWindows();
    return 0;
}
