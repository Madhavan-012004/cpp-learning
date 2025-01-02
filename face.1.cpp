#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace dlib;
using namespace std;

int main() {
    // Load Haar Cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load("C:/Users/madha/Downloads/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading Haar Cascade for face detection!" << endl;
        return -1;
    }

    // Load Dlib models
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    anet_type net;

    try {
        deserialize("C:/Users/madha/Downloads/shape_predictor_68_face_landmarks.dat") >> sp;
        deserialize("C:/Users/madha/Downloads/dlib_face_recognition_resnet_model_v1 (3).dat") >> net;
    } catch (exception &e) {
        cerr << "Error loading Dlib models: " << e.what() << endl;
        return -1;
    }

    // Load and process the reference image
    string referenceImagePath = "M:/cpp/Dataset/Madhavan/WhatsApp Image 2024-12-08 at 08.23.00_1818539d.jpg";
    Mat referenceImage = imread(referenceImagePath);
    if (referenceImage.empty()) {
        cerr << "Error loading reference image!" << endl;
        return -1;
    }

    cv_image<bgr_pixel> dlibRefImage(referenceImage);
    std::vector<matrix<float, 0, 1>> knownFaceDescriptors;
    std::vector<string> knownNames;

    // Detect face in the reference image
    std::vector<dlib::rectangle> refFaces = detector(dlibRefImage);
    if (refFaces.empty()) {
        cerr << "No face detected in the reference image!" << endl;
        return -1;
    }

    for (auto &face : refFaces) {
        auto shape = sp(dlibRefImage, face);
        matrix<rgb_pixel> faceChip;
        extract_image_chip(dlibRefImage, get_face_chip_details(shape), faceChip);
        knownFaceDescriptors.push_back(net(faceChip));
        knownNames.push_back("Madhavan");
    }

    // Open video capture
    VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    while (videoCapture.read(frame)) {
        cv_image<bgr_pixel> dlibFrame(frame);
        std::vector<dlib::rectangle> detectedFaces = detector(dlibFrame);

        for (auto &face : detectedFaces) {
            auto shape = sp(dlibFrame, face);
            matrix<rgb_pixel> faceChip;
            extract_image_chip(dlibFrame, get_face_chip_details(shape), faceChip);
            matrix<float, 0, 1> faceDescriptor = net(faceChip);

            // Compare with known faces
            string detectedName = "Unknown";
            double minDistance = 0.6; // Threshold for face matching
            for (size_t i = 0; i < knownFaceDescriptors.size(); i++) {
                double distance = length(faceDescriptor - knownFaceDescriptors[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    detectedName = knownNames[i];
                }
            }

            // Draw rectangle and name
            rectangle(frame, Rect(face.left(), face.top(), face.width(), face.height()), Scalar(255, 0, 0), 2);
            putText(frame, detectedName, Point(face.left(), face.top() - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }

        // Show the video frame with detections
        imshow("Face Recognition", frame);

        // Exit on pressing 'Esc'
        if (waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
