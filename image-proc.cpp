#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <iostream>

using namespace cv;

// Function declerations
cv::Mat grayScale(const cv::Mat&);
cv::Mat CannyEdgeDetector(const cv::Mat&, int, int, double, int, int);
cv::Mat applyGaussianFilter(const cv::Mat&, const cv::Mat&);
cv::Mat createGaussianKernel(int, double);
cv::Mat createCustomKernel();
std::pair<cv::Mat, cv::Mat> GradientsIntensity(const cv::Mat&);
cv::Mat NonMaxSuppression(const cv::Mat&, const cv::Mat&);
cv::Mat DoubleThresholding(const cv::Mat&, int, int);
cv::Mat Hysteresis(const cv::Mat&);
cv::Mat Halftone(const cv::Mat&, int);
cv::Mat floyedSteinberg(const cv::Mat&);


int main(){
    // Reading target image
    std::string image_path = samples::findFile("Lenna.png");
    cv::Mat originImg = imread(image_path, IMREAD_COLOR);
    if(originImg.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Task 1: Grayscale
    cv:: Mat grayImg = grayScale(originImg);
    imwrite("grayScale.png", grayImg);

    // Task 2: Canny Edge Detection
    int kernel = 2; // 1 - custom kernel , 2 - gaussian kernel
    int ksize = 5; // for gaussian kernel
    double sigma = 1.0; // // for gaussian kernel
    int lowThreshold = 100;
    int strongThreshold = 200;
    cv:: Mat cedImg = CannyEdgeDetector(originImg, kernel, ksize, sigma, lowThreshold, strongThreshold);
    imwrite("cannyEdgeDetector.png", cedImg);

    // Task 3: Haftone
    cv:: Mat halftoneImg = Halftone(grayImg, 4);
    imwrite("halftone.png", halftoneImg);

    // Task 4: FloyedSteinberg
    cv:: Mat fsImg = floyedSteinberg(grayImg);
    fsImg *= 16;
    imwrite("floyedSteinbergImg.png", fsImg);

    return 0;
}

// Task 1: Apply Grayscale filter
cv::Mat grayScale(const cv::Mat& img) {
    cv::Mat grayscaleImage(img.rows, img.cols, CV_8UC1); // 8 BPP, 1 channel
    for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                cv::Vec3b color = img.at<cv::Vec3b>(row, col); // Access RGB pixel (ord: BGR)
                uchar grayscale = static_cast<uchar>(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]);
                grayscaleImage.at<uchar>(row, col) = grayscale; // Assign grayscale value
            }
    }
    return grayscaleImage;
}

// Task 2: Apply CannyEdgeDetector filter
cv::Mat CannyEdgeDetector(const cv::Mat& img, int k, int ksize, double sigma, int lowThreshold, int strongThreshold) {
// step A - grayscale
    cv:: Mat grayScaleImg = grayScale(img);

// step B - noise reduction
    cv:: Mat kernel;
    if (k == 1)
        kernel = createCustomKernel();
    else if (k == 2)
        kernel = createGaussianKernel(ksize, sigma);
    cv:: Mat noiseReductionImg = applyGaussianFilter(grayScaleImg, kernel);
    
// step C - gradient intensity (gradient calculation)
    std::pair<cv::Mat, cv::Mat> gradients = GradientsIntensity(noiseReductionImg);
    cv::Mat gradientImg = gradients.first;
    cv::Mat directionImg = gradients.second;

// step D - NMS for edge sharpening
    cv::Mat NonMaxSuppressionImg = NonMaxSuppression(gradientImg, directionImg);
   // cv::normalize(NonMaxSuppressionImg, NonMaxSuppressionImg, 0, 255, cv::NORM_MINMAX, CV_8U);
    
// step E - double thresholding (detection of weak and strong edges)
    cv::Mat doubleThresholdingImg = DoubleThresholding(NonMaxSuppressionImg, lowThreshold, strongThreshold);

// step F - Hysteresis (edge tracking and connection of disconected lines)
    cv::Mat hysteresisImg = Hysteresis(doubleThresholdingImg);
    
    return hysteresisImg;
}

// Task 2.1: Apply Gaussian filter manually
cv::Mat applyGaussianFilter(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat filteredImage = img.clone(); // Clone the input image for output
    int ksize = kernel.rows; // Assume a square kernel
    int offset = ksize / 2;

    // Apply convolution
    for (int i = offset; i < img.rows - offset; i++) {
        for (int j = offset; j < img.cols - offset; j++) {
            double sum = 0.0;
            for (int m = -offset; m <= offset; m++) {
                for (int n = -offset; n <= offset; n++) {
                    sum += img.at<uchar>(i + m, j + n) * kernel.at<double>(m + offset, n + offset);
                }
            }
            filteredImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum); // Assign to output
        }
    }
    return filteredImage;
}
// Generate a Gaussian kernel
cv::Mat createGaussianKernel(int ksize, double sigma) {
    cv::Mat kernel(ksize, ksize, CV_64F); // Double-precision kernel
    double sum = 0.0;
    int offset = ksize / 2;

    for (int i = -offset; i <= offset; i++) {
        for (int j = -offset; j <= offset; j++) {
            double value = (1 / (2 * CV_PI * sigma * sigma)) *
                           exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + offset, j + offset) = value;
            sum += value;
        }
    }
    // Normalize the kernel
    kernel /= sum;
    return kernel;
} 
cv::Mat createCustomKernel() {
     cv:: Mat kernel = (cv::Mat_<double>(3, 3) << 
                1,  2, 1,
                2,  4, 2,
                1,  2, 1);
      kernel /= 16;  
      return kernel;         
}

// Task 2.2: Gradient calculation
std::pair<cv::Mat, cv::Mat> GradientsIntensity(const cv::Mat& img){
    cv::Mat floatImg;
    img.convertTo(floatImg, CV_64F);
    cv:: Mat Gx = (cv::Mat_<double>(3, 3) << 
                1,  0, -1,
                2,  0, -2,
                1,  0, -1);
    cv:: Mat Gy;
    cv:: transpose(Gx,Gy);
    cv::Mat verticalImg(floatImg.rows, floatImg.cols, CV_32F, cv::Scalar(0));
    cv::Mat horizontaImg(floatImg.rows, floatImg.cols, CV_32F, cv::Scalar(0));

    for (int i = 1; i < floatImg.rows - 1; i++) {
        for (int j = 1; j < floatImg.cols - 1; j++) {
            double sumX = 0.0;
            double sumY = 0.0;
            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    sumX += floatImg.at<double>(i + m, j + n) * Gx.at<double>(m + 1, n + 1);
                    sumY += floatImg.at<double>(i + m, j + n) * Gy.at<double>(m + 1, n + 1);
                }
            }
            verticalImg.at<float>(i, j) = sumX;
            horizontaImg.at<float>(i, j) = sumY;
        }
    }

    cv::Mat gradientImg(floatImg.rows, floatImg.cols, CV_32F, cv::Scalar(0));
    cv::Mat directionImg(floatImg.rows, floatImg.cols, CV_32F, cv::Scalar(0));
    for (int i = 1; i < gradientImg.rows -1; i++) {
        for (int j = 1; j < gradientImg.cols -1; j++) {
            float xVal = verticalImg.at<float>(i, j);
            float yVal =  horizontaImg.at<float>(i, j);
            gradientImg.at<float>(i, j) = std::sqrt(xVal * xVal + yVal * yVal);
            directionImg.at<float>(i, j) = std::atan2(yVal, xVal);
        }
    }
    return {gradientImg, directionImg};
}

// Task 2.3: Non-Maximum Suppression
cv::Mat NonMaxSuppression(const cv::Mat& gradientImg, const cv::Mat& directionImg) {
    // Create an output image initialized to zero
    cv::Mat nmsImg = cv::Mat::zeros(gradientImg.size(), CV_32F);

    for (int i = 1; i < gradientImg.rows - 1; i++) {
        for (int j = 1; j < gradientImg.cols - 1; j++) {
            float magnitude = gradientImg.at<float>(i, j);
            float angle = directionImg.at<float>(i, j);
            // Normalize angle to [0, 180) degrees
            angle = fmod((angle * 180.0 / CV_PI) + 180.0, 180.0);

            // Determine which neighbors to compare
            float neighbor1 = 0, neighbor2 = 0;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                // Horizontal (0°): Compare left and right neighbors
                neighbor1 = gradientImg.at<float>(i, j - 1);
                neighbor2 = gradientImg.at<float>(i, j + 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal (45°): Compare top-right and bottom-left
                neighbor1 = gradientImg.at<float>(i - 1, j + 1);
                neighbor2 = gradientImg.at<float>(i + 1, j - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                // Vertical (90°): Compare top and bottom neighbors
                neighbor1 = gradientImg.at<float>(i - 1, j);
                neighbor2 = gradientImg.at<float>(i + 1, j);
            } else if (angle >= 112.5 && angle < 157.5) {
                // Diagonal (135°): Compare top-left and bottom-right
                neighbor1 = gradientImg.at<float>(i - 1, j - 1);
                neighbor2 = gradientImg.at<float>(i + 1, j + 1);
            }
            // Keep the pixel if it is a local maximum
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                nmsImg.at<float>(i, j) = magnitude;
            } else {
                nmsImg.at<float>(i, j) = 0; // Suppress pixel
            }
        }
    }
    return nmsImg;
}
// Task 2.4: Double thresholding
cv::Mat DoubleThresholding(const cv::Mat& img, int lowThreshold, int strongThreshold){
    // Create an output image initialized to zero
    cv::Mat dtImg = cv::Mat::zeros(img.size(), CV_8U);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            float pixelValue = img.at<float>(i, j);
            // Non-edges
            if(pixelValue < lowThreshold)
                dtImg.at<uchar>(i, j) = 0;
            // Weak edges
            else if(pixelValue < strongThreshold)
                dtImg.at<uchar>(i, j) = 128;
            // Strong edges
            else{
                dtImg.at<uchar>(i, j) = 255;
            }
        }
    }
    return dtImg;
}
// Task 2.5: Edge detection
cv::Mat Hysteresis(const cv::Mat& img){
    // Create an output image initialized to zero
    cv::Mat hstImg = img.clone();
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if(img.at<uchar>(i , j) == 128){
                int max_value = std::max({img.at<uchar>(i-1 , j-1),
                                        img.at<uchar>(i-1 , j),
                                        img.at<uchar>(i , j-1),
                                        img.at<uchar>(i+1 , j+1),
                                        img.at<uchar>(i-1 , j+1),
                                        img.at<uchar>(i+1 , j-1),
                                        img.at<uchar>(i+1 , j),
                                        img.at<uchar>(i , j+1)});
                    if(max_value==255){
                        hstImg.at<uchar>(i , j) = 255;
                    }
                    else
                        hstImg.at<uchar>(i , j) = 0;
            }
        }
    }
    return hstImg;
}

// Task 3: Halftone
cv::Mat Halftone(const cv::Mat& grayscaleImg, int blockSize) {
    // Ensure blockSize is even for proper halftone blocks
    if (blockSize % 2 != 0) {
        throw std::invalid_argument("blockSize must be an even number.");
    }

    // Half the block size for each axis
    int halfBlock = blockSize / 2;

    // Create a larger image to hold the halftone pattern
    cv::Mat halftoneImg = cv::Mat::zeros(grayscaleImg.rows * halfBlock, grayscaleImg.cols * halfBlock, CV_8U);

    // Predefine the patterns
    uchar pattern1[2][2] = {{0, 0}, {0, 0}};       // For intensity <= 0.2 * 255
    uchar pattern2[2][2] = {{0, 255}, {0, 0}};     // For intensity <= 0.4 * 255
    uchar pattern3[2][2] = {{0, 255}, {255, 0}};   // For intensity <= 0.6 * 255
    uchar pattern4[2][2] = {{0, 255}, {255, 255}}; // For intensity <= 0.8 * 255
    uchar pattern5[2][2] = {{255, 255}, {255, 255}}; // For intensity > 0.8 * 255

    for (int i = 0; i < grayscaleImg.rows; i++) {
        for (int j = 0; j < grayscaleImg.cols; j++) {
            uchar intensity = grayscaleImg.at<uchar>(i, j);  // Access grayscale pixel
            uchar (*selectedPattern)[2] = nullptr;          // Pointer to selected pattern

            // Select the appropriate pattern based on intensity
            if (intensity <= 0.2 * 255) {
                selectedPattern = pattern1;
            } else if (intensity <= 0.4 * 255) {
                selectedPattern = pattern2;
            } else if (intensity <= 0.6 * 255) {
                selectedPattern = pattern3;
            } else if (intensity <= 0.8 * 255) {
                selectedPattern = pattern4;
            } else {
                selectedPattern = pattern5;
            }

            // Map the selected pattern to the corresponding block in the halftone image
            for (int bi = 0; bi < halfBlock; bi++) {
                for (int bj = 0; bj < halfBlock; bj++) {
                    halftoneImg.at<uchar>(i * halfBlock + bi, j * halfBlock + bj) = 
                        selectedPattern[bi % 2][bj % 2];
                }
            }
        }
    }

    return halftoneImg;
}
// Task 4: FloyedSteinberg
cv::Mat floyedSteinberg(const cv::Mat& img){
    cv::Mat grayImg = img.clone();
    cv::Mat fsImg = cv::Mat::zeros(img.size(), CV_8U);; //img.clone();
    double a = 7/16;
    double b = 3/16;
    double c = 5/16;
    double d = 1/16;
    double error;
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float value = grayImg.at<uchar>(i,j)/16 + 0.5;
            fsImg.at<uchar>(i,j) = trunc(value);
            error = grayImg.at<uchar>(i,j) - fsImg.at<uchar>(i,j);
            cv::saturate_cast<uchar>(grayImg.at<uchar>(i, j + 1) + a * error);
            cv::saturate_cast<uchar>(grayImg.at<uchar>(i+1, j - 1) + b * error);
            cv::saturate_cast<uchar>(grayImg.at<uchar>(i+1, j ) + c * error);
            cv::saturate_cast<uchar>(grayImg.at<uchar>(i+1, j + 1) + d * error);
        }
    }

    return fsImg;
}



// USEFUL STUFF
// print matrix:   std::cout << "Matrix:\n" << img << std::endl;
// imshow("Display window", img);
 // int k = waitKey(0); // Wait for a keystroke in the window
// if(k == 's')
    // {
    //     imwrite("Lenna.png", img);
    // }
 // imshow("Display window", grayImg);
    // int k = waitKey(0);
  