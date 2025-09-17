# libcnn.dll Usage Manual

A simple C++ YOLO inference library converted from yolo8onnx Python library.

## Overview

libcnn.dll provides a simple interface for YOLO object detection using ONNX Runtime. It supports both CUDA and CPU execution with automatic fallback.

## Library Files

- `libcnn.dll` - Main library
- `libcnn.lib` - Import library for linking
- `cnn_impl.h` - Header file with class definitions

## Dependencies

Required DLLs (included in release):
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `onnxruntime_providers_tensorrt.dll`
- `opencv_world4110.dll`
- `opencv_videoio_ffmpeg4110_64.dll`

## API Reference

### CnnModel Class

```cpp
namespace cnn {
    class CNN_API CnnModel {
    public:
        CnnModel(const std::string& path, float conf_th = 0.3f, float iou_th = 0.4f);
        ~CnnModel() = default;

        std::vector<std::vector<float>> infer(const cv::Mat& image);
        std::tuple<cv::Mat, std::pair<int, int>, float> resize(const cv::Mat& image, cv::Size shape);
    };
}
```

### Constructor

```cpp
CnnModel(const std::string& path, float conf_th = 0.3f, float iou_th = 0.4f)
```

**Parameters:**
- `path` - Path to ONNX model file
- `conf_th` - Confidence threshold (default: 0.3)
- `iou_th` - IoU threshold for NMS (default: 0.4)

**Example:**
```cpp
cnn::CnnModel model("yolo_model.onnx", 0.5f, 0.4f);
```

### infer() Method

```cpp
std::vector<std::vector<float>> infer(const cv::Mat& image)
```

**Parameters:**
- `image` - Input OpenCV image (BGR format)

**Returns:**
Vector of detections, where each detection is a vector containing:
- `[0]` - x coordinate (top-left)
- `[1]` - y coordinate (top-left)
- `[2]` - width
- `[3]` - height
- `[4]` - confidence score (0.0 - 1.0)
- `[5]` - class ID

**Example:**
```cpp
cv::Mat image = cv::imread("test.jpg");
auto detections = model.infer(image);

for (const auto& detection : detections) {
    int x = static_cast<int>(detection[0]);
    int y = static_cast<int>(detection[1]);
    int w = static_cast<int>(detection[2]);
    int h = static_cast<int>(detection[3]);
    float confidence = detection[4];
    int class_id = static_cast<int>(detection[5]);

    std::cout << "Detection: class=" << class_id
              << " conf=" << confidence
              << " box=[" << x << "," << y << "," << w << "," << h << "]" << std::endl;
}
```

### resize() Method

```cpp
std::tuple<cv::Mat, std::pair<int, int>, float> resize(const cv::Mat& image, cv::Size shape)
```

**Parameters:**
- `image` - Input OpenCV image
- `shape` - Original image size

**Returns:**
- `cv::Mat` - Resized and padded image
- `std::pair<int, int>` - Padding offsets (top, left)
- `float` - Scale factor

**Note:** This method is used internally by `infer()` but exposed for advanced usage.

## Usage Example

```cpp
#include <opencv2/opencv.hpp>
#include "cnn_impl.h"

int main() {
    try {
        // Initialize model
        cnn::CnnModel model("yolo_model.onnx", 0.3f, 0.4f);

        // Load image
        cv::Mat image = cv::imread("test_image.jpg");
        if (image.empty()) {
            std::cerr << "Failed to load image" << std::endl;
            return -1;
        }

        // Run inference
        auto detections = model.infer(image);

        // Process results
        std::cout << "Found " << detections.size() << " detections:" << std::endl;

        for (const auto& detection : detections) {
            int x = static_cast<int>(detection[0]);
            int y = static_cast<int>(detection[1]);
            int w = static_cast<int>(detection[2]);
            int h = static_cast<int>(detection[3]);
            float confidence = detection[4];
            int class_id = static_cast<int>(detection[5]);

            // Draw bounding box
            cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

            // Add label
            std::string label = "Class " + std::to_string(class_id) +
                               " (" + std::to_string(confidence) + ")";
            cv::putText(image, label, cv::Point(x, y-10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Save result
        cv::imwrite("output.jpg", image);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

## CMake Integration

```cmake
# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories("path/to/cnn/headers")
include_directories("path/to/onnxruntime/include")

# Link directories
link_directories("path/to/cnn/lib")
link_directories("path/to/onnxruntime/lib")

# Create executable
add_executable(my_app main.cpp)

# Link libraries
target_link_libraries(my_app
    optimized opencv_world4110
    libcnn
    onnxruntime
)
```

## Model Requirements

- ONNX format YOLO model (YOLOv8 recommended)
- Input shape: [1, 3, height, width]
- Output shape: [1, num_classes+4, num_detections]
- Square input size (e.g., 640x640)

## Performance Notes

- CUDA acceleration is automatically enabled if available
- Model warmup is performed during initialization (10 iterations)
- First inference may be slower due to memory allocation
- Subsequent inferences are optimized for speed

## Error Handling

The library may throw exceptions for:
- Invalid model path
- Corrupted ONNX file
- Memory allocation failures
- ONNX Runtime errors

Always wrap model usage in try-catch blocks for robust error handling.

## Class ID Mapping

Class IDs depend on your model training. For COCO-trained models:
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- ... (80 classes total)

Refer to your model's documentation for the complete class mapping.