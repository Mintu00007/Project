# Stock Marcket Chart Pattern Identifier using Machine Learning
## Project Overview: Stock Market Chart Pattern Identifier Using CNN

### Introduction

In the realm of financial markets, chart patterns like "Double Top" and "Double Bottom" are crucial for predicting future price movements. This project aims to create an automated system using Convolutional Neural Networks (CNN) to identify such patterns in stock market charts, thereby aiding traders and financial analysts in making informed decisions.

### Project Goals

1. **Automate Pattern Detection**: Develop a CNN-based model to detect traditional chart patterns in stock market charts.
2. **Enhance Trading Strategies**: Provide traders with a tool to automate the detection of critical chart patterns.

### Data Collection and Preprocessing

The project starts with collecting stock market chart images, which are then processed to extract relevant features and sub-images. The preprocessing steps include:

1. **Image Upload**: Users upload stock market chart images through a web interface.
2. **Image Conversion**: Convert uploaded images to a format compatible with OpenCV.
3. **Sub-Image Extraction**: Divide the main image into smaller blocks or sub-images for detailed analysis.

### Extracting Sub-Images

The function `extract_sub_images(img, num_rows, num_cols)` splits the uploaded image into smaller segments to analyze different parts of the chart individually. This involves:

1. **Grid Division**: Dividing the image into a grid based on specified rows and columns.
2. **Coordinate Calculation**: Calculating coordinates to define sub-image boundaries.
3. **Sub-Image Extraction**: Extracting and storing these segments for further analysis.

### Contour Detection and Filtering

To process only relevant parts of the image, we use contour detection to filter out blank or irrelevant sub-images. The function `detect_contours_and_filter(sub_images)` handles this task by:

1. **Grayscale Conversion**: Converting each sub-image to grayscale.
2. **Noise Reduction**: Applying Gaussian blur to reduce noise.
3. **Edge Detection**: Using the Canny edge detection method to find contours.
4. **Contour Analysis**: Analyzing the detected contours to determine if the sub-image contains significant patterns.
5. **Filtering**: Retaining only those sub-images with meaningful contours.

### Model Architecture and Training

The core of the project is a CNN trained to recognize patterns in the sub-images. The model architecture includes:

1. **Convolutional Layers**: Extracting features from images through convolution operations.
2. **Pooling Layers**: Reducing the dimensionality of features while retaining essential information.
3. **Fully Connected Layers**: Interpreting features and making predictions.

The training process involves:

1. **Dataset Preparation**: Creating a labeled dataset of sub-images.
2. **Model Training**: Training the CNN to accurately detect patterns.
3. **Validation and Testing**: Validating the model's performance and fine-tuning it to achieve desired accuracy.

### Prediction and Visualization

The trained model predicts patterns in new chart images. The function `predict_pattern(model, sub_image)` processes each sub-image and outputs the detected pattern. This involves:

1. **Image Resizing**: Resizing the sub-image to match the CNN input size.
2. **Normalization**: Normalizing pixel values to improve model performance.
3. **Prediction**: Using the trained model to predict patterns.
4. **Result Interpretation**: Classifying the pattern as "Double Top", "Double Bottom", or "No Pattern Found".

### User Interface and Integration

The project includes a user-friendly interface where users can upload images and view results. The interface:

1. **Image Upload**: Allows users to upload stock market chart images.
2. **Pattern Display**: Displays the original image and detected patterns in sub-images.
3. **Prediction Results**: Shows prediction results with appropriate labels.

### Conclusion

This project applies CNNs in financial market analysis, automating the detection of crucial chart patterns. By combining image processing and deep learning, it provides a valuable tool for traders and analysts. Future enhancements could include expanding the pattern library, real-time analysis, improving accuracy, and integrating with trading platforms, thereby significantly impacting how traders analyze stock market data.
