# Gas-Station-Price-Table-OCR
This project showcases a comprehensive method to extract and recognize information from gas station price tables using various image processing techniques and a custom-trained machine learning model.

## Overview

This project is designed to process images of gas station price tables, extract text and numbers, and accurately recognize and display the prices. The method leverages a custom-trained machine learning model on the SHVN Dataset, along with various image processing techniques to handle different table layouts and formats.
The model recognizes standard formats and is a work in progress, currently identifying the majority of gas station tables.

## System Overview

This project integrates multiple components into a cohesive system for extracting and recognizing prices from gas station tables. The pipeline consists of the following steps:

1. **Image Preprocessing**:
   - Enhances image quality (contrast adjustment, resizing, etc.).
   - Applies contour detection to locate regions of interest.

2. **Optical Character Recognition (OCR)**:
   - Uses PaddleOCR to extract text from identified regions.

3. **Custom ML Model**:
   - Processes segmented images to predict numeric values using a TensorFlow Lite model.

4. **Post-Processing**:
   - Matches recognized text with detected numbers.
   - Validates and formats the final output to ensure accuracy.



<p align="center">
    <img src="https://raw.githubusercontent.com/lodist/Gas-Station-Price-Table-OCR/main/sample%20pictures/system_architecture_diagram.png" width="100%" />
</p>





## Image examples

<p align="center">
  <img src="https://github.com/lodist/Gas-Station-Price-Table-OCR/assets/75701170/f41147ce-30b6-4f48-952f-e8ffdc8bd987" width="20%" />
  <img src="https://github.com/lodist/Gas-Station-Price-Table-OCR/assets/75701170/ddba261d-f7c0-495a-b55f-80b7ea75e5b5" width="20%" />
  <img src="https://github.com/lodist/Gas-Station-Price-Table-OCR/assets/75701170/fe0b22a3-76b8-4243-8f81-03e572978bd2" width="13%" />
</p>

### Input Requirements

- Supported Formats: `.png`, `.jpg`
- Recommended Resolution: 300 DPI or higher for best accuracy.


### Key Features

- **Machine Learning**: Utilizes convolutional neural networks (CNNs) to train a custom model for recognizing numeric and textual information from images.
- **Image Preprocessing**: Enhances image quality by adjusting contrast, converting to grayscale, and resizing images for optimal processing.
- **Contour Detection and Processing**: Identifies and processes contours to segment regions of interest, isolating specific areas containing price information.
- **Optical Character Recognition (OCR)**: Uses PaddleOCR to extract text from images, converting visual information into readable and actionable data.
- **Image Enhancement**: Improves image resolution and clarity to aid in accurate text and number extraction.
- **Model Prediction**: Employs the trained model to predict prices from the segmented regions, ensuring precise recognition.
- **Post-Processing**: Matches and validates extracted text and numbers to generate accurate and reliable results.
- **Adaptive Learning**: Continuously improves the model based on new data inputs, enabling it to handle various and evolving price table formats effectively.

## Why This Project?

This repository showcases how to integrate machine learning models, OCR, and image processing into a complete solution. It demonstrates:

- **Seamless ML Integration**: Combining PaddleOCR and a TensorFlow Lite model into a single pipeline.
- **Error Resilience**: Robust handling of poor-quality images and unexpected input formats.
- **Scalable Design**: Designed for batch processing and easy extension with additional OCR models.
- **Lightweight Deployment**: The system is optimized to be lightweight, enabling seamless integration as part of a mobile or web app.


## Extensibility

This system is designed to be modular and easily extendable:

- **Replaceable OCR Engine**: Swap out PaddleOCR with another OCR library like Tesseract or EasyOCR.
- **Model Customization**: Retrain the TensorFlow Lite model on custom datasets for other use cases.
- **Flexible Output Formats**: Extend post-processing to generate outputs in JSON, Excel, or other formats.
- **Mobile-Friendly Design**: Its lightweight architecture makes it suitable for deployment within apps, ensuring efficient on-device processing.


## Installation

To get started, follow these steps:

1. Open **Command Prompt** as an administrator:
   - Press `Win + S`, search for **Command Prompt**.
   - Right-click on **Command Prompt** and select **Run as administrator**.
  
2. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/lodist/Gas-Station-Price-Table-OCR.git
cd Gas-Station-Price-Table-OCR
pip install -r requirements.txt
```
## Usage

To use the code, simply call the `process_image` function with the path to your image:

```python
from OCR_gas_station_table import process_image

result = process_image(image_path='path/to/your/image.png')
print(result)
```

### Example

```python
result = process_image(image_path='sample_gas_station_image.png')
print(result)
{'Bf 95': '1.88', 'Bf 98': '1.96', 'Diesel': '1.95'}
```

## Model Training

The model used in this project is trained on the SHVN Dataset and then converted to TensorFlow Lite format. If you wish to train your own model or retrain the existing model, you can follow these steps:

- **Prepare the Dataset**: Download and preprocess the SHVN Dataset.
- **Train the Model**: Use the training script provided in the repository to train the model.
- **Convert to TFLite**: Convert the trained model to TensorFlow Lite format for efficient deployment.


## Execution Performance

After the initial deployment, which takes approximately 30 seconds, the script processes each image in **2-3 seconds** to produce results.
This performance ensures quick and efficient processing for practical applications.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the [LICENSE](LICENSE) file for more details.


## Contact

For any inquiries or commercial use, please contact me at lorisdistefano@protonmail.com.
