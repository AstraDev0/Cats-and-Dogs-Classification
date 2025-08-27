# Cats and Dogs Classification

A simple image classification project using PCA, HOG, and Sobel features with a Nearest Neighbor classifier. Developed as part of a learning exercise by Matthis Brocheton.

## Requirements

Ensure you have Python 3 installed. Install the necessary packages using pip:

```bash
pip install numpy opencv-python scikit-learn scikit-image
````

## Dataset Structure

Prepare your dataset with the following folder structure:

```
data/
├── train/
│   ├── Cat/
│   └── Dog/
└── test/
    ├── Cat/
    └── Dog/
```

## How to Run

Run the classification script from the terminal:

```bash
python image_classification.py --train data/train --test data/test --n_components 50 --target_size 128 128
```

### Arguments

* `--train`: Path to the training dataset
* `--test`: Path to the testing dataset
* `--n_components`: Number of PCA components (default: 50)
* `--target_size`: Resize images to specified width and height (default: 128x128)

## Output

After execution, the script will output the classification accuracy in the terminal, for example:

```
Accuracy: 58.50%
```

## Features Used

* **Sobel Filters**: Edge detection
* **PCA**: Dimensionality reduction
* **HOG (Histogram of Oriented Gradients)**: Captures texture and shape
* **SAD (Sum of Absolute Differences)**: Image comparison for Nearest Neighbor classification

## Key Challenges

* Flattening multi-dimensional feature maps for the classifier
* Optimizing computation-heavy operations (Sobel, PCA)
* Efficient label assignment during data loading

## Conclusion

This project provided hands-on experience with classic image processing and machine learning techniques. It offered practice in feature engineering and basic classification algorithms, achieving a final accuracy of **58.50%**.
