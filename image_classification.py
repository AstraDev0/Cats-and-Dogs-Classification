import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.feature import hog
import os
import argparse

def detect_label(img_path):
    if "Cat" in img_path:
        return 0
    elif "Dog" in img_path:
        return 1
    return -1

def sobel_features(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.flatten()

def hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def pca_features(images, n_components):
    flat_images = np.array([img.flatten() for img in images])
    pca = PCA(n_components=n_components)
    return pca.fit_transform(flat_images)

def get_image_paths(path):
    images = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                images.append(os.path.join(folder_path, file))
    return images

def load_images(paths, target_size):
    images, labels = [], []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(detect_label(path))
    return images, labels

def extract_features(images, n_components):
    sobel_list = [sobel_features(img) for img in images]
    hog_list = [hog_features(img) for img in images]
    pca_list = pca_features(images, n_components)
    return np.array(sobel_list), np.array(pca_list), np.array(hog_list)

def sad_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def nearest_neighbor(train_feat, train_labels, test_feat):
    predictions = []
    for test_vec in test_feat:
        distances = [sad_distance(test_vec, train_vec) for train_vec in train_feat]
        nearest_idx = np.argmin(distances)
        predictions.append(train_labels[nearest_idx])
    return predictions

def accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return (correct / len(labels)) * 100

def run(train_path, test_path, n_components, target_size):
    train_paths = get_image_paths(train_path)
    test_paths = get_image_paths(test_path)

    train_imgs, train_labels = load_images(train_paths, target_size)
    test_imgs, test_labels = load_images(test_paths, target_size)

    sobel_train, pca_train, hog_train = extract_features(train_imgs, n_components)
    sobel_test, pca_test, hog_test = extract_features(test_imgs, n_components)

    train_features = np.concatenate([sobel_train, pca_train, hog_train], axis=1)
    test_features = np.concatenate([sobel_test, pca_test, hog_test], axis=1)

    preds = nearest_neighbor(train_features, train_labels, test_features)
    print(f"Accuracy: {accuracy(preds, test_labels):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cats and Dogs Classifier')
    parser.add_argument('--train', type=str, required=True, help='Training dataset path')
    parser.add_argument('--test', type=str, required=True, help='Testing dataset path')
    parser.add_argument('--n_components', type=int, default=50, help='PCA components')
    parser.add_argument('--target_size', type=int, nargs=2, default=(128,128), help='Resize target size')

    args = parser.parse_args()
    run(args.train, args.test, args.n_components, tuple(args.target_size))
