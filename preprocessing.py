import os
import cv2
from image_processing import func

# Create directories if they don't exist
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
for directory in [data_dir, train_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set the maximum number of files to use for training
num_files_to_train = float('inf')  # Set to infinity for maximum

# Iterate through each class directory in the train directory
for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_train_dir) or not os.path.isdir(class_test_dir):
        continue
    
    # Ensure train and test directories for the class exist
    for directory in [class_train_dir, class_test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Count files processed
    files_processed_train = 0

    # Iterate through files in the class directory for training
    for file_name in os.listdir(class_train_dir):
        file_path = os.path.join(class_train_dir, file_name)
        processed_image = func(file_path)

        # Save processed image to train directory
        cv2.imwrite(file_path, processed_image)

        # Increment count
        files_processed_train += 1

    print(f"Train: {class_name}, Files Processed: {files_processed_train}")


# Iterate through each class directory in the test directory
for class_name in os.listdir(test_dir):
    class_test_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_test_dir):
        continue

    # Count files processed
    files_processed_test = 0

    # Iterate through files in the class directory for testing
    for file_name in os.listdir(class_test_dir):
        file_path = os.path.join(class_test_dir, file_name)
        processed_image = func(file_path)

        # Save processed image to test directory
        cv2.imwrite(file_path, processed_image)

        # Increment count
        files_processed_test += 1

    print(f"Test: {class_name}, Files Processed: {files_processed_test}")

print("Processing complete.")