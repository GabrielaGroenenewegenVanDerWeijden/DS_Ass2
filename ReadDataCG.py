import os
import numpy as np
import kagglehub
import cv2

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def get_training_data(data_dir):
    data = []

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) / 255.0  # Normalize

                data.append((resized_arr, class_num))
            except Exception as e:
                print(f"Error processing {img}: {e}")


    return np.array(data, dtype=object)


# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = get_training_data('Data/chest_xray/chest_xray/train')
test = get_training_data('Data/chest_xray/chest_xray/test')
val = get_training_data('Data/chest_xray/chest_xray/val')