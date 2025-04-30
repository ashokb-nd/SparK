# get statistics of the fleet edge image pixel values. mean, std
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

dataset_root = "/data4/ashok/training/mlflow/SparK/data"
take_first_n_images = 10000
image_paths = glob.glob(os.path.join(dataset_root, "**/*.jpg"), recursive=True)

# for pixels
running_sum= 0
running_count = 0
running_sq_sum = 0

storage  = {'running_means': [], 'running_stds': [], 'running_counts': []}

# for image_path in image_paths:
import tqdm
for image_path in tqdm.tqdm(image_paths[:take_first_n_images]):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    running_sum += np.sum(image, axis=(0, 1))
    running_sq_sum += np.sum(image**2, axis=(0, 1))
    running_count += image.shape[0] * image.shape[1]
    
    mean = running_sum / running_count
    std = np.sqrt(running_sq_sum / running_count - mean**2)
    storage['running_means'].append(mean)
    storage['running_stds'].append(std)
    storage['running_counts'].append(running_count)


print("Final Mean:", mean)
print("Final Std:", std)


# plot the running means and stds
plt.plot(storage['running_means'], label='Running Means')
plt.plot(storage['running_stds'], label='Running Stds')
plt.xlabel('Image Index')
plt.ylabel('Value')
plt.title('Running Means and Stds')
plt.legend()
plt.show()