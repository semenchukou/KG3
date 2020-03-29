import matplotlib.pyplot as plt
from skimage import data
import glob
import cv2

import thresholds

for img in glob.glob("images/*.jpg"):
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

    block_size = 35
    adaptive_thresh = thresholds.threshold_adaptive(image, block_size, offset=10)
    binary_adaptive = image > adaptive_thresh

    mean_thresh = thresholds.local_mean(image, 3)
    medium_thresh = thresholds.local_medium(image)

    fig, axes = plt.subplots(nrows=4, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()

    ax[0].imshow(image)
    ax[0].set_title('Original')

    ax[1].imshow(binary_adaptive)
    ax[1].set_title('Adaptive thresholding')

    ax[2].imshow(mean_thresh)
    ax[2].set_title('local mean thresholding')

    ax[3].imshow(mean_thresh)
    ax[3].set_title('local medium thresholding')

    for a in ax:
        a.axis('off')

    plt.show()
