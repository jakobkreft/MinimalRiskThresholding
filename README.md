
## Minimal Risk Thresholding Implementation

This Python program applies the Minimal Risk Thresholding algorithm.

### Required Packages

- **NumPy:** For numerical operations and handling image data as arrays.
- **OpenCV:** For reading and manipulating image files.
- **Matplotlib:** For plotting the histogram and visualizing the images.

You can install them using pip:
```bash
pip install numpy opencv-python matplotlib
```

### Usage
Simply run the Python file with a specified image path:
```bash
python thresholding.py
```

### Algorithm

The Minimal Risk Thresholding algorithm operates in the following manner:

1. **Smoothed Histogram Calculation:** 
2. **Peak Identification:** Identifies two distinct peaks (Apart at least the `Dmin` distance)
3. **Optimal Threshold Calculation:** Computes an optimal threshold value that minimally risks misclassifying pixels. It uses a quadratic equation derived from the statistical parameters of the smoothed histogram.
4. **Image Thresholding:** Applies the calculated threshold to the image
5. **Visualization:** The program also visualizes the original grayscale image, the thresholded image, and the smoothed histogram with marked peaks and threshold.
