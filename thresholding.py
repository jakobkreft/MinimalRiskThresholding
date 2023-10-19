import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_smoothed_histogram(image, L):
    """
    Compute a smoothed histogram of a grayscale image.

    Parameters:
    - image: A grayscale image.
    - L: The number of levels in the image.
    
    Returns:
    - A smoothed histogram of the image.
    """
    # Compute the histogram
    histogram = cv2.calcHist([image], [0], None, [L], [0, L]).flatten()
    histogram /= histogram.sum()  # Normalize the histogram

    # Smooth the histogram
    smoothed_histogram = np.zeros_like(histogram)
    for i in range(2, L - 2):
        smoothed_histogram[i] = np.average(histogram[i-2:i+3], weights=[1,2,3,2,1])

    return smoothed_histogram

def find_two_peak_indices(smoothed_histogram, Dmin):
    """
    Find the indices of the two highest peaks in the histogram.

    Parameters:
    - smoothed_histogram: The smoothed histogram.
    - Dmin: Minimum distance between the two peaks.
    
    Returns:
    - The indices of the two highest peaks.
    """
    sorted_indices = np.argsort(smoothed_histogram)
    peak1 = sorted_indices[-1]
    peak2 = next(idx for idx in reversed(sorted_indices) if abs(idx - peak1) >= Dmin)

    return peak1, peak2

def compute_threshold_parameters(smoothed_histogram, u1, u2):
    """Compute parameters needed for threshold calculation."""
    P1 = smoothed_histogram[:u1].sum()
    P2 = 1 - P1
    
    indices = np.arange(len(smoothed_histogram))
    m1 = np.sum(indices[:u1] * smoothed_histogram[:u1]) / P1
    m2 = np.sum(indices[u1:] * smoothed_histogram[u1:]) / P2

    sigma1_sq = np.sum((indices[:u1] - m1)**2 * smoothed_histogram[:u1]) / P1
    sigma2_sq = np.sum((indices[u1:] - m2)**2 * smoothed_histogram[u1:]) / P2

    return P1, P2, m1, m2, sigma1_sq, sigma2_sq

def compute_optimal_threshold(P1, P2, m1, m2, sigma1_sq, sigma2_sq):
    """Compute the optimal threshold value using the computed parameters."""
    A = sigma1_sq - sigma2_sq
    B = 2 * (m1 * sigma2_sq - m2 * sigma1_sq)
    C = m2**2 * sigma1_sq - m1**2 * sigma2_sq + 2 * sigma1_sq * sigma2_sq * np.log((P1 * sigma2_sq) / (P2 * sigma1_sq))
    
    t1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    t2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)

    return t2

def plot_histogram_with_peaks_threshold(histogram, u1, u2, t):
    """Plot the histogram with peaks and threshold line."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(histogram)), histogram, width=1)
    plt.plot(u1, histogram[u1], 'ro')
    plt.plot(u2, histogram[u2], 'ro')
    plt.axvline(x=t, color='k', linestyle='--')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title(f"Peaks at {u1}, {u2} and Threshold at {t}")
    plt.savefig('histogram_with_peaks_threshold.png')
    plt.show()

def threshold_image(image, t):
    """Threshold the image based on the calculated threshold."""
    return np.where(image < t, 0, 255).astype(np.uint8)

def display_images(original: np.array, thresholded: np.array):
    """Display the original and thresholded images side by side."""
    cv2.imshow('Original Image', original)
    cv2.imshow('Thresholded Image', thresholded)

def main(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"The file at path {image_path} could not be found or is not a valid image file.")

    smoothed_histogram = compute_smoothed_histogram(image, L=256)
    u1, u2 = find_two_peak_indices(smoothed_histogram, Dmin=30)
    
    if u1 > u2:
        u1, u2 = u2, u1

    params = compute_threshold_parameters(smoothed_histogram, u1, u2)
    threshold = compute_optimal_threshold(*params)
    
    thresholded_image = threshold_image(image, threshold)
    
    plot_histogram_with_peaks_threshold(smoothed_histogram, u1, u2, threshold)
    display_images(image, thresholded_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main('tat.jpg')
    except Exception as e:
        print(str(e))