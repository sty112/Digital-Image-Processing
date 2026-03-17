import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def butterworth_notch_reject(shape, centers, D0, n=2, relative_to_center=True):
    """
    Create a Butterworth Notch Reject filter
    shape: (rows, cols) of the image
    centers: list of (u, v) coordinates for notch centers
    D0: radius of the notch (cutoff frequency)
    n: order of the filter
    relative_to_center: if True, centers are relative to center; if False, absolute positions
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate matrices
    if relative_to_center:
        u = np.arange(rows)
        v = np.arange(cols)
        u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    else:
        # Use absolute coordinates
        u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Start with all ones (pass everything)
    H = np.ones((rows, cols))
    
    # For each notch center, create a notch filter
    for center_u, center_v in centers:
        # Distance from this notch center
        Dk = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        
        if relative_to_center:
            # Distance from symmetric point (since FFT is symmetric)
            Dk_sym = np.sqrt((u + center_u)**2 + (v + center_v)**2)
            # Butterworth notch reject formula for this notch pair
            Hk = (1 / (1 + (D0 / (Dk + 1e-6))**(2*n))) * (1 / (1 + (D0 / (Dk_sym + 1e-6))**(2*n)))
        else:
            # Just single notch at absolute position
            Hk = 1 / (1 + (D0 / (Dk + 1e-6))**(2*n))
        
        # Multiply with existing filter
        H = H * Hk
    
    return H

def butterworth_highpass_filter(shape, cutoff, order=2):
    """
    Create a Butterworth high-pass filter to reject high frequencies (moiré pattern)
    This is actually a low-pass filter that keeps low frequencies
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate matrices
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Distance from center
    D = np.sqrt(u**2 + v**2)
    
    # Butterworth low-pass filter (keeps low frequencies, rejects high frequencies)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    
    return H

def freq_filter(image):
    image = image.astype(np.float64) / 255.0
    # Compute the discrete Fourier Transform of the image
    image_fft = np.fft.fft2(image)

    # Shift the zero-frequency component to the center of the spectrum
    fourier = np.fft.fftshift(image_fft)
    
    # Calculate the magnitude of the Fourier Transform
    magnitude = 20*np.log(np.abs(fourier) + 1)  # Add 1 to avoid log(0)
    
    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    # Get center coordinates
    center_row = image.shape[0] // 2
    center_col = image.shape[1] // 2
    
    # Define 8 notch locations (ABSOLUTE positions from your image)
    notch_centers = [
        (43, 55), 
        (40, 111), 
        (84, 55),
        (81, 111),   
        (165, 57),    
        (161, 114),   
        (207, 58), 
        (204, 114),  
    ]
    
    # Butterworth notch filter parameters
    D0 = 10  # Radius of each notch (adjust: larger = wider notch)
    n = 2    # Order (higher = sharper notch edges)
    
    # Create the notch filter with absolute coordinates
    butterworth_filter = butterworth_notch_reject(image.shape, notch_centers, D0, n, relative_to_center=False)
    
    # Apply the filter
    filter_freq = fourier * butterworth_filter
    
    # Inverse FFT
    shifted_filter_freq = np.fft.ifftshift(filter_freq)
    filter_img = np.fft.ifft2(shifted_filter_freq)
    
    # Take absolute value to get real image from complex result
    result_img = np.abs(filter_img)

    # Calculate magnitude of filtered frequency domain for visualization
    filtered_magnitude = 20*np.log(np.abs(filter_freq) + 1)
    filtered_magnitude = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    # Normalize Butterworth filter for visualization
    filter_vis = cv2.normalize(butterworth_filter, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Display results using matplotlib
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Original Magnitude')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(filter_vis, cmap='gray')
    plt.title(f'Butterworth Notch Filter (D0={D0}, n={n})')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(result_img, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(filtered_magnitude, cmap='gray')
    plt.title('Filtered Magnitude')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Show the difference
    plt.imshow(image - result_img, cmap='gray')
    plt.title('Removed Noise')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('car_butterworth_filtered.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Also save the filtered image
    cv2.imwrite('car_filtered_result.png', result_img * 255)
    # cv2.imshow('Original Magnitude', magnitude)
    # cv2.waitKey(0)
    
def main():
    img= cv2.imread('C:\\Users\\sty123\\Desktop\\hw\\image_processing\\HW4_2\\car-moire-pattern.tif', cv2.IMREAD_GRAYSCALE)
    freq_filter(img)
    
    

if __name__ == "__main__":
    main()