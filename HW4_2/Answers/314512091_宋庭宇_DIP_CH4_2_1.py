import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def freq_filter(image):
    image = image.astype(np.float64) / 255.0
    # Compute the discrete Fourier Transform of the image
    image_fft = np.fft.fft2(image)

    # Shift the zero-frequency component to the center of the spectrum
    fourier = np.fft.fftshift(image_fft)
    
    # # calculate the magnitude of the Fourier Transform
    # magnitude = 20*np.log(cv2.magnitude(fourier[:,:,0],fourier[:,:,1]))

    # # Scale the magnitude for display
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    filter = np.ones_like(image, dtype=np.complex128)
    filter[387, 475] = 0
    filter[437, 525] = 0

    filter_freq = fourier * filter
    shifted_filter_freq = np.fft.ifftshift(filter_freq)
    filter_img = np.fft.ifft2(shifted_filter_freq)
    
    # Take absolute value to get real image from complex result
    result_img = np.abs(filter_img)

    # Save the filtered image
    cv2.imwrite('astronaut_filtered.png', result_img * 255)

    # Calculate magnitude of filtered frequency domain for visualization
    # filtered_magnitude = 20*np.log(cv2.magnitude(filter_freq[:,:,0], filter_freq[:,:,1]))
    # filtered_magnitude = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # cv2.imshow('Original Magnitude', magnitude)
    # cv2.imshow('Filtered Magnitude', filtered_magnitude)
    
    # cv2.imshow('Filtered Image', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Display the images and Fourier spectra
    plt.figure(figsize=(12, 10))
    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')

    # Original Fourier Spectrum
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(fourier), cmap='gray', vmin=0, vmax=5000)
    plt.axis('off')
    plt.title('Original Fourier Spectrum')

    # Result Image
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(result_img), cmap='gray')
    plt.axis('off')
    plt.title('Result Image')

    # Result Fourier Spectrum
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(filter_freq), cmap='gray', vmin=0, vmax=5000)
    plt.title('Result Fourier Spectrum')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("astronaut_processed.png")

def main():
    img= cv2.imread('C:\\Users\\sty123\\Desktop\\hw\\image_processing\\HW4_2\\astronaut-interference.tif', cv2.IMREAD_GRAYSCALE)
    freq_filter(img)
    
if __name__ == "__main__":
    main()