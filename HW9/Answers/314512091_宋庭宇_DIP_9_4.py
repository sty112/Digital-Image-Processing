import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as PILImage

def Find_Word_Edge(Image):
    copy_img = Image.copy()
    h, w = copy_img.shape
    fft_img = np.fft.fft2(copy_img.astype(np.float32))
    fshift = np.fft.fftshift(fft_img)
    filter_mask = np.ones_like(copy_img, dtype=np.complex128)
    h_filter, w_filter = filter_mask.shape
    h_mid, w_mid = h_filter // 2, w_filter // 2
    for i in range(1,h_mid):
        for j in range(w_mid-1,w_mid+2):
            if i > h_mid - 5 and i < h_mid - 0:
                filter_mask[i, j] = 0
            if i+h_mid < h_mid + 5 and i+h_mid > h_mid + 0:
                filter_mask[i+h_mid, j] = 0
    for j in range(1,w_mid):
        if j > w_mid - 5 and j < w_mid - 0:
            filter_mask[h_mid, j] = 0
        if j+w_mid < w_mid + 5 and j+w_mid > w_mid + 0:
            filter_mask[h_mid, j+w_mid] = 0
    filter_freq = fshift * filter_mask
    magnitude = np.abs(filter_freq)
    magnitude_log = np.log1p(magnitude)
    mag_norm = magnitude_log - magnitude_log.min()
    denom = (mag_norm.max() + 1e-12)
    mag_norm = mag_norm / denom
    magnitude_uint8 = (mag_norm * 255).astype(np.uint8)
    f_ishift = np.fft.ifftshift(filter_freq)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    recon = img_back - img_back.min()
    recon = recon / (recon.max() + 1e-12)
    recon_uint8 = (recon * 255).astype(np.uint8)
    return recon_uint8, magnitude_uint8
    
def main():
    path = 'text-spotshade.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed, spectrum = Find_Word_Edge(img)
    ret, binary_img = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Processed Image')
    plt.imshow(processed, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Magnitude Spectrum')
    plt.imshow(spectrum, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()