import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as PILImage

def freq_filter(Image):
    f = np.fft.fft2(Image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = Image.shape
    mask = np.ones_like(fshift, dtype=np.complex128)
    h_filter, w_filter = mask.shape
    coords = [(343, 379), (343, 393)]
    for (r, c) in coords:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = 0
            r_sym = (h - r) % h
            c_sym = (w - c) % w
            mask[r_sym, c_sym] = 0
    fshift_filtered = fshift * mask
    magnitude = np.abs(fshift_filtered)
    magnitude_log = np.log1p(magnitude)
    mag_norm = magnitude_log - magnitude_log.min()
    denom = (mag_norm.max() + 1e-12)
    mag_norm = mag_norm / denom
    magnitude_uint8 = (mag_norm * 255).astype(np.uint8)
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    recon = img_back - img_back.min()
    recon = recon / (recon.max() + 1e-12)
    recon_uint8 = (recon * 255).astype(np.uint8)

    return recon_uint8, magnitude_uint8

def main():
    path = 'text-sineshade.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filtered_img, fshift = freq_filter(img)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()