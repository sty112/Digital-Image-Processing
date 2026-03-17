import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_atmospheric_degradation(M, N, k=0.002):
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            exponent = -k * (((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6))
            H[u, v] = np.exp(exponent)
    return H

def apply_circular_cutoff(H, radius, M, N):
    H_cutoff = H.copy()
    center_value = H[M // 2, N // 2]
    
    for u in range(M):
        for v in range(N):
            distance_squared = (u - M // 2) ** 2 + (v - N // 2) ** 2
            if distance_squared > radius ** 2:
                H_cutoff[u, v] = center_value
    
    return H_cutoff

def apply_inverse_filter(image, degradation_func):
    image_fft = np.fft.fft2(image)
    image_fft_shifted = np.fft.fftshift(image_fft)
    restored_fft = image_fft_shifted / degradation_func
    restored = np.abs(np.fft.ifft2(np.fft.ifftshift(restored_fft)))
    
    return restored

def apply_wiener_filter(image, degradation_func, K=0.04):
    image_fft = np.fft.fft2(image)
    image_fft_shifted = np.fft.fftshift(image_fft)
    H_squared = degradation_func ** 2
    wiener_coefficient = H_squared / (H_squared + K)
    restored_fft = (wiener_coefficient / degradation_func) * image_fft_shifted
    restored = np.abs(np.fft.ifft2(np.fft.ifftshift(restored_fft)))
    return restored

def main():
    image = cv2.imread('Fig5.25.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image 'Fig5.25.jpg'")
        print("Please make sure the image is in the same directory as the script.")
        return
    
    M, N = image.shape
    print(f"Image loaded: {M}x{N} pixels")
    print("\nCreating atmospheric turbulence degradation model...")
    k_turbulence = 0.002
    H = create_atmospheric_degradation(M, N, k=k_turbulence)
    print("Applying inverse filter with circular cutoff...")
    cutoff_radius = 40
    H_cutoff = apply_circular_cutoff(H, cutoff_radius, M, N)
    restored_inverse = apply_inverse_filter(image, H_cutoff)
    print("Applying Wiener filter...")
    wiener_K = 0.04
    restored_wiener = apply_wiener_filter(image, H, K=wiener_K)
    print("\nDisplaying results...")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original (Degraded)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(restored_inverse, cmap='gray')
    plt.title(f'Inverse Filter\n(radius={cutoff_radius})')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(restored_wiener, cmap='gray')
    plt.title(f'Wiener Filter\n(K={wiener_K})')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    output_filename = 'Fig5.25_result.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Result saved as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()