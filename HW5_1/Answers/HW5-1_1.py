import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
import cv2

def cutting_off(H, r):
    H_cutting = H.copy()
    for u in range(M):
        for v in range(N):
            if ((u - M // 2) ** 2 + (v - N // 2) ** 2) > (r ** 2):
                H_cutting[u, v] = H[M // 2, N // 2]
    return H_cutting

img = cv2.imread('Fig5.25.jpg', 0)
M, N = img.shape

G = np.fft.fft2(img)
G_shift = np.fft.fftshift(G)
H_shift = np.zeros((M, N))
k = 0.002
cutting_r = 40
wiener_k = 0.04

for u in range(M):
    for v in range(N):
        exponent = -k * (((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6))
        H_shift[u, v] = np.exp(exponent)

img_FFT = fftshift(fftn(img))

H_cutting = cutting_off(H_shift, cutting_r)
img_inverse_FFT = img_FFT / H_cutting
img_inverse = np.abs(ifftn(img_inverse_FFT))

K_shift = H_shift ** 2
img_wiener_FFT = ((K_shift/(K_shift + wiener_k)) / H_shift) * img_FFT
img_wiener = np.abs(ifftn(img_wiener_FFT))

plt.figure(figsize = (10, 4))
plt.subplot(1, 3, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(img_inverse, cmap = 'gray')
plt.title('After inverse filter'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(img_wiener, cmap = 'gray')
plt.title('After Wiener filter'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig("Fig5.25_result.png")
plt.show()
