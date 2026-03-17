from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_process(image):
    c = 1
    k = 2
    img_copy = image.copy()
    laplacian_kernel = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
    laplacian_image = cv2.filter2D(img_copy, -1, laplacian_kernel)
    sharpened_image = img_copy + k * laplacian_image.astype(np.uint8)
    sobel_x = cv2.Sobel(img_copy, cv2.CV_32F, 1, 0, ksize = 3)
    sobel_y = cv2.Sobel(img_copy, cv2.CV_32F, 0, 1, ksize = 3)
    sobel_image = cv2.convertScaleAbs(np.abs(sobel_x) + np.abs(sobel_y))
    sobel_box_image = cv2.boxFilter(np.sqrt(sobel_x**2 + sobel_y**2), -1, (5, 5))
    mask = laplacian_image * sobel_box_image
    sharp_img = cv2.add(img_copy, mask.astype(np.uint8))
    image_power = c * np.power(sharp_img,0.5)
    return laplacian_image, sharpened_image, sobel_image, sobel_box_image, mask, sharp_img, image_power


def main():
    # path = "C:\\Users\\sty123\\Desktop\\hw\\image_processing\\HW3_4\\Bodybone.bmp"
    path = "C:\\Users\\sty123\\Desktop\\hw\\image_processing\\HW3_4\\fish.jpg"
    image = Image.open(path)
    cv2_image = np.array(image)
    laplacian_image, sharpened_image, sobel_image, sobel_box_image, mask, sharp_img, image_power = image_process(cv2_image)
    
    # Normalize the output for display (Laplacian can have negative values)
    image_power = cv2.normalize(image_power, None, 0, 255, cv2.NORM_MINMAX)
    image_power = np.uint8(image_power)

    images = [cv2_image, laplacian_image, sharpened_image, sobel_image, sobel_box_image, mask, sharp_img, image_power]
    titles = ['Original Image', 'Laplacian Image', 'Laplacian Sharpened Image', 'Sobel Gradient', 
              'Sobel Smooth', 'Masked Image', 'Sharpened mask image', 'Enhanced Image']

    plt.figure(figsize=(12, 8))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 
