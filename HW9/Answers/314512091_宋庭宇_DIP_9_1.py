import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as PILImage

def Morpho(Image):
    copy_img = Image.copy()
    ret, binary_img = cv2.threshold(copy_img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
    return dilated_img
    
def main():
    path = 'text-broken.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    final_img = Morpho(img)
    cv2.imwrite('text-fix.png', cv2.bitwise_not(final_img))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)   
    plt.title('Processed Image')
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()