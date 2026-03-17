import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as PILImage

def Find_Word_Edge(Image):
    copy_img = Image.copy()
    contours, _ = cv2.findContours(copy_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = copy_img.shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w == width or y + h == height:
            cv2.drawContours(copy_img, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    return copy_img
    
def main():
    path = 'text.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    final_img = Find_Word_Edge(img)
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