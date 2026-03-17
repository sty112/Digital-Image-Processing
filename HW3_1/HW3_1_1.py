from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_histogram(image):
    cv2_image = np.array(image)
    print(cv2_image.shape)
    img_len = len(cv2_image)
    pixel_count = 0
    y = 0
    arr = []
    hist,bins = np.histogram(cv2_image, bins=256, range=(0, 256))
    fig, ax = plt.subplots()
    ax.bar(bins[:-1],hist)
    plt.title('Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('pixel number')
    plt.show()   
    # cv2.imshow( 'Image', cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    path = "aerial_view.tif"
    img = Image.open(path)
    show_histogram(img)

if __name__ == "__main__":
    main()