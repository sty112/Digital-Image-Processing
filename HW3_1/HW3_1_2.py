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
    image_equi = histogram_equi(cv2_image)
    hist_equi, bins_equi = np.histogram(image_equi, bins=256, range=(0, 256))
    plt.subplots()
    plt.bar(bins_equi[:-1],hist_equi)
    plt.title('Histogram Equalized')
    plt.xlabel('Intensity')
    plt.ylabel('pixel number')
    plt.show()
    # cv2.imshow( 'Equalized Image', image_equi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def histogram_equi(image):
    image_equi = cv2.equalizeHist(image)
    return image_equi 

def main():
    path = "aerial_view.tif"
    img = Image.open(path)
    show_histogram(img)

if __name__ == "__main__":
    main()