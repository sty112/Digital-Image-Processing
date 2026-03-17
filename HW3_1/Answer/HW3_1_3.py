from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_histogram(image):
    cv2_image = np.array(image)
    z_q = np.arange(0,256)
    print(np.sum(z_q ** 0.4))
    c = 1/np.sum(z_q ** 0.4)
    print(c)
    p_z = c * (z_q ** 0.4)
    target_cdf = np.round(np.cumsum(p_z) / np.sum(p_z) * 255).astype(np.uint8)
    img_equi = histogram_equi(cv2_image)
    img_match = histogram_match(img_equi, target_cdf)
    hist, bins = np.histogram(img_match, bins=256, range=(0, 256))
    plt.subplots()
    plt.bar(bins[:-1],hist)
    plt.title('Histogram Matched')
    plt.xlabel('Intensity')
    plt.ylabel('pixel number')
    plt.show()
    # cv2.imshow("Equalized", img_equi)
    # cv2.imshow("Matched", img_match)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def histogram_equi(image):
    image_equi = cv2.equalizeHist(image)
    return image_equi

def histogram_match(image_equi, target_cdf):
    hist, _ = np.histogram(image_equi.flatten(), bins=256, range=(0,256))
    equi_cdf = np.round(hist.cumsum() / hist.sum() * 255).astype(np.uint8)
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(target_cdf - equi_cdf[i])
        mapping[i] = np.argmin(diff)
    img_match = mapping[image_equi]
    return img_match

def main():
    path = "aerial_view.tif"
    img = Image.open(path)
    show_histogram(img)

if __name__ == "__main__":
    main()
