from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_filter(image,width,std_width,height,std_height):
    kernel_width = cv2.getGaussianKernel(width,std_width)
    kernel_height = cv2.getGaussianKernel(height,std_height)
    gaussian = np.outer(kernel_height, kernel_width)
    mask = cv2.filter2D(image,-1,gaussian)
    fil_img = image/mask
    return mask, fil_img

def main():
    path = "N1.bmp"
    width = 160
    height = 480
    std_width = 40
    std_height = 120
    img = Image.open(path)
    cv2_image = np.array(img)
    copy_img = cv2_image.copy()
    mask,filtered_image = gaussian_filter(cv2_image,width,std_width,height,std_height)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    imgs = [(copy_img, "Original Image"), (mask, "Mask"), (filtered_image, "Filtered Image")]

    for ax, (im, title) in zip(axes, imgs):
        # normalize for display if needed
        if im.dtype != np.uint8:
            im_disp = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            im_disp = im
        ax.imshow(im_disp, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()