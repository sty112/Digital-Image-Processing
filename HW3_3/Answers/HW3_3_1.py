from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_filter(image,dim,std):
    kernel = cv2.getGaussianKernel(dim,std)
    gaussian = np.outer(kernel, kernel.transpose())
    mask = cv2.filter2D(image,-1,gaussian)
    fil_img = image/mask
    return mask, fil_img

def main():
    path = "checkerboard1024-shaded.tif"
    dim = 256
    std = 64
    img = Image.open(path)
    cv2_image = np.array(img)
    copy_img = cv2_image.copy()
    mask,filtered_image = gaussian_filter(cv2_image,dim,std)
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