# notes:
# Statistics obtained from an image's histogram can be used for image enhancement.we denote r_i as the intensity value of pixels, and p(r_i)
# as the probability of some pixels occurring in an image. So the mean and variance of some pixel distribution is:
# mean = sum((r_i * n_i)/(L-1)) for i=0 to L-1 = sum(r_i) = sum(r_i * p(r_i)) for i=0 to L-1 where n_i is the number of pixels with intensity r_i
# variance = sum((r_i - meam)^2*p(r_i))) for i=0 to L-1 where variance is the measure of image contrast.

# So why do we want to calculate mean and variance?

# the "global mean & variance" of an image can be useful for gross adjustments in overall intensity & contrast. 

# But a more powerful tool is the "local mean & variance" where it lets us do "local enhancement" depending on the image characteristics in a
# neighbourhood about each pixel in an image. Let S_xy denote "a neighbourhood of specified size, centered at pixel (x,y)". The mean of that neighbourhood
# is:
# mean_S_xy = sum(r_i * p_S_xy(r_i)) for i = 0 to L-1
# variance_S_xy = sum((r_i-mean_S_xy)^2 * p_S_xy(r_i)) for i = 0 to L-1
# where local mean is the average intensity of neighbourhood S_xy & local variance is the measure of intensity contrast of neighbourhood S_xy.

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def global_cal(image):
    glb_mean = np.mean(image) #np.mean computes the mean of all pixels in the image
    glb_var = np.var(image)   #np.var computes the variance of all pixels in the image
    return glb_mean,glb_var
    
def local_cal(arr):
    local_mean = np.mean(arr)
    local_var = np.var(arr)
    return local_mean, local_var
def histogram_equi(image):
    img_for_equi = cv2.equalizeHist(image)
    return img_for_equi

def neighbourhood_enhance(copy_image,C,left,right,up,down,x,y):
    dtype = copy_image.dtype
    def mul_clip(i, j):
        v = float(copy_image[i, j]) * float(C)   # do multiplication in float to avoid uint8 wrap/overflow
        #print(np.clip(v, 0, 255).astype(dtype))
        return np.clip(v, 0, 255).astype(dtype)  # clip then cast back to original dtype

    copy_image[x, y] = mul_clip(x, y)
    return copy_image


def local_image_enhance(image):
    C = 22.8 # some constant to multiply by f(x,y)
    k0,k1 = 0.0,0.1
    k2,k3 = 0.0,0.1
    glb_mean,glb_var = global_cal(image)
    copy_image = image.copy()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            arr = []
            left = x-1
            right = x+1
            up = y-1
            down = y+1
            arr.append(image[x][y])
            if left < 0:
                arr.append(1)
            else:
                arr.append(image[left][y])
            if right > image.shape[0]-1:
                arr.append(1)
            else:
                arr.append(image[right][y])
            if up < 0:
                arr.append(1)
            else:
                arr.append(image[x][up])
            if down > image.shape[1]-1:
                arr.append(1)
            else:
                arr.append(image[x][down])
            if left < 0 or up < 0:
                arr.append(1)
            else:
                arr.append(image[left][up])
            if left < 0 or down > image.shape[1]-1:
                arr.append(1)
            else:
                arr.append(image[left][down])
            if right > image.shape[0]-1 or up < 0:
                arr.append(1)
            else:
                arr.append(image[right][up])
            if right > image.shape[0]-1 or down > image.shape[1]-1:
                arr.append(1)
            else:
                arr.append(image[right][down])
            local_mean, local_var = local_cal(arr)
            if local_mean >= k0 * glb_mean and local_mean <= k1 * glb_mean and local_var >= k2 * glb_var and local_var <= k3 * glb_var:
                neighbourhood_enhance(copy_image,C,left,right,up,down,x,y)
    return copy_image

def main():
    path = "hidden_object.jpg"
    img = Image.open(path).convert('L') # convert to grayscale
    img = np.array(img)
    global_enhanced_img = histogram_equi(img)
    local_enhanced_img = local_image_enhance(img)
    # Use matplotlib to display each image alongside its histogram
    images = [
        ("Original", img),
        ("Global Enhanced", global_enhanced_img),
        ("Local Enhanced", local_enhanced_img),
    ]

    n = len(images)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 4*n))

    for row, (title, im) in enumerate(images):
        ax_img = axes[row, 0]
        ax_hist = axes[row, 1]

        # show image (grayscale)
        ax_img.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax_img.set_title(title)
        ax_img.axis('off')

        # histogram
        ax_hist.hist(im.ravel(), bins=256, range=(0,256), color='black')
        ax_hist.set_title(f"{title} histogram")
        ax_hist.set_xlim(0,255)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()