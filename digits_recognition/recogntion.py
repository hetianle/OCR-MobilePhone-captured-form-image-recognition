
from PIL import Image
import tesserocr
import cv2
import numpy as np
import tensorflow as tf



def recognize_digit(img):


    return 0


def recognize_handwritten_chinese(img):

    return "0"

def recognize_print_chinese(img):
    pil_img_grid = Image.fromarray(img)

    content = tesserocr.image_to_text(pil_img_grid)
    return content



def ocr_grid_recognition(img_grid):
    def segmentation(img):
        def resize(image, size, back_value):
            pre_x, pre_y = image.shape

            if pre_x > pre_y:
                temp_img = np.zeros(shape=(pre_x, pre_x), dtype=np.uint8)
                stride_left = int((pre_x - pre_y) / 2)
                stride_right = int((pre_x - pre_y) / 2 + pre_y)
                for i in range(pre_x):
                    for j in range(pre_x):
                        if j < stride_left:
                            temp_img[i, j] = back_value
                        elif j >= stride_left and j < stride_right:
                            temp_img[i, j] = image[i, j - stride_left]
                        elif j >= stride_right:
                            temp_img[i, j] = back_value
                imggg = Image.fromarray(temp_img).resize(size)


            else:
                temp_img = np.zeros(shape=(pre_y, pre_y), dtype=np.uint8)
                stride_top = int((pre_y - pre_x) / 2)
                stride_bottom = int((pre_y - pre_x) / 2 + pre_x)
                for i in range(pre_y):
                    for j in range(pre_y):
                        if i < stride_top:
                            temp_img[i, j] = back_value
                        elif i >= stride_top and i < stride_bottom:
                            temp_img[i, j] = image[i - stride_top, j]
                        elif i >= stride_bottom:
                            temp_img[i, j] = back_value
                imggg = Image.fromarray(temp_img).resize(size)

            iiii = np.array(imggg.getdata(), dtype=np.uint8).reshape(size)
            # Image.fromarray(iiii).show()

            return iiii

        w, h = img.shape
        # m=count.max
        # mmm=np.where(count == count.max)
        # back_value=value[np.where(count==count.max)]

        img = img[2:w - 2, 2:h - 2]

        # im_gray = cv2.GaussianBlur(img, (5, 5), 0)

        # Threshold the image
        # Image.fromarray(img).show()

        ret, im_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Image.fromarray(im_th).show()

        image, contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Get rectangles contains each contour

        # Image.fromarray(image).show()

        rects = [cv2.boundingRect(ctr) for ctr in contours]

        rects.sort(key=lambda ctr: (ctr[0] + ctr[2] / 2))

        # value, count = np.unique(img.ravel(), return_counts=True)
        # value = list(value)
        # count = list(count)
        # back_value = value[count.index(max(count))]  ## count the most digit value as background

        # Image.fromarray(img).show()

        # imggg=resize(img,(28,28),back_value)
        # resize(np.zeros((2, 25), dtype=np.uint8), (28, 28), back_value)
        resized_sements = []
        for rect in rects:
            # Draw the rectangles
            patch = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            # Image.fromarray(patch).show()

            # Image.fromarray(patch).show()
            patch_x,path_y=patch.shape
            if patch_x<4 or path_y<4:
                continue

            resized_patch = resize(patch, (28, 28), back_value=0)
            # Image.fromarray(resized_patch).show()
            resized_sements.append(resized_patch)

        return resized_sements


    imgdigit_lists=segmentation(img_grid)
    #
    # for i in imgdigit_lists:
    #     (x_length,y_length)=img_grid.shape
    #     shrinked_img_grid=img_grid[1:x_length-1,1:y_length-1]

    return imgdigit_lists