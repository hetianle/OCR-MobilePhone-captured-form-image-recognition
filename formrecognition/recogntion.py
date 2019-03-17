from PIL import Image
import tesserocr
import cv2
import numpy as np
# import formrecognition.predict as pd
import tensorflow as tf


def predict(data):

    restore_graph=tf.Graph()
    with tf.Session(graph=restore_graph) as restore_sess:
        # predict_data = standarlization_input()
        # mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

        # t = (mnist.test.images[0:3])*255
        #
        #
        #
        # a, b, c = np.uint8(t[0].reshape((28,28))), np.uint8(t[1].reshape((28,28))), np.uint8(t[2].reshape((28,28)))
        #
        # Image.fromarray(a).show()
        # Image.fromarray(b).show()
        # Image.fromarray(c).show()
        #


        restore_saver=tf.train.import_meta_graph('digits_recognition/data-20000.meta')
        restore_saver.restore(restore_sess,'digits_recognition/data-20000')
        input=restore_graph.get_tensor_by_name('input_tensor:0')

        labels=restore_graph.get_tensor_by_name('label_tensor:0')

        dropout=restore_graph.get_tensor_by_name('dropout:0')
        # inp=restore_graph.get_operation_by_name('input_tensor')
        # restore_graph.
        predict_op=tf.get_collection('predict_op')[0]
        # inputsss=tf.get_collection('input')[0]
        # inp.inputs
        # print(mnist.test.images[0])
        # t=mnist.test.images[0:3]

        # a,b,c=t[0],t[1],t[2]

        # feed_tensor=[input,labels,dropout]
        rr=restore_sess.run(predict_op,feed_dict={input: data,dropout:1.0})

        return rr


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


def segmentation(img):
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
        patch_x, path_y = patch.shape
        if patch_x < 4 or path_y < 4:
            continue

        resized_patch = resize(patch, (28, 28), back_value=0)
        # Image.fromarray(resized_patch).show()
        resized_sements.append(resized_patch)

    return resized_sements





def ocr_grid_recognition(img_grid):

    imgdigit_lists=segmentation(img_grid)
    if len(imgdigit_lists)==0:
        return None

    #
    # for i in imgdigit_lists:
    #     (x_length,y_length)=img_grid.shape
    #     shrinked_img_grid=img_grid[1:x_length-1,1:y_length-1]

    # grid_num=len(imgdigit_lists)

    data=np.array([(np.float32(k.ravel()) / 258.0) for k in imgdigit_lists])

    result = str(predict(data))
    # print(result)
    # feed_dict={feed_tensor[0]:data,feed_tensor[1]:np.zeros(shape=(3,10),dtype=np.float32),feed_tensor[2]:1.0}
    #
    # a=restore_sess.run(predict_op,feed_dict=feed_dict)

    # for i in range(len(imgdigit_lists)):
    #     # iiiii=Image.fromarray((digit_list[i]).reshape((28,28)))
    #     # iiiii.show(title='i')
    #     # iiiii.save('/Users/tianle/Desktop/digits_recognition/dot/dot%s.png'%(str(filesname)))
    #     # filesname+=1
    #     imgdigit_lists[i]=(np.float32(imgdigit_lists[i].ravel()) / 255.0)
    #     # feed_dict
    #     # session.run(predict_op, feed_dict={input: predict_data, labels: mnist.test.labels[5:9], dropout: 1.0})

    return result