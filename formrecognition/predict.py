import tensorflow as tf
# from formrecognition.recogntion import ocr_grid_recognition
import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


# def predict(img):

filesname=0

#
# def standarlization_input(grid_img):
#
#     # data_path='/Users/tianle/Desktop/digits_recognition/data'
#     # data_path.sort()
#     # files=[data_path+'/'+f for f in os.listdir(data_path)]
#     # files.sort()
#     # for i in files:
#     imgindex=205
#     #
#     # filesname=imgindex
#     # tmp_img=cv2.imread('/Users/tianle/Desktop/digits_recognition/data/%s.png'%(str(imgindex)),cv2.IMREAD_GRAYSCALE)
#
#     digit_list=ocr_grid_recognition(grid_img)
#     # digit_list=mnist.test.images[5:11]
#
#     for i in range(len(digit_list)):
#         # iiiii=Image.fromarray((digit_list[i]).reshape((28,28)))
#         # iiiii.show(title='i')
#         # iiiii.save('/Users/tianle/Desktop/digits_recognition/dot/dot%s.png'%(str(filesname)))
#         # filesname+=1
#         digit_list[i]=(np.float32(digit_list[i].ravel()) / 255.0)
#
#
#         print()
#     return np.array(digit_list)




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
#
# session, predict_op, feed_dict = restore_tf_model()
#
# print(feed_dict.items())
#
# # feed_dict[tf.Tensor("dropout:0", dtype=tf.float32)]=np
# feed_dict[tf.Tensor("input_tensor:0",shape=(None, 784),dtype=tf.float32)]=np.zeros(shape=(100,784),dtype=np.float32)
# print()
#
#     # print(argmax(mnist.test.labels[0:3]))
    # print(restore_sess.run(tf.argmax(labels,1),feed_dict={input: mnist.test.images[5:11],labels:mnist.test.labels[5:11],dropout:1.0}))