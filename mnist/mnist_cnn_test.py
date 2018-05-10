from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage import color,io,transform,filters,morphology,measure
import matplotlib.pyplot as plt
import numpy as np
import mnist_data
import cnn_model

#load image & convert to gray scale
img=color.rgb2gray(io.imread('test.png'))

#smooth & threshold to remove noise
img=filters.gaussian(img,sigma=1)
thresh = filters.threshold_li(img)
img =(img <= thresh)*1.0

plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')

#convex to find the digits
#convex = morphology.convex_hull_object(img)
props=measure.regionprops(measure.label(img, connectivity=2))
print(props[0].bbox)
rowspan=props[0].bbox[2]-props[0].bbox[0]
colspan=props[0].bbox[3]-props[0].bbox[1]
half_len=np.maximum(colspan,rowspan)/2
props[0].centroid
#img[80:180,100:200,:]
#region_img=props[0].image
cropped=img[int(props[0].centroid[0]-half_len):int(props[0].centroid[0]+half_len),int(props[0].centroid[1]-half_len):int(props[0].centroid[1]+half_len)]
#cropped=img[10:20,30:40]
plt.subplot(2,2,2)
plt.imshow(cropped,cmap='gray')



mask=np.zeros(cropped.shape)
mask[cropped==0]=-0.5
mask[cropped==1]=0.5

plt.subplot(2,2,3)
plt.imshow(mask,cmap='gray')

input=transform.resize(mask,(28,28))




#input=morphology.dilation(mask,morphology.square(2))

#outfile = open('image_3.txt', 'rb')
#input = np.load(outfile)
#input=np.reshape(input,(28,28))
plt.subplot(2,2,4)
plt.imshow(input,cmap='gray')


input=np.reshape(input,(1,784))

is_training = tf.placeholder(tf.bool, name='MODE')
# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
y = cnn_model.CNN(x, is_training=is_training)

# Add ops to save and restore all the variables
#tf.reset_default_graph()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

# Restore variables from disk
saver = tf.train.Saver()
saver.restore(sess, '.\\model\\model.ckpt')

acc_buffer = []
# Loop over all batches
y_final = sess.run(y, feed_dict={x: input,is_training: False})
print (y_final)
print ('final result is :{0}'.format(np.argmax(y_final,1)))
sess.close()
#correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))

plt.title('result :{0}'.format(np.argmax(y_final,1)))
plt.show()
