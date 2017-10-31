import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

import dataset_nonMNIST as datasetNM
#import BatchDatasetReader as dataset
#import DataParser as scene_parsing
from model import *

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 5000, 'Total training iter')
flags.DEFINE_integer('step', 500, 'Save after ... iteration')
flags.DEFINE_integer('image_size', 256, 'Image size')
flags.DEFINE_integer('dataDir', './', 'Directory containing training images')
flags.DEFINE_integer('numClasses', 10, 'Number of classes')

"""image_options = {'resize': True, 'resize_size': FLAGS.image_size}
train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
train_dataset_reader = dataset.BatchDatset(train_records, image_options)
train_images, train_annotations = train_dataset_reader.next_batch_inference_partitioned(part)

validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
valid_images, valid_annotations = validation_dataset_reader.next_batch_inference_partitioned(part)"""

train_images, train_labels, validation_images, validation_labels, data_sets
 = datasetNM.read_train_sets(FLAGS.dataDir, FLAGS.image_size, FLAGS.numClasses, validation_size=1)

gen = BatchGenerator(train_images, train_labels)
test_im = np.array([im.reshape((FLAGS.image_size, FLAGS.image_size,3)) for im in valid_images])
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3], name='left')
right = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3], name='right')
with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2

left_output = mynet(left, reuse=False)

right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)


# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#setup tensorboard	
	tf.summary.scalar('step', global_step)
	tf.summary.scalar('loss', loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train.log', sess.graph)

	#train iter
	for i in range(FLAGS.train_iter):
		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

		_, l, summary_str = sess.run([train_step, loss, merged], 
			feed_dict={left:b_l, right:b_r, label: b_sim})
		
		writer.add_summary(summary_str, i)
		print "\r#%d - Loss"%i, l

		
		if (i + 1) % FLAGS.step == 0:
			#generate test
			feat = sess.run(left_output, feed_dict={left:test_im})
			
			labels = train_annotations
			# plot result
			f = plt.figure(figsize=(16,9))
			for j in range(10):
			    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
			    	'.', c=c[j],alpha=0.8)
			plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
			plt.savefig('img/%d.jpg' % (i + 1))

	saver.save(sess, "model/model.ckpt")





