import tensorflow as ts 

save_file = './train_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, save_file)
	test_accuracy = sess.run(accuracy, feed_dict={features:mnist.test.images, labels:mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))