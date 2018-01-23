import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

import data_provider
import common_flags
from preprocessing import inception_preprocessing

class DataProviderTest(tf.test.TestCase):
    def setUp(self):
        tf.test.TestCase.setUp(self)

    def test_preprocessed_image_values_are_in_range(self):
        for split_name in ['train', 'val']:
            is_training = True if split_name == 'train' else False
            image_shape = (5, 4, 3)
            fake_image = np.random.randint(low=0, high=255, size=image_shape)
            image_tf = inception_preprocessing.preprocess_image(
                fake_image,
                height=5,
                width=4,
                is_training=is_training)

            with self.test_session() as sess:
                image_np = sess.run(image_tf)

            self.assertEqual(image_shape, image_np.shape)
            min_value, max_value = np.min(image_np), np.max(image_np)
            self.assertTrue((-1.28 < min_value) and (min_value < 1.27))
            self.assertTrue((-1.28 < min_value) and (max_value < 1.27))

    def test_provided_data_has_correct_shape(self):
        dataset_name = 'flowers'
        model_name = 'inception_v3'
        for split_name in ['train']:
            is_training = True if split_name == 'train' else False
            dataset = common_flags.create_dataset(dataset_name, split_name)
            batch_size = 4
            data = data_provider.get_data(dataset,
                                          model_name,
                                          batch_size=batch_size,
                                          is_training=is_training,
                                          height=224,
                                          width=224)
            with self.test_session() as sess, slim.queues.QueueRunners(sess):
                images_np, labels_np = sess.run([data.images, data.labels])
            self.assertEqual(images_np.shape, (batch_size, 224, 224, 3))

if __name__=='__main__':
    tf.test.main()
