"""Tests for nets.inception_v3"""

import numpy as np
import tensorflow as tf

import inception_v3
from inception_utils import inception_arg_scope
slim = tf.contrib.slim

class InceptionV3Test(tf.test.TestCase):

    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = inception_v3.inception_v3(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith(
                        'InceptionV3/Logits/SpatialSqueeze'))
        self.assertListEqual(logits.get_shape().as_list(), [batch_size, num_classes])
        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                             [batch_size, num_classes])

    def testBuildPreLogitsNetwork(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = None
        inputs = tf.random_uniform((batch_size, height, width, 3))
        AvgPool, end_points = inception_v3.inception_v3(inputs, num_classes)
        self.assertTrue(AvgPool.op.name.startswith(
                        'InceptionV3/Logits/AvgPool'))
        self.assertListEqual(AvgPool.get_shape().as_list(), [batch_size, 1, 1, 2048])
        self.assertFalse('Logits' in end_points)
        self.assertFalse('Predictions' in end_points)

    def testBuildBaseNetwork(self):
        batch_size = 5
        height, width = 299, 299
        inputs = tf.random_uniform((batch_size, height, width, 3))
        final_endpoint, endpoints = inception_v3.inception_v3_base(inputs)
        self.assertTrue(final_endpoint.op.name.startswith(
            'InceptionV3/Mixed_7c'))
        self.assertListEqual(final_endpoint.get_shape().as_list(),
                             [batch_size, 8, 8, 2048])
        expected_endpoints = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                              'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                              'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
                              'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                              'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']
        self.assertItemsEqual(endpoints.keys(), expected_endpoints)

    def testBuildOnlyUptoFinalPoint(self):
        batch_size = 5
        height, width = 299, 299
        endpoints = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                      'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
                      'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                      'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']

        for index, endpoint in enumerate(endpoints):
            with tf.Graph().as_default():
                inputs = tf.random_uniform((batch_size, height, width, 3))
                final_endpoint, end_points = inception_v3.inception_v3_base(
                    inputs, final_endpoint=endpoint)
                self.assertTrue(final_endpoint.op.name.startswith(
                    'InceptionV3/' + endpoint))
                self.assertItemsEqual(endpoints[:index+1], end_points)
    def testBuildAndCheckAllEndPointsUptoMixed7c(self):
        batch_size = 5
        height, width = 299, 299

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = inception_v3.inception_v3_base(
            inputs, final_endpoint='Mixed_7c')
        end_points_shapes = {'Conv2d_1a_3x3': [batch_size, 149, 149, 32],
                             'Conv2d_2a_3x3': [batch_size, 147, 147, 32],
                             'Conv2d_2b_3x3': [batch_size, 147, 147, 64],
                             'MaxPool_3a_3x3': [batch_size, 73, 73, 64],
                             'Conv2d_3b_1x1': [batch_size, 73, 73, 80],
                             'Conv2d_4a_3x3': [batch_size, 71, 71, 192],
                             'MaxPool_5a_3x3': [batch_size, 35, 35, 192],
                             'Mixed_5b': [batch_size, 35, 35, 256],
                             'Mixed_5c': [batch_size, 35, 35, 288],
                             'Mixed_5d': [batch_size, 35, 35, 288],
                             'Mixed_6a': [batch_size, 17, 17, 768],
                             'Mixed_6b': [batch_size, 17, 17, 768],
                             'Mixed_6c': [batch_size, 17, 17, 768],
                             'Mixed_6d': [batch_size, 17, 17, 768],
                             'Mixed_6e': [batch_size, 17, 17, 768],
                             'Mixed_7a': [batch_size, 8, 8, 1280],
                             'Mixed_7b': [batch_size, 8, 8, 2048],
                             'Mixed_7c': [batch_size, 8, 8, 2048]}
        self.assertItemsEqual(end_points_shapes.keys(), end_points.keys())
        for endpoint_name in end_points_shapes:
            expected_shape = end_points_shapes[endpoint_name]
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(), expected_shape,
                                 'endpoint_name: {}, expected_shape: {}'.format(endpoint_name, expected_shape))

    def testModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 299, 299
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3_base(inputs)
        total_params, _ = slim.model_analyzer.analyze_vars(
            slim.get_model_variables())
        self.assertAlmostEqual(21802784, total_params)

    def testBuildEndPoints(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = inception_v3.inception_v3(inputs, num_classes)
        self.assertTrue('Logits' in end_points)
        logits = end_points['Logits']
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        self.assertTrue('AuxLogits' in end_points)
        aux_logits = end_points['AuxLogits']
        self.assertListEqual(aux_logits.get_shape().as_list(),
                             [batch_size, num_classes])
        self.assertTrue('Mixed_7c' in end_points)
        pre_pool = end_points['Mixed_7c']
        self.assertListEqual(pre_pool.get_shape().as_list(),
                             [batch_size, 8, 8, 2048])
        self.assertTrue('PreLogits' in end_points)
        pre_logits = end_points['PreLogits']
        self.assertListEqual(pre_logits.get_shape().as_list(),
                             [batch_size, 1, 1, 2048])

    def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = inception_v3.inception_v3(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Mixed') or key.startswith('Conv')]

        _, end_points_with_multiplier = inception_v3.inception_v3(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=0.5)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(0.5 * original_depth, new_depth, key)

    def testBuildEndPointsWithDepthMultiplierGreaterThanOne(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = inception_v3.inception_v3(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Mixed') or key.startswith('Conv')]

        _, end_points_with_multiplier = inception_v3.inception_v3(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=2.0)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(2.0 * original_depth, new_depth, key)

    def testRaiseValueErrorWithInvalidDepthMultiplier(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with self.assertRaises(ValueError):
            _, end_points = inception_v3.inception_v3(inputs, num_classes, depth_multiplier=-0.1)
        with self.assertRaises(ValueError):
            _, end_points = inception_v3.inception_v3(inputs, num_classes, depth_multiplier=0.0)

    def testHalfSizeImages(self):
        batch_size = 5
        height, width = 150, 150
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = inception_v3.inception_v3(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('InceptionV3/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        pre_pool = end_points['Mixed_7c']
        self.assertListEqual(pre_pool.get_shape().as_list(),
                             [batch_size, 3, 3, 2048])


    def testUnknownImageShape(self):
        batch_size = 2
        height, width = 299, 299
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))

        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = inception_v3.inception_v3(inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
            pre_pool = end_points['Mixed_7c']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 8, 8, 2048])

    def testGlobalPoolUnknownImageShape(self):
        tf.reset_default_graph()
        batch_size = 2
        height, width = 400, 600
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))

        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = inception_v3.inception_v3(inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
            pre_pool = end_points['Mixed_7c']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 11, 17, 2048]), list(pre_pool_out.shape)

    def testUnknowBatchSize(self):
        batch_size = 1
        height, width = 299, 299
        num_classes = 1000
        images = tf.random_uniform((batch_size, height, width, 3))

        inputs = tf.placeholder(tf.float32, (None, height, width, 3))
        logits, _ = inception_v3.inception_v3(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('InceptionV3/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [None, num_classes])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            # images.eval()  is qual to tf.get_default_session().run(images)
            output = sess.run(logits, {inputs: images.eval()}) #???
            self.assertEquals(output.shape, (batch_size, num_classes))

    def testEvaluation(self):
        batch_size = 2
        height, width = 299, 299
        num_classes = 1000
        # this is a tensor and convert to a value when call 'eval_inputs.eval() or sess.run(eval_inputs)'
        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = inception_v3.inception_v3(eval_inputs, num_classes,
                                              is_training=False)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (batch_size, ))

    def testTrainEvalWithReuse(self):
        train_batch_size = 5
        eval_batch_size = 2
        height, width = 150, 150
        num_classes = 1000
        train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
        inception_v3.inception_v3(train_inputs, num_classes)
        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = inception_v3.inception_v3(eval_inputs, num_classes,
                                              is_training=False, reuse=True)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertTrue(output.shape, (eval_batch_size, ))

    def testLogitsNotSqueezed(self):
        num_classes = 25
        images = tf.random_uniform([1, 299, 299, 3])
        logits, _ = inception_v3.inception_v3(images,
                                              num_classes=num_classes,
                                              spatial_squeeze=False)
        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            logits_out = sess.run(logits)
            self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])

    def testMisc(self):
        batch_size = 2
        num_classes = 1000
        images = tf.random_uniform([batch_size, 299, 299, 3])
        Logits, endpoints = inception_v3.inception_v3(images, num_classes)
        for key in endpoints.keys():
            endpoint = endpoints[key]
            self.assertTrue(endpoint.op.name.startswith('InceptionV3'))

