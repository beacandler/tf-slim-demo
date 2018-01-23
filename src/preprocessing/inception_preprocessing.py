"""Functions to pre-process input data for model"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import logging


def apply_with_random_selector(image, func, num_cases):
    """random select a mode case to func(image, case)"""
    # random select a mode
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([
        func(control_flow_ops.switch(image, tf.equal(case, sel))[1], case)
         for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distorted the color of Tensor image.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering is not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_coverd=1.0,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                scope=None):
    """distorted a bounding box

    :param image: 3-D Tensor
    :param bbox: 3-D float Tensor
    :return:
    3-D flaot Tensor of distorted bounding box
    """
    with tf.name_scope(scope, 'distort_bounding_box', [image, bbox]):
        sampled_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_coverd,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            use_image_if_no_bounding_boxes=True
        )
        begin, size , bbox_for_draw = sampled_distorted_bounding_box
        cropped_image = tf.slice(image, begin, size)
        return cropped_image, bbox_for_draw

def preprocess_for_train(image, height, width, bbox=None,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
    """Distort one image for training a network.

    :param image: 3-D Tensor of Image
    :param height: excepted integer
    :param width:  excepted integer
    :param bbox:   3-D float Tensor
    :param fast_mode: Optional boolean
    :param scope: Optional scope
    :param add_image_summaries: Enable image summary
    :return:
    3-D float Tensor of distorted image used for training [-1, 1]
    """
    # a context manager for use when defining a Python op.
    # This context manager validates that given values are from the same graph, makes that graph
    # the default graph, and pushes a name scope in that graph
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.dtype is not tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        if add_image_summaries:
            tf.summary.image('image_with_box', image_with_box)
        cropped_image, bbox_for_draw = distorted_bounding_box_crop(image, bbox)
        # restore the shape because dynamic slice based upon the bbox_size loses the third dimension
        cropped_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox_for_draw)
        if add_image_summaries:
            tf.summary.image('image_with_distorted_bounding_box', image_with_distorted_box)

        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            cropped_image,
            lambda x, resize_mode: tf.image.resize_images(x, [height, width], resize_mode),
            num_cases=num_resize_cases)
        if add_image_summaries:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        # Randomly flip the image horizontally
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        num_color_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, color_mode: distort_color(x, color_mode,fast_mode),
            num_cases=num_color_cases)
        if add_image_summaries:
            tf.summary.image('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)

        return distorted_image

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    """Prepare one image for evaluation

    if height and width are specified it would output a image with that size
    by applying resize_bilinear.

    if central_fraction is specified it would crop the central fraction of
    the input image
    :return:
        3-D float Tensor of prepared image
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype is not tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)

            # remove all specific size 1 dimensions
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
    return image

def preprocess_image(image, height, width,
                     is_training=True,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
    """Pre-process one image for train or eval

    :param image: 3-D tensor [height, width, channels] with the image. If dtype is
    tf.float32 then the range should be [0, 1], otherwise it would converted to tf.float32
    assuming the range is [0, MAX], where MAX is largest positive representable number for
    int(8/16/32) data type , (see 'tf.image.convert_image_dtype' for detail)
    :param height: integer, image excepted height
    :param width: integer, image excepted width
    :param is_training: Boolean. if true it would transform image for train
    :param bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, crds]
    where each crds is [0,1), arranged as [ymin, xmin, ymax, xmax]
    :param fast_mode: Optional boolean, if True avoids slower transformations
    :param add_image_summaries: Enable image summaries

    :return:
        3-D float tensor containing an appropriately scaled image
    :raise
        ValueError: If user does not provide bounding box
    """
    if is_training:
        return preprocess_for_train(image, height, width, bbox, fast_mode,
                                    add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(image, height, width)