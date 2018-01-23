import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import shutil

import dataset_utils

class Config:
    def __init__(self):
        self.dataset_dir = '/tmp/flower_photos/images'
        self.pairs_dir = '/tmp/flower_photos/npy/'
        self.tfrecord_dir = '/tmp/flower_photos/tfrecord/'
        self.split_ratio = 0.1
        self.clear = True
        if self.clear:
            if os.path.exists(self.pairs_dir):
                shutil.rmtree(self.pairs_dir)
            if os.path.exists(self.tfrecord_dir):
                shutil.rmtree(self.tfrecord_dir)
        for dir in [self.pairs_dir, self.tfrecord_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        for phase in ['train', 'val']:
            if not os.path.exists(os.path.join(self.tfrecord_dir, phase)):
                os.makedirs(os.path.join(self.tfrecord_dir, phase))

cfg = Config()

def tf_writer(phase):
    print 'phase: {}'.format(phase)
    pairs_npy_path = os.path.join(cfg.pairs_dir, 'leaders_{}_pairs.npy'.format(phase))
    pairs_npy = np.load(pairs_npy_path)
    nofpairs = len(pairs_npy)
    print 'will write {} pairs'.format(nofpairs)
    tfrecords_name = os.path.join(cfg.tfrecord_dir, phase, '{}_0.tfrecords'.format(phase))
    print 'creating tfrecords writer: {}'.format(tfrecords_name)
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    for ind in range(nofpairs):
        pair = pairs_npy[ind]
        if ind > 0 and ind % 1000 == 0:
            writer.close()
            tfrecords_name = os.path.join(cfg.tfrecord_dir, phase,'{}_{}.tfrecords'.format(phase, ind))
            print 'creating tfrecords writer: {}'.format(tfrecords_name)
            writer = tf.python_io.TFRecordWriter(tfrecords_name)
        fn = pair[0]
        label = pair[1]
        try:
            im = Image.open(fn)
        except IOError:
            print 'cannot load file: {}'.format(fn)
            continue
        im = np.array(im)
        im = im.astype(np.uint8)
        # read file name
        image_data = tf.gfile.FastGFile(fn, 'rb').read()
        height , width = im.shape[0], im.shape[1]
        example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, int(label))
        writer.write(example.SerializeToString())

def get_dataset():
    clss_name = os.listdir(cfg.dataset_dir)
    clss_name.remove('LICENSE.txt')
    # clss_name.append('other')
    print 'all leaders: {}, num of leaders: {}'.format(clss_name, len(clss_name))
    clss_name.sort()
    print 'all sorted leaders: {}'.format(clss_name)
    leaders_label_npy = np.array(clss_name)
    np.save(os.path.join(cfg.pairs_dir, 'leaders_label'), leaders_label_npy)
    leaders2label = dict(zip(clss_name, range(len(clss_name))))
    print 'leaders2label: {}'.format(leaders2label)

    leaders_pair = []
    for leader in clss_name:
        leader_faces = os.listdir(os.path.join(cfg.dataset_dir, leader))
        for face in leader_faces:
            leaders_pair.append((os.path.join(cfg.dataset_dir, leader, face), leaders2label[leader]))
    print 'all leaders face pairs : {}'.format(len(leaders_pair))
    rng = range(len(leaders_pair))
    random.shuffle(rng)
    num_leaders_pair_train = len(leaders_pair) - int(len(leaders_pair) * cfg.split_ratio)
    num_leaders_pair_val = int(len(leaders_pair) * cfg.split_ratio)
    print 'num of train face images: {}, num of val face images: {}, split ratio : {}'.\
        format(num_leaders_pair_train, num_leaders_pair_val, cfg.split_ratio)
    leaders_pair = np.array(leaders_pair)
    leaders_pair_train = leaders_pair[rng[:num_leaders_pair_train]]
    leaders_pair_val = leaders_pair[rng[num_leaders_pair_train:]]
    fn_train_npy = np.array(leaders_pair_train)
    fn_val_npy = np.array(leaders_pair_val)

    np.save(os.path.join(cfg.pairs_dir, 'leaders_train_pairs'), fn_train_npy)
    np.save(os.path.join(cfg.pairs_dir, 'leaders_val_pairs'), fn_val_npy)
    print 'done'


def tf_writer_test():
    pass

def main():
    get_dataset()
    for phase in ['train', 'val']:
        tf_writer(phase)
    tf_writer_test()


if __name__ == '__main__':
    main()