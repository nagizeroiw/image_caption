import tensorflow as tf
from PIL import Image
from nets.inception_resnet_v2 import *
import numpy as np
import cPickle
import progressbar
slim = tf.contrib.slim


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


if __name__ == '__main__':

    # model restoration for tensorflow
    checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'

    # useful dataset pointers (to find images)
    dataset_dir = '/home/kesu/image_caption/dataset/'
    train_vids_path = dataset_dir + 'train_vids.pkl'
    valid_vids_path = dataset_dir + 'valid_vids.pkl'
    vid2name_path = dataset_dir + 'vid2name.pkl'
    feature_path = dataset_dir + 'feature.pkl'

    # [train/valid]_vids : list of vids like 'vid2'
    train_vids = load_pkl(train_vids_path)
    valid_vids = load_pkl(valid_vids_path)
    # vid2name : dictionary='vid2' -> 'data*/*train*/*.jpg'
    vid2name = load_pkl(vid2name_path)

    vids = train_vids + valid_vids

    # print len(image_names)
    # exit()

    input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3),
                                  name='input_image')
    scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

    # Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(scaled_input_tensor,
                                                 is_training=False)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    feature = dict()

    pbar = progressbar(maxval=len(vids))
    pbar.start()

    for i, vid in enumerate(vids):

        # init
        # feature[vid] = np.zeros((1, 1536))

        # find image
        image_path = dataset_dir + vid2name[vid]

        # pre-processing for inception-resnet-v2
        im = Image.open(image_path).resize((299, 299))
        im = np.array(im)
        im = im.reshape(-1, 299, 299, 3)

        # predict_values, logit_values, feature_tensor =
        #     sess.run([end_points['Predictions'], logits,
        #              end_points['PreLogitsFlatten']],
        #              feed_dict={input_tensor: im})

        feature_tensor = sess.run(end_points['PreLogitsFlatten'],
                                  feed_dict={input_tensor: im})

        assert (feature_tensor.shape == (1, 1536))

        feature[vid] = feature_tensor

        pbar.update(i + 1)

        # to get prediction results
        # print (np.max(predict_values), np.max(logit_values))
        # print (np.argmax(predict_values), np.argmax(logit_values))

        # print feature[vid].shape
        # print feature[vid]
        # exit()

    pbar.finish()

    # save feature dictionary

    print '>>> len(feature)', len(feature)
    print '>>> feature vector sample:', feature['vid0']

    dump_pkl(feature, feature_path)
