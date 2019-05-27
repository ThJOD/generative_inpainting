import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--segmentation', default=None, type=str,
                    help='Where the segmentationFiles are saved.')
parser.add_argument('--segmentationClasses', default=8, type=int,
                    help='How many segmentation classes there are.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    #ng.get_gpus(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()

    batch_testing = False
    segmentation = False
    if os.path.isdir(args.image):
        batch_testing = True
        assert os.path.isdir(args.output)
        assert os.path.isdir(args.mask)
        if not args.segmentation is None:
            segmentation = True
            assert os.path.isdir(args.segmentation)

    model = InpaintCAModel()
    image = []
    mask = []
    segmentations = []
    if batch_testing:
        for root, dirs, files in os.walk(args.image, topdown=False):
            files.sort()
            for name in files:
                image.append(cv2.imread(os.path.join(root,name)))
        for root, dirs, files in os.walk(args.mask, topdown=False):
            files.sort()
            for name in files:
                mask.append(cv2.imread(os.path.join(root,name)))
        if segmentation:
            for root, dirs, files in os.walk(args.segmentation, topdown=False):
                files.sort()
                for name in files:
                    segmentations.append(cv2.imread(os.path.join(root,name))[:,:,0:1])
            assert len(image) == len(segmentations)
        assert len(image) == len(mask)
        
        for i in range(len(image)):
            assert image[i].shape == mask[i].shape
    else:
        image = cv2.imread(args.image)
        mask = cv2.imread(args.mask)
        if segmentation:
            segmentations = cv2.imread(args.segmentation)[:,:,0:1]
            assert image.shape[0:2] == segmentations.shape[0:2]
        assert image.shape == mask.shape

    if batch_testing:
        for idx in range(len(image)):
            h, w, _ = image[idx].shape
            grid = 8
            image[idx] = image[idx][:h//grid*grid, :w//grid*grid, :]
            mask[idx] = mask[idx][:h//grid*grid, :w//grid*grid, :]
            if segmentation:
                segmentations[idx] = segmentations[idx][:h//grid*grid, :w//grid*grid, :]
                segmentations[idx] = np.expand_dims(segmentations[idx], 0)
            print('Shape of image: {}'.format(image[idx].shape))
            image[idx] = np.expand_dims(image[idx], 0)
            mask[idx] = np.expand_dims(mask[idx], 0)
    else:
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        if segmentation:
                    segmentations = segmentations[:h//grid*grid, :w//grid*grid, :]

        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    #sess_config = tf.ConfigProto()
    sess_config = tf.ConfigProto(        device_count = {'GPU': 0}    )
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph_gated(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(args.output, result[0][:, :, ::-1])
