#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from datasets.single_file_imdb import single_file_imdb
from datasets.custom_voc_structure_db import custom_voc_structure_db
import caffe
import argparse
import pprint
import numpy as np
import sys
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None)
    clsGroup = parser.add_mutually_exclusive_group()
    clsGroup.add_argument('--vocClasses', action='store_const', dest='classes',
                          const=datasets.imdb.imdb.USE_PASCAL_CLASSES,
                          help='Use Pascal VOC classes for training.')
    clsGroup.add_argument('--cocoClasses', action='store_const', dest='classes',
                          const=datasets.imdb.imdb.USE_COCO_CLASSES,
                          help='Use coco classes for training.')

    parser.add_argument('--imdb', dest='imdb_names', action='append',
                        help='One or more pascal or coco datasets to train on')
    parser.add_argument('--customvoc', dest='custom_voc_paths', action='append',
                        help='One or more text files describing custom VOC-style \
datasets to train on.  Interpreted relative to --datadir if that argument is \
provided.')
    parser.add_argument('--textdb', dest='textdb_paths', action='append',
                        help='One or more text files containing both full paths \
to images and their annotations.')
    parser.add_argument('--imdbdir', dest='imdb_data_dir',
                        help='root directory of pascal VOC databases.')
    parser.add_argument('--customvocdir', dest='custom_voc_data_dir',
                        help='root directory of custom pascal VOC-style databases.')

    # TODO add param for indicating the type of dataset structure (voc, coco, other)
    parser.add_argument('--out', dest='out_dir',
                        help='Directory in which to store the output weights',
                        type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if not (args.imdb_names or args.custom_voc_paths or args.textdb_paths):
        parser.error('No datasets specified!')

    if args.textdb_paths and not args.classes:
        parser.error('With textdbs, must provide one of --vocClasses, --cocoClasses args')

    return args


def get_roidb_from_imdb(imdb):
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return roidb


def get_combined_roidb(imdbs):
    roidbs = [get_roidb_from_imdb(imdb) for imdb in imdbs]
    combined_roidb = roidbs[0]
    if len(roidbs) > 1:
        for roidb in roidbs[1:]:
            combined_roidb.extend(roidb)
    return combined_roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    all_imdbs = []
    if args.textdb_paths:
        # Create a single_file_imdb for each one
        for textdb_path in args.textdb_paths:
            textdb = single_file_imdb(textdb_path, args.classes)
            all_imdbs.append(textdb)

    kwargs = {}
    if args.classes:
        kwargs['class_set'] = args.classes

    if args.custom_voc_paths:
        if args.custom_voc_data_dir:
            kwargs['devkit_path'] = args.custom_voc_data_dir
        for path in args.custom_voc_paths:
            all_imdbs.append(custom_voc_structure_db(path, **kwargs))

    if args.imdb_names:
        if args.imdb_data_dir:
            kwargs['devkit_path'] = args.imdb_data_dir
        for imdb_name in args.imdb_names:
            all_imdbs.append(get_imdb(imdb_name, **kwargs))

    # combine all of the imdbs
    roidb = get_combined_roidb(all_imdbs)
    print '{:d} roidb entries'.format(len(roidb))

    output_name = '+'.join([imdb.name for imdb in all_imdbs])
    if args.out_dir:
        output_dir = os.path.join(args.out_dir, output_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        dummy_outdir_imdb = datasets.imdb.imdb(output_name, args.classes)
        output_dir = get_output_dir(dummy_outdir_imdb)

    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
