# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# File modified by Anqi Xu and Andrew Holliday, inspired
# by coco.py
# --------------------------------------------------------

from datasets.imdb import imdb
import os
import numpy as np
import scipy.sparse
import cPickle
import uuid


# An imdb built around a single text file that lists images and annotations.
class single_file_imdb(imdb):
    def __init__(self, image_set_txt_path, class_set):
        # strip off the .txt from the filename to use as our name.
        imdb_name = os.path.basename(image_set_txt_path).rpartition('.')[0]
        imdb.__init__(self, imdb_name, class_set)
        # Parameters
        self.config = {'use_salt': True,  # generic parameter: use uuid to salt comp_id
                       'cleanup': True,  # generic parameter: remove temp files created for evaluations
                       'min_size': 2,     # selective search: filter small boxes
                       'top_k': 2000,  # region proposals: top K proposal to consider
                       }
        # name, paths
        self._data_path = os.path.dirname(image_set_txt_path)
        self._image_set_txt = os.path.basename(image_set_txt_path)

        # read in the file, parsing it into a list of image indexes and annotations
        image_filenames, image_gt_boxes = self.parse_text_dataset()
        self._image_index = image_filenames

        # build an roidb from the annotations
        self._roidb = []
        for gt_boxes in image_gt_boxes:
            boxes = np.zeros((len(gt_boxes), 4), dtype=np.uint16)
            gt_classes = np.zeros(len(gt_boxes), dtype=np.int32)
            overlaps = np.zeros((len(gt_boxes), self.num_classes), dtype=np.float32)
            seg_areas = np.zeros(len(gt_boxes), dtype=np.float32)
            for i, (left, top, right, bottom, cls) in enumerate(gt_boxes):
                boxes[i, :] = left, top, right, bottom
                cls_idx = self._class_to_ind[cls]
                gt_classes[i] = cls_idx
                overlaps[i, cls_idx] = 1.0
                seg_areas = (right - left + 1) * (bottom - top + 1)

            overlaps = scipy.sparse.csr_matrix(overlaps)
            self._roidb.append({'boxes': boxes,
                                'gt_classes': gt_classes,
                                'gt_overlaps': overlaps,
                                'flipped': False,
                                'seg_areas': seg_areas})

        # TODO add support for .png?
        self._image_exts = ['.JPEG', '.JPG', '.jpg']
        # We're building the roidb right here, so we don't need a handler.
        # self._roidb_handler = self.gt_roidb
        self.competition_mode(False)

    def parse_text_dataset(self):
        db_path = os.path.join(self._data_path, self._image_set_txt)
        with open(db_path, 'r') as f:
            image_filenames = []
            image_gt_boxes = []
            gt_boxes = []

            for line_idx, line in enumerate(f.readlines()):
                if len(line) == 0:
                    # ignore empty lines
                    continue
                if line[0] == ' ':
                    # this line describes a bounding box for the previous image
                    if len(image_filenames) == 0:
                        raise ValueError('found bbox entries before first image path on line %d' % (line_idx + 1))
                    line = line.replace(',', '')
                    splitLine = line.split()
                    cls = splitLine[4]
                    left, top, right, bottom = map(float, splitLine[:4])
                    gt_boxes.append((left, top, right, bottom, cls))
                else:
                    # this line describes an image
                    if len(image_filenames) > 0:
                        # finalize the ground-truth box collection for the last image
                        image_gt_boxes.append(gt_boxes)
                        gt_boxes = []
                    line = line.strip()
                    image_filenames.append(os.path.join(self._data_path, line))

            if len(gt_boxes) > 0:
                # After the end of the file, append the last collection of boxes
                image_gt_boxes.append(gt_boxes)

        return image_filenames, image_gt_boxes

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_index[i]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future
        calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                pklRoidb = cPickle.load(fid)
            if pklRoidb[0]['gt_overlaps'].shape[1] == self.num_classes:
                print '{} gt roidb loaded from {}'.format(self.name, cache_file)
                return pklRoidb
            else:
                print 'roidb loaded from {} had wrong class count. Ignoring.'\
                    .format(cache_file)

        # The roidb we build in __init__ is the ground truth roidb.
        gt_roidb = self.roidb

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self._image_set +
                                             self._year +
                                             '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
