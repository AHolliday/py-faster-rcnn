# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# File modified by Anqi Xu, inspired by coco.py
# --------------------------------------------------------

from datasets.imdb import imdb
from datasets.pascal_voc import pascal_voc
import xml.etree.ElementTree as ET
import os
import numpy as np
import scipy.sparse
import uuid


# custom image db, with VOC-style structure and COCO-set classes+labels
class custom_voc_structure_db(pascal_voc):
    def __init__(self, image_set_txt_path, devkit_path=None, class_set=None):
        if class_set is None:
            class_set = imdb.USE_PASCAL_CLASSES
        name = os.path.basename(image_set_txt_path).rpartition('.')[0]
        imdb.__init__(self, name, class_set)
        # Parameters
        self.config = {'use_salt': True,  # generic parameter: use uuid to salt comp_id
                       'cleanup': True,  # generic parameter: remove temp files created for evaluations
                       'min_size': 2,     # selective search: filter small boxes
                       'top_k': 2000,  # region proposals: top K proposal to consider
                       'use_diff': False,  # VOC-specific: exclude samples labeled as difficult
                       }
        # name, paths
        if devkit_path is None:
            self._data_path = os.path.dirname(image_set_txt_path)
            self._image_set_txt = os.path.basename(image_set_txt_path)
        else:
            self._data_path = devkit_path
            self._image_set_txt = image_set_txt_path

        self._image_index = self._load_image_set_index()
        # TODO add support for .png?
        self._image_exts = ['.JPEG', '.JPG', '.jpg']
        self._roidb_handler = self.gt_roidb
        self.competition_mode(False)

    # NOTE: modified from pascal_voc.py to handle multiple image
    # suffixes, not just '.jpg'
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = ''
        for _image_ext in self._image_exts:
            image_path = os.path.join(self._data_path, 'JPEGImages',
                                      index + _image_ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # NOTE: modified from pascal_voc.py
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if os.path.isfile(self._image_set_txt):
            # The path leads directly to a file, so read it in.
            image_set_file = self._image_set_txt
        else:
            # image_set_txt does not reference a file, so it must be relative to
            # data_path.
            image_set_file = os.path.join(self._data_path, self._image_set_txt)
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    # NOTE: modified from pascal_voc.py, so imitates VOC-style 1-indexing
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        return self._load_pascal_annotation_xml(filename)

    def _load_pascal_annotation_xml(self, xmlpath):
        tree = ET.parse(xmlpath)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Assume these are 1-indexed
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            name = obj.find('name').text.lower().strip()
            if name == 'object':
                cls = 0
            else:
                cls = self._class_to_ind[name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    # NOTE: modified from pascal_voc.py.
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
