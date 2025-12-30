import numpy as np
import os
import json
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import pandas as pd
import glob

def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


class UAVDark135Dataset(BaseDataset):
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.UAVDark135_path
        self.sequence_list = self._get_sequence_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_list = self._filter_sequence_list_by_attribute(attribute, self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno_revise/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        target_visible = 1

        frames_path = '{}/Sequences/{}'.format(self.base_path, sequence_name)

        jpg_files = glob.glob(frames_path + '/*.jpg')
        jpg_count = len(jpg_files)
        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, jpg_count + 1)]

        return Sequence(sequence_name, frames_list, 'uavdark135', ground_truth_rect.reshape(-1, 4), target_visible=target_visible)

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'nt-vot211_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        return [s for s in seq_list if att in self.att_dict[s]]

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)  # frames start from 1

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['basketballplayer1', 'basketballplayer2', 'basketballplayer3', 'bike1', 'bike10', 'bike11', 'bike2', 'bike3', 'bike4', 'bike5', 'bike6', 'bike7', 'bike8', 'bike9', 'boat1', 'boat2', 'building1', 'building2', 'bus1', 'bus2', 'bus3', 'car1', 'car10', 'car11', 'car12', 'car13', 'car14', 'car15', 'car16', 'car17', 'car18', 'car19', 'car2', 'car3', 'car4', 'car5', 'car6', 'car7', 'car8', 'car9', 'car_l1', 'car_l2', 'car_l3', 'car_l4', 'car_l5', 'car_l6', 'car_l7', 'dancing1', 'dancing2', 'girl1', 'girl2', 'girl3', 'girl4', 'girl5', 'girl6_1', 'girl6_2', 'girl7', 'group1', 'group2_1', 'group2_2', 'group3', 'group4_1', 'group4_2', 'group5', 'group6', 'group7', 'group8', 'group9_1', 'group9_2', 'house', 'jeep', 'jogging_man', 'minibus1', 'minibus2', 'motorbike1', 'motorbike2', 'motorbike3', 'motorbike4', 'motorbike5', 'pedestrian1', 'pedestrian10', 'pedestrian2', 'pedestrian3', 'pedestrian4', 'pedestrian5_1', 'pedestrian5_2', 'pedestrian5_3', 'pedestrian6', 'pedestrian7_1', 'pedestrian7_2', 'pedestrian7_3', 'pedestrian8', 'pedestrian9', 'pedestrian_l', 'person1', 'person10_1', 'person10_2', 'person11', 'person12_1', 'person12_2', 'person12_3', 'person13', 'person14', 'person15', 'person16_1', 'person16_2', 'person17', 'person18', 'person19', 'person2', 'person3_1', 'person3_2', 'person3_3', 'person4', 'person5', 'person6', 'person7', 'person8', 'person9', 'running', 'running_girl', 'running_man', 'signpost1', 'signpost2', 'signpost3', 'signpost4', 'signpost5', 'signpost6', 'tennisplayer', 'tricycle', 'truck1', 'truck2', 'truck3', 'valleyballplayer1', 'valleyballplayer2']
        return sequence_list
