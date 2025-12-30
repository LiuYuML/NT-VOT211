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


class DarkTrack2021Dataset(BaseDataset):
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.DarkTrack2021_path
        self.sequence_list = self._get_sequence_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_list = self._filter_sequence_list_by_attribute(attribute, self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        target_visible = 1

        frames_path = '{}/data_seq/{}'.format(self.base_path, sequence_name)

        jpg_files = glob.glob(frames_path + '/*.jpg')
        jpg_count = len(jpg_files)
        frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, jpg_count + 1)]

        return Sequence(sequence_name, frames_list, 'darktrack2021', ground_truth_rect.reshape(-1, 4), target_visible=target_visible)

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
        sequence_list = ['car_15', 'building_2', 'bus_1', 'bus_2', 'bus_3', 'bus_4', 'bus_5', 'bus_6', 'car_1', 'car_10', 'car_11', 'car_12', 'car_13', 'car_14', 'car_15', 'car_16', 'car_17', 'car_18', 'car_19', 'car_2', 'car_20', 'car_21', 'car_22', 'car_23', 'car_24', 'car_25', 'car_26', 'car_27', 'car_28', 'car_29', 'car_3', 'car_30', 'car_31', 'car_32', 'car_33', 'car_34', 'car_35', 'car_36', 'car_37', 'car_38', 'car_39', 'car_4', 'car_40', 'car_41', 'car_42', 'car_43', 'car_44', 'car_5', 'car_6', 'car_7', 'car_8', 'car_9', 'dog_1', 'motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'person_1', 'person_10', 'person_11', 'person_12', 'person_13', 'person_14', 'person_15', 'person_16', 'person_17', 'person_18', 'person_19', 'person_2', 'person_20', 'person_21', 'person_22', 'person_23', 'person_24', 'person_25', 'person_26', 'person_27', 'person_28', 'person_29', 'person_3', 'person_30', 'person_31', 'person_32', 'person_33', 'person_34', 'person_4', 'person_5', 'person_6', 'person_7', 'person_8', 'person_9', 'skating_1', 'skating_2', 'taxi_1', 'taxi_2', 'truck_1', 'truck_10', 'truck_11', 'truck_12', 'truck_13', 'truck_2', 'truck_3', 'truck_4', 'truck_5', 'truck_6', 'truck_7', 'truck_8', 'truck_9', 'van_1']
        return sequence_list
