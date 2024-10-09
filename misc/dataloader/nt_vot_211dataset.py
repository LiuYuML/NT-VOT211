import numpy as np
import os
import json
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import pandas as pd


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


class NT_VOT211Dataset(BaseDataset):
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.nt_vot211_path
        self.sequence_list = self._get_sequence_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_list = self._filter_sequence_list_by_attribute(attribute, self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/GT/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        target_visible = 1

        frames_path = '{}/sequences/{}'.format(self.base_path, sequence_name)

        frames_list = ['{}/img_{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'nt-vot211', ground_truth_rect.reshape(-1, 4), target_visible=target_visible)

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
        sequence_list = ['Achilles_Heel_Left', 'Achilles_Heel_Right', 'all_black_man', 'a_moving_object', 'a_slow_walker', 'a_walking_girl', 'a_white_fast_car', 'backpacker_and_his_white_bag', 'badminton_ doubles_in_9_PM', 'badminton_ doubles_in_9_PM_2', 'basketball', 'basketball_in_solo_practice', 'basketball_in_solo_practice2', 'baskteball_in_the_night', 'bike_among_building', 'black_car_downhill', 'black_coat_girl', 'black_long_haired_girl_on_her_way_to_dormitory', 'black_shirt_boy', 'black_shirt_with_black_pants_boy', 'black_top_with_white_shoes_boy', 'blue_player_on_the_play_ground', 'blurred_pedestrian_cross_the_street', 'boy_in_volleyball_competition', 'brother_grapefruit_wonder_around', 'brown_girl_on_electronic_mobile', 'bubble_tea_on_the_hand', 'bubble_tea_on_trash_bin', 'cap_guy', 'car_in_the_shadow', 'car_on_the_main_avenue', 'chainlink_fence_basketball', 'conference_of_four', 'darkgray_shirt_boy', 'drifting_white_bike', 'elegant_runner', 'elegant_walker', 'fast_bike', 'fast_bike2', 'fast_biker3', 'fast_bike_under_bright_light', 'fast_black_biker', 'fast_white_bicycle', 'female_runner', 'floodlight_in_campus', 'food_truck', 'food_trucks_off_duty', 'football_girl_in_blue', 'football_on_the_field', 'football_on_the_field2', 'football_on_the_play_ground', 'football_on_the_play_ground2', 'football_on_the_play_ground3', 'football_on_the_play_ground4', 'FW_runner', 'garbage_collection_truck', 'girls_together', 'girl_alone_in_the_night', 'girl_in_red_uphill', 'girl_in_volleyball_competition', 'girl_in_white_suit', 'girl_moving_towards_camera', 'gray_pants_white_shoes_boy', 'gray_pants_white_shoes_girl', 'green_player_on_the_play_ground', 'group_of_stranger_1', 'group_of_stranger_2', 'group_of_stranger_3', 'Gujars_on_the_playground', 'Gujars_on_the_street', 'Gujars_outside the window', 'helicopter_in_the_dawn', 'HeNan_bro_in_dark_corridor', 'his_a_lone_ranger', 'Indistinct_light', 'Is_that_a_trash_car', 'it_takes_two_1', 'it_takes_two_2', 'jump_like_Ali', 'keeping_strolling_on_runway', 'kind_man_on_the_road', 'large_trucks_outside_the_campus', 'least_valuable_player_in_xju', 'lighting_arm_walking', 'lighting_bus1', 'lighting_bus2', 'light_spot_on_the_street', 'lonely_walker', 'lovers_in_campus', 'lovers_outside the window', 'male_runner', 'man_in_raining', 'man_in_the_group', 'man_in_the_group2', 'man_leaving_the_library', 'man_moving_around', 'mineral_water_on_trash_bin', 'moon_in_the_sky', 'moon_in_the_sky2', 'moon_on_national_day', 'night_volleyball_competition', 'night_walker_with_his_phone', 'oh_shift_here_we_go_again', 'On_the_way_back_to_dormitory1', 'On_the_way_back_to_dormitory2', 'on_the_way_back_to_student_dormitory', 'orange_crane', 'pacer_under_bright_light', 'pacer_under_bright_light2', 'pacer_under_bright_light3', 'pacer_under_bright_light2_another_pacer', 'passenger_from_library', 'passenger_on_street_1', 'passenger_on_street_2', 'passenger_with_his_phone', 'passer_under_lantern', 'pathway_walker1', 'pathway_walker2', 'pathway_walker3', 'pattern_oh_the_trash_bin', 'pedestrian_on_street', 'perfect_runner', 'player_in_black_on_the_play_ground4', 'player_in_green_on_the_field', 'quiz_on_basketball_shooting', 'raining_day_passed_by', 'raining_night_two_man1', 'raining_night_two_man2', 'red_car_on_the_street', 'red_clothed_gatekeeper', 'red_clothed_gatekeeper2', 'red_electric_bicycle_rider', 'red_mobile_car_in_shadow', 'red_shirt_boy', 'red_shirt_walker_in_bright_light', 'red_shirt_with_black_pants_boy', 'red_sign_on_the_street', 'red_taxi', 'roam_around_pillar_lamp_1', 'roam_around_pillar_lamp_2', 'roam_around_pillar_lamp_3', 'runner_fade_in', 'runner_in_black_shirt', 'runner_in_midnight_1', 'runner_in_midnight_2', 'scenes_around_the_faculty_building1', 'shadow_fade_in', 'shadow_of_mine', 'shaking_lantern', 'shaking_light2', 'shaking_lisght', 'shuttle_bus_in_campus', 'shuttle_run_red_runner', 'silver_car_in_dazzling_light', 'single_snow_walker', 'slow_biker_uphill', 'slow_driving_electronic_mobile', 'smurf_is_getting_away', 'smurf_on_the_playground', 'snow_walker1', 'snow_walker2', 'solo_on_the_playground', 'solo_practice', 'soybean_milk_on_trash_bin', 'stranger_solo', 'streetball_in_xju', 'streetball_in_xju_basketball', 'strolling_twins', 'sunflower_that_looks_like_Dugtrio', 'third_wheel_in_dating_1', 'third_wheel_in_dating_2', 'tiny_marathon_in_midnight', 'twins_in_badminton1', 'twins_in_badminton2', 'twins_in_xju1', 'twins_in_xju2', 'unknown_girl', 'unknown_lighting_object1', 'unknown_lighting_object2', 'vertical_lighting_tube_1', 'vertical_lighting_tube_2', 'vertical_striped_jacket_boy', 'vertical_wick', 'vertical_wick2', 'vertical_window', 'very_nice_candy_text', 'walker_far_away', 'walking_girl', 'wander_man_around_buildings', 'white_bag_boy', 'white_car_disappeared_round_a_corner', 'white_car_in_jam_road', 'white_car_in_the_crowd', 'white_car_on_slow_wheel', 'white_car_on_street', 'white_man_on_street1', 'white_man_on_street2', 'white_mobile_car_on_the_street', 'white_shirt_black_pants_white_shoes_boy', 'white_shirt_boy', 'white_shirt_with_black_pants_boy', 'white_sweatshirt_boy', 'white_walking_man_on_street', 'windows_in_the_dormitory', 'window_of_dormitory', 'window_of_dormitory2', 'window_of_dormitory3', 'window_on_building', 'window_on_building2', 'xju_avenue_1_1.mp4', 'zebra_crossing']
        return sequence_list
