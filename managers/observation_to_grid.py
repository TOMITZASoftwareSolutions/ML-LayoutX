from unittest import TestCase
import numpy as np


class ObservationConverter():
    def convert_observation_to_2d_map(self, observation_1d, observation_params):
        min_x = int(observation_params['min']['@x'])
        min_z = int(observation_params['min']['@z'])
        max_x = int(observation_params['max']['@x'])
        max_z = int(observation_params['max']['@z'])

        map_2d = np.reshape(observation_1d, (max_x - min_x + 1, max_z - min_z + 1)).T

        return map_2d

    def calculate_map_origins(self, observation_params, agent_placement_params):
        min_x = int(observation_params['min']['@x'])
        min_z = int(observation_params['min']['@z'])

        agent_x = int(agent_placement_params['@x'])
        agent_z = int(agent_placement_params['@z'])

        return agent_x + min_x, agent_z + min_z


class ObservationToGridTest(TestCase):
    def setUp(self):
        self.observation_1d = ['air', 'air', 'air', 'building_block', 'carpet', 'building_block', 'air', 'air', 'air']
        self.observation_params = {
            'min': {
                '@x': '-1',
                '@y': '0',
                '@z': '-1'
            },
            'max': {
                '@x': '1',
                '@y': '0',
                '@z': '1'
            }
        }

        self.agent_placement = {
            '@x': '0',
            '@z': '0',
            '@y': '4'
        }

        self.observation_converter = ObservationConverter()

    def test_conversion(self):
        expected_2d = np.array([
            ['air', 'building_block', 'air'],
            ['air', 'carpet', 'air'],
            ['air', 'building_block', 'air'],
        ])

        map_2d = self.observation_converter.convert_observation_to_2d_map(self.observation_1d, self.observation_params)

        np.testing.assert_equal(map_2d, expected_2d)

    def test_origin_calculation(self):
        expected_x, expected_y = -1, -1

        center_x, center_y = self.observation_converter.calculate_map_origins(self.observation_params,
                                                                              self.agent_placement)

        self.assertEqual(expected_x, center_x)
        self.assertEqual(expected_y, center_y)
