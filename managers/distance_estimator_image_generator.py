import numpy as np
from unittest import TestCase
import itertools
import json
from PIL import Image
import os
import random


class DistanceEstimatorImageGeneratorManager():
    '''
        generating images only on vertical for now (z)
    '''

    def generate_coordinates(self, training_buildings=['buildings/building-1', 'buildings/building-2','buildings/building-3'],
                             validation_buildings=['buildings/building-4']):
        all_coordinates = []

        for building in training_buildings:
            building_2d, origin_x, origin_z = self.load_building_2d_for_and_coordinates(building)
            building_name = building.split('/')[1]
            coordinates = self.generate_coordinates_for_building(building_2d, origin_x, origin_z, True, building_name)
            all_coordinates.append(coordinates)

        for building in validation_buildings:
            building_2d, origin_x, origin_z = self.load_building_2d_for_and_coordinates(building)
            building_name = building.split('/')[1]
            coordinates = self.generate_coordinates_for_building(building_2d, origin_x, origin_z, False, building_name)
            all_coordinates.append(coordinates)

        all_coordinates = list(itertools.chain(*all_coordinates))

        return all_coordinates

    def load_building_2d_for_and_coordinates(self, filename):
        with open('{0}.xyz'.format(filename)) as f:
            coordinates = json.load(f)

        building_2d = np.loadtxt(filename, dtype=int, delimiter=',')
        return building_2d, coordinates['x_min'], coordinates['z_min']

    def generate_coordinates_for_building(self, building_2d, origin_x, origin_z, train, building):
        coordinates = []

        for (z, x), value in np.ndenumerate(building_2d):
            if value == 1:
                distance = self.calculate_distance(z, x, building_2d)
                x = x + origin_x + 0.5
                z = z + origin_z + 0.5

                # coordinate = self.create_coordinate(x, z, train, distance, 0, building)
                spiced_coordinates = self.spice_coordinates(x, z, train, distance, 0, building)
                coordinates += spiced_coordinates

        return coordinates

    def calculate_distance(self, z, x, building_2d):
        slice = building_2d[:, x]
        distance = 0
        for i in range(z + 1, len(slice)):
            if slice[i] != 1:
                break
            else:
                distance += 1
        return distance

    def spice_coordinates(self, x, z, train, distance, yaw, building):

        delta_xs = [x + random.uniform(-0.4, 0.4) for i in range(3)]
        delta_zs = [z + random.uniform(-0.4, 0.4) for i in range(3)]
        delta_pitchs = [0 + random.uniform(-7, 7) for i in range(3)]
        delta_yaws = [yaw + random.uniform(-5, 5) for i in range(3)]

        coordinates = [self.create_coordinate(dx, dz, train, distance, dyaw, building, dpitch) for dx, dz, dpitch, dyaw
                       in itertools.product(delta_xs, delta_zs, delta_pitchs, delta_yaws)]

        coordinates.append(self.create_coordinate(x, z, train, distance, yaw, building))

        return coordinates

    def create_coordinate(self, x, z, train, distance, yaw, building, pitch=0):
        coordinate = dict()
        coordinate['x'] = x
        coordinate['z'] = z
        coordinate['y'] = 4
        coordinate['pitch'] = pitch
        coordinate['yaw'] = yaw
        coordinate['train'] = train
        coordinate['distance'] = distance
        coordinate['building'] = building
        return coordinate

    def save(self, coordinate, image):
        im = Image.fromarray(image)
        distance = coordinate['distance']
        train = 'train' if coordinate['train'] else 'validate'

        folder = 'data/{0}/{1}'.format(train, distance)

        if not os.path.exists(folder):
            os.makedirs(folder)

        image_path = '{0}/{1}_{2}_{3}_{4}_{5}_{6}.jpg'.format(folder, coordinate['building'], coordinate['x'],
                                                              coordinate['y'], coordinate['z'], coordinate['yaw'],
                                                              coordinate['pitch'])

        im.save(image_path)


class DistanceEstimatorImageGeneratorManagerTest(TestCase):
    def setUp(self):
        self.image_generator = DistanceEstimatorImageGeneratorManager()
        self.building_2d = np.array([
            [2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2],
            [2, 1, 1, 2, 2, 2],
            [2, 2, 2, 2, 0, 0],
        ])

    def test_generate_coordinates(self):
        origin_x = 0
        origin_z = -12

        coordinates = self.image_generator.generate_coordinates_for_building(self.building_2d, origin_x, origin_z, True,
                                                                             'test')

        self.assertEqual(10, len(coordinates))

        self.assertEqual(coordinates[0]['distance'], 2)
        self.assertEqual(coordinates[2]['distance'], 1)
        self.assertEqual(coordinates[4]['distance'], 1)

        # there's a offset of 1 on both axis... not sure how to mathematically explain
        self.assertEqual(coordinates[4]['z'], -9.5)

    def test_generate_coordinates_full(self):
        training_buildings = ['test_buildings/building-1']
        validation_buildings = ['test_buildings/building-2']
        coordinates = self.image_generator.generate_coordinates(training_buildings, validation_buildings)

        self.assertEqual(len(coordinates), 3)
        self.assertEqual(coordinates[2]['train'], False)
        self.assertEqual(coordinates[0]['train'], True)
        self.assertEqual(coordinates[0]['distance'], 0)

    def test_spice_coordinates(self):
        coordinates = self.image_generator.spice_coordinates(0, 0, False, 0, 0, 'whatever')

        self.assertEqual(len(coordinates), 3 * 3 * 5 * 5 + 1)

        self.assertTrue(-0.5 < coordinates[0]['x'] < 0.5)
        self.assertTrue(-0.5 < coordinates[1]['z'] < 0.5)
        self.assertEqual(len(set([coordinate['x'] for coordinate in coordinates])),4)
        self.assertEqual(len(set([coordinate['z'] for coordinate in coordinates])),4)
        self.assertEqual(len(set([coordinate['yaw'] for coordinate in coordinates])),6)
        self.assertEqual(len(set([coordinate['pitch'] for coordinate in coordinates])),6)
        self.assertTrue(coordinates[0]['yaw'] != 0)
        self.assertTrue(coordinates[5]['pitch'] != 0)
