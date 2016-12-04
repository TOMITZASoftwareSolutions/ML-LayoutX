from unittest import TestCase
import numpy as np
import json


class BuidingPlanningManager():
    def create_and_save_plans(self, map_2d, origin_x, origin_z, file_name_base='buildings/building'):
        building_plans = self.create_building_plans(map_2d, origin_x, origin_z)
        for i, plan in enumerate(building_plans):
            building_plan, building_coord = building_plans[i]
            filename = '{0}-{1}'.format(file_name_base, i + 1)
            np.savetxt(filename, building_plan.T, fmt='%d', delimiter=',')
            coordinates_filename = '{0}-{1}.xyz'.format(file_name_base, i + 1)
            with open(coordinates_filename, mode='w') as fh:
                json.dump(building_coord, fh)

    def create_building_plans(self, map_2d, origin_x, origin_z):
        already_found_blocks = []
        buildings = []
        while True:
            building_blocks, building_plan, building_coordinates = self.create_building_plan(map_2d, origin_x, origin_z,
                                                                                             already_found_blocks)
            if building_plan is not None:
                buildings.append((building_plan, building_coordinates))
            else:
                break
        return buildings

    def create_building_plan(self, map_2d, origin_x, origin_z, already_found_blocks=[]):
        building_blocks = self.search_building_blocks(map_2d, already_found_blocks)
        building_plan = None
        building_coordinates = dict()
        if building_blocks:
            x_min, x_max, z_min, z_max = self.find_extremes(building_blocks)
            building_coordinates['x_min'] = x_min + origin_x
            building_coordinates['z_min'] = z_min + origin_z
            building_coordinates['x_max'] = x_max + origin_x
            building_coordinates['z_max'] = z_max + origin_z
            building_plan = np.zeros((x_max - x_min + 1, z_max - z_min + 1))
            for x in range(x_min, x_max + 1):
                for y in range(z_min, z_max + 1):
                    if map_2d[x, y] == 'brick_block':
                        building_plan[x - x_min, y - z_min] = 2
                    elif map_2d[x, y] == 'carpet':
                        building_plan[x - x_min, y - z_min] = 1

            for coord in building_blocks:
                already_found_blocks.append(coord)

        return building_blocks, building_plan, building_coordinates

    def search_building_blocks(self, map_2d, already_found_blocks):
        building_blocks = []
        start_x, start_y = self.search_start_point(map_2d, already_found_blocks)
        if start_x is not None and start_y is not None:
            building_blocks.append((start_x, start_y))
            self.goto_next_block(start_x, start_y, map_2d, building_blocks)

        return building_blocks

    def search_start_point(self, map_2d, already_found_blocks):
        width, length = map_2d.shape
        for i in range(0, width):
            for j in range(0, length):
                if map_2d[i, j] == 'brick_block' and (i, j) not in already_found_blocks:
                    return i, j
        return None, None

    def goto_next_block(self, x, y, map_2d, building_blocks):
        height, width = map_2d.shape
        for i, j in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            next_x = x + i
            next_y = y + j
            if next_x < 0 or next_y < 0 or next_x == height or next_y == width:
                continue
            current_block = map_2d[next_x, next_y]
            if current_block == 'brick_block' and (next_x, next_y) not in building_blocks:
                building_blocks.append((next_x, next_y))
                self.goto_next_block(next_x, next_y, map_2d, building_blocks)

    def find_extremes(self, building_blocks):
        x_min, x_max, y_min, y_max = None, None, None, None
        for block in building_blocks:
            x, y = block
            if x_min is None or x < x_min:
                x_min = x
            if x_max is None or x > x_max:
                x_max = x
            if y_min is None or y < y_min:
                y_min = y
            if y_max is None or y > y_max:
                y_max = y
        return x_min, x_max, y_min, y_max


class BuildingPlanningTest(TestCase):
    def setUp(self):
        self.map_2d = np.array([['air', 'brick_block', 'brick_block', 'brick_block', 'air', 'air', 'air', 'air'],
                                ['air', 'brick_block', 'carpet', 'brick_block', 'air', 'brick_block', 'brick_block',
                                 'brick_block'],
                                ['air', 'brick_block', 'carpet', 'brick_block', 'air', 'brick_block', 'carpet',
                                 'brick_block'],
                                ['air', 'brick_block', 'brick_block', 'brick_block', 'air', 'brick_block', 'air',
                                 'brick_block'],
                                ['air', 'air', 'air', 'air', 'air', 'air', 'air', 'air'],
                                ['air', 'air', 'air', 'air', 'air', 'air', 'air', 'air']
                                ])
        self.building_manager = BuidingPlanningManager()

    def test_search_start(self):
        start_x, start_y = self.building_manager.search_start_point(self.map_2d, [])
        self.assertEqual(start_x, 0)
        self.assertEqual(start_y, 1)

    def test_search_building(self):
        building_blocks = self.building_manager.search_building_blocks(self.map_2d, [])

        self.assertEqual(len(building_blocks), 10)
        self.assertListEqual(building_blocks,
                             [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1)])

    def test_create_building_plan(self):
        building_2d_expected = np.array([[2, 2, 2],
                                         [2, 1, 2],
                                         [2, 1, 2],
                                         [2, 2, 2]
                                         ])

        building_block, building_2d, building_coord = self.building_manager.create_building_plan(self.map_2d, 0, 0)

        np.testing.assert_equal(building_2d, building_2d_expected)

    def test_create_multiple_building(self):
        building_plans = self.building_manager.create_building_plans(self.map_2d, 0, 0)

        self.assertEqual(2, len(building_plans))

        second_building_2d = np.array([[2, 2, 2],
                                       [2, 1, 2],
                                       [2, 0, 2]]
                                      )

        np.testing.assert_equal(building_plans[1][0], second_building_2d)

    def test_l_shape_building(self):
        map_2d = np.array([
            ['brick_block', 'brick_block', 'brick_block', 'brick_block', 'brick_block', 'brick_block', 'brick_block'],
            ['brick_block', 'carpet', 'carpet', 'carpet', 'carpet', 'carpet', 'brick_block'],
            ['brick_block', 'carpet', 'carpet', 'carpet', 'brick_block', 'brick_block', 'brick_block'],
            ['brick_block', 'carpet', 'carpet', 'carpet', 'brick_block', 'air', 'air'],
            ['brick_block', 'carpet', 'carpet', 'carpet', 'brick_block', 'air', 'air'],
            ['brick_block', 'brick_block', 'brick_block', 'brick_block', 'brick_block', 'air', 'air']
        ])

        building_plans = self.building_manager.create_building_plans(map_2d, 0, 0)

        building_plans_expected = np.array([
            [2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 2, 2, 2],
            [2, 1, 1, 1, 2, 0, 0],
            [2, 1, 1, 1, 2, 0, 0],
            [2, 2, 2, 2, 2, 0, 0],

        ])

        np.testing.assert_equal(building_plans[0][0], building_plans_expected)

    def test_create_files(self):
        self.building_manager.create_and_save_plans(self.map_2d, 0, 0, file_name_base='test_buildings/building')
