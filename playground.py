import MalmoPython
import os
import sys
import time
import random
import json
import numpy as np
from managers.building_planning import BuidingPlanningManager
from managers.observation_to_grid import ObservationConverter
from managers.distance_estimator_image_generator import DistanceEstimatorImageGeneratorManager
import xmltodict
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set to INFO if you want fewer messages

mission_file = 'mission'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    mission_info = xmltodict.parse(mission_xml)

agent_host = MalmoPython.AgentHost()

try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print 'ERROR:', e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)

my_mission_record = MalmoPython.MissionRecordSpec('recordings/recordings')

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print "Error starting mission:", e
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print "Waiting for the mission to start ",
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print "Error:", error.text

print
print "Mission running ",


def create_building_plans(observations, mission_info):
    grid = observations['agent_surround']

    grid_info = mission_info['Mission']['AgentSection']['AgentHandlers']['ObservationFromGrid']['Grid']
    agent_placement_info = mission_info['Mission']['AgentSection']['AgentStart']['Placement']

    observation_converter = ObservationConverter()
    map_2d = observation_converter.convert_observation_to_2d_map(grid, grid_info)
    center_x, center_y = observation_converter.calculate_map_origins(grid_info, agent_placement_info)

    building_planning_manager = BuidingPlanningManager()
    building_planning_manager.create_and_save_plans(map_2d, center_x, center_y)


distance_image_generator = DistanceEstimatorImageGeneratorManager()

coordinates = distance_image_generator.generate_coordinates()
index = 0

# Loop until mission ends:
while world_state.is_mission_running:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    if world_state.number_of_observations_since_last_state > 0:  # Have any observations come in?
        msg = world_state.observations[-1].text  # Yes, so get the text
        observations = json.loads(msg)

        coordinate = coordinates[index]
        index += 1
        x = coordinate['x']
        y = coordinate['y']
        z = coordinate['z']
        yaw = coordinate['yaw']
        pitch = coordinate['pitch']
        agent_host.sendCommand("tp {0} {1} {2}".format(x, y, z))
        agent_host.sendCommand("setYaw {0}".format(yaw))
        agent_host.sendCommand("setPitch {0}".format(pitch))

        if index % 100 == 0:
            print 'Progress {0}/{1}'.format(index, len(coordinates))

        time.sleep(0.2)

        world_state = agent_host.getWorldState()
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            logger.info("Waiting for frames...")
            time.sleep(0.05)
            world_state = agent_host.getWorldState()

        pixels = world_state.video_frames[0].pixels
        frame_height = int(mission_info['Mission']['AgentSection']['AgentHandlers']['VideoProducer']['Height'])
        frame_width = int(mission_info['Mission']['AgentSection']['AgentHandlers']['VideoProducer']['Width'])
        frame = np.array(pixels).reshape((frame_height, frame_width, 3))

        distance_image_generator.save(coordinate, frame)

    for error in world_state.errors:
        print "Error:", error.text

print
print "Mission ended"


# TODO create agent hosts

# TODO create mission
