from __future__ import absolute_import
from __future__ import print_function
from PyQt5 import QtWidgets, QtCore
import threading

import random
import numpy as np
import tqdm
import os
import sys
import optparse
import shutil
import pickle
import NNagent
import detect
from os import walk
import time

MIN_BATCH_SIZE = 64
gamma = 0.99
MIN_TARGET_COUNTER = 3
MEMORY_SIZE = 20000
episodes = 20000
epsilon_decay = 0.00005
PRE_TRAIN_STEPS = 500
learning_rate = 0.001
TARGET_NETWORK_alpha = 0.001
leaky_relu_rate = 0.01

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def generate_routefile(i):
    random.seed(i)
    #print("random %d" % i)
    with open("data/cross.rou.xml", "w") as routes:
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<routes>""", file=routes)
        print("""<vTypeDistribution id="mixed">
                    <vType id="car" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                    <vType id="bus" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.3" latAlignment="compact" probability="{}"/>
                    <vType id="truck" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed1">
                    <vType id="car1" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle1" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="bus1" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="truck1" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                    <vType id="sedan1" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.4" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped1" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed2">
                    <vType id="car2" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.3" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle2" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="bus2" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.2" latAlignment="compact" probability="{}"/>
                    <vType id="truck2" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan2" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped2" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed3">
                    <vType id="car3" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle3" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="bus3" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="truck3" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan3" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.3" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped3" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        print("""<routes>
                <routeDistribution id="r0" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="3"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="19"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="3"/>
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="3"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="19"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="3"/>
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="19"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="3"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="3"/>
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="3"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="3"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="19"/>
                </routeDistribution>
                <routeDistribution id="r1" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="4"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="17"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="4"/>
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="4"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="17"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="4"/>
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="17"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="4"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="4"/>
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="4"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="4"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="17"/>
                </routeDistribution>
                <routeDistribution id="r2" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="15"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="70"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="15"/>
                </routeDistribution>
                <routeDistribution id="r3" departSpeed="random">             
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="60"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="20"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="20"/>                
                </routeDistribution>
                <routeDistribution id="r4" departSpeed="random">                
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="20"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="60"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="20"/>                                
                </routeDistribution>
                <routeDistribution id="r5" departSpeed="random">                                                
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="25"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="25"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="50"/>
                </routeDistribution>
            </routes>        
            <flow id="mixed0" begin="0" number="100" vehsPerHour="500" route="r0" type="mixed3" departLane="random" departPosLat="random"/>
            <flow id="mixed5" begin="0" number="100" vehsPerHour="500" route="r5" type="mixed2" departLane="random" departPosLat="random"/>
            <flow id="mixed1" begin="50" number="100" vehsPerHour="500" route="r4" type="mixed1" departLane="random" departPosLat="random"/>
            <flow id="mixed3" begin="50" number="100" vehsPerHour="500" route="r2" type="mixed" departLane="random" departPosLat="random"/>
            <flow id="mixed4" begin="100" number="100" vehsPerHour="500" route="r3" type="mixed1" departLane="random" departPosLat="random"/>
            <flow id="mixed" begin="100" number="100" vehsPerHour="500" route="r1" type="mixed3" departLane="random" departPosLat="random"/>
            <flow id="mixed2" begin="300" number="100" vehsPerHour="500" route="r3" type="mixed2" departLane="random" departPosLat="random"/>
            <flow id="mixed6" begin="300" number="100" vehsPerHour="500" route="r5" type="mixed" departLane="random" departPosLat="random"/>""",
              file=routes)
        print("</routes>", file=routes)

def generate_test_integrate(i):
    random.seed(i)
    #print("random %d" % i)
    with open("data/cross.rou.xml", "w") as routes:
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<routes>""", file=routes)
        print("""<vTypeDistribution id="mixed">
                    <vType id="car" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                    <vType id="bus" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.3" latAlignment="compact" probability="{}"/>
                    <vType id="truck" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed1">
                    <vType id="car1" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle1" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="bus1" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="truck1" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                    <vType id="sedan1" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.4" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped1" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed2">
                    <vType id="car2" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.3" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle2" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="bus2" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.2" latAlignment="compact" probability="{}"/>
                    <vType id="truck2" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan2" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.5" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped2" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        p = []
        p.append(random.randint(32, 43))
        p.append(random.randint(40, 55))
        p.append(random.randint(5, 8))
        p.append(random.randint(5, 8))
        p.append(random.randint(10, 15))
        p.append(random.randint(10, 15))
        print("""<vTypeDistribution id="mixed3">
                    <vType id="car3" vClass="passenger" speedDev="0.2" minGap = "1" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="motorcycle3" vClass="motorcycle" speedDev="0.4" minGap = "0.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="bus3" vClass="bus" speedDev="0.15" minGap = "1.5" sigma = "0.5" latAlignment="compact" probability="{}"/>
                    <vType id="truck3" vClass="truck" speedDev="0.1" minGap = "1.5" sigma = "0.4" latAlignment="compact" probability="{}"/>
                    <vType id="sedan3" vClass="taxi" speedDev="0.2" minGap = "1" sigma = "0.3" latAlignment="compact" probability="{}" color="grey"/>
                    <vType id="moped3" vClass="moped" speedDev="0.4" minGap = "0.5" sigma = "0.6" latAlignment="compact" probability="{}"/>
                </vTypeDistribution>""".format(p[0], p[1], p[2], p[3], p[4], p[5]), file=routes)
        print("""<routes>                
                <routeDistribution id="r0" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="3"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="19"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="3"/>
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="3"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="19"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="3"/>
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="19"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="3"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="3"/>
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="3"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="3"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="19"/>
                </routeDistribution>
                <routeDistribution id="r1" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="4"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="17"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="4"/>
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="4"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="17"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="4"/>
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="17"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="4"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="4"/>
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="4"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="4"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="17"/>
                </routeDistribution>
                <routeDistribution id="r2" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="51o 1o 4i 54i" probability="20"/>
                    <route id="route1" color="1,1,0" edges="51o 1o 2i 52i" probability="60"/>
                    <route id="route2" color="1,1,0" edges="51o 1o 3i 53i" probability="20"/>
                </routeDistribution>
                <routeDistribution id="r3" departSpeed="random">             
                    <route id="route6" color="1,1,0" edges="52o 2o 1i 51i" probability="60"/>
                    <route id="route7" color="1,1,0" edges="52o 2o 4i 54i" probability="20"/>
                    <route id="route8" color="1,1,0" edges="52o 2o 3i 53i" probability="20"/>                
                </routeDistribution>
                <routeDistribution id="r4" departSpeed="random">                
                    <route id="route3" color="1,1,0" edges="53o 3o 1i 51i" probability="20"/>
                    <route id="route4" color="1,1,0" edges="53o 3o 4i 54i" probability="60"/>
                    <route id="route5" color="1,1,0" edges="53o 3o 2i 52i" probability="20"/>                                
                </routeDistribution>
                <routeDistribution id="r5" departSpeed="random">                                                
                    <route id="route9" color="1,1,0" edges="54o 4o 1i 51i" probability="20"/>
                    <route id="route10" color="1,1,0" edges="54o 4o 2i 52i" probability="20"/>
                    <route id="route11" color="1,1,0" edges="54o 4o 3i 53i" probability="60"/>
                </routeDistribution>             
                <routeDistribution id="r1o" departSpeed="random">
                    <route id="route0" color="1,1,0" edges="1o 4i" probability="20"/>
                    <route id="route1" color="1,1,0" edges="1o 2i" probability="60"/>
                    <route id="route2" color="1,1,0" edges="1o 3i" probability="20"/>
                </routeDistribution>
                <routeDistribution id="r2o" departSpeed="random">             
                    <route id="route6" color="1,1,0" edges="2o 1i" probability="60"/>
                    <route id="route7" color="1,1,0" edges="2o 4i" probability="20"/>
                    <route id="route8" color="1,1,0" edges="2o 3i" probability="20"/>                
                </routeDistribution>
                <routeDistribution id="r3o" departSpeed="random">                
                    <route id="route3" color="1,1,0" edges="3o 1i" probability="20"/>
                    <route id="route4" color="1,1,0" edges="3o 4i" probability="60"/>
                    <route id="route5" color="1,1,0" edges="3o 2i" probability="20"/>                                
                </routeDistribution>
                <routeDistribution id="r4o" departSpeed="random">                                                
                    <route id="route9" color="1,1,0" edges="4o 1i" probability="20"/>
                    <route id="route10" color="1,1,0" edges="4o 2i" probability="20"/>
                    <route id="route11" color="1,1,0" edges="4o 3i" probability="60"/>
                </routeDistribution>
            </routes>                          
            <flow id="mixed4" begin="10" number="1" vehsPerHour="1000" route="r0" type="mixed1" departLane="random" departPosLat="random"/>""", file=routes)
        print("</routes>", file=routes)

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def vehnums(phase, X):
    phase = phase // 2
    x = np.zeros((4, 6))

    x[3, 0] = traci.lanearea.getLastStepVehicleNumber("401") + traci.lanearea.getLastStepVehicleNumber("400")
    x[3, 1] = traci.lanearea.getLastStepVehicleNumber("411") + traci.lanearea.getLastStepVehicleNumber("410")
    x[3, 2] = traci.lanearea.getLastStepVehicleNumber("421") + traci.lanearea.getLastStepVehicleNumber("420")
    x[3, 3] = traci.lanearea.getLastStepVehicleNumber("431") + traci.lanearea.getLastStepVehicleNumber("430")

    x[2, 0] = traci.lanearea.getLastStepVehicleNumber("301") + traci.lanearea.getLastStepVehicleNumber("300")
    x[2, 1] = traci.lanearea.getLastStepVehicleNumber("311") + traci.lanearea.getLastStepVehicleNumber("310")
    x[2, 2] = traci.lanearea.getLastStepVehicleNumber("321") + traci.lanearea.getLastStepVehicleNumber("320")
    x[2, 3] = traci.lanearea.getLastStepVehicleNumber("331") + traci.lanearea.getLastStepVehicleNumber("330")

    x[1, 0] = traci.lanearea.getLastStepVehicleNumber("201") + traci.lanearea.getLastStepVehicleNumber("200")
    x[1, 1] = traci.lanearea.getLastStepVehicleNumber("211") + traci.lanearea.getLastStepVehicleNumber("210")
    x[1, 2] = traci.lanearea.getLastStepVehicleNumber("221") + traci.lanearea.getLastStepVehicleNumber("220")
    x[1, 3] = traci.lanearea.getLastStepVehicleNumber("231") + traci.lanearea.getLastStepVehicleNumber("230")

    x[0, 0] = traci.lanearea.getLastStepVehicleNumber("101") + traci.lanearea.getLastStepVehicleNumber("100")
    x[0, 1] = traci.lanearea.getLastStepVehicleNumber("111") + traci.lanearea.getLastStepVehicleNumber("110")
    x[0, 2] = traci.lanearea.getLastStepVehicleNumber("121") + traci.lanearea.getLastStepVehicleNumber("120")
    x[0, 3] = traci.lanearea.getLastStepVehicleNumber("131") + traci.lanearea.getLastStepVehicleNumber("130")

    #if phase == 0:
    X = np.zeros((6, 24))
    X[phase] = np.squeeze(x.reshape(24, 1))/100 # /np.maximum(np.sum(x), 1)
    #print()
    #print(f"{np.sum(x)}")
    # X = X.reshape(24 * 6, 1)

    return X.reshape(144, 1), X

def get_vehicle_numbers(phase):
    phase = phase // 2
    temp = []
    for i in range(24):
        if i % 6 < 4:
            temp.append(np.random.randint(3, 10))
        else:
            temp.append(0)
    temp = detect.predict_count()
    print(temp[:6])
    print(temp[6:12])
    print(temp[12:18])
    print(temp[18:])
    x = np.array(temp)
    X = np.zeros((6, 24))
    X[phase] = np.squeeze(x.reshape(24, 1))/100 # /np.maximum(np.sum(x), 1)

    return X.reshape(144, 1), temp


def normalrun(ft = 40):
    steps = 0
    phase = 0
    durs = 0
    vehsp = []
    vehlist = {}

    while traci.simulation.getMinExpectedNumber() > 0:
        if durs <= 0:
            traci.trafficlight.setPhase("0", phase)
            if phase % 2 == 0:
                traci.trafficlight.setPhaseDuration("0", 3)
                durs = 3
            else:
                traci.trafficlight.setPhaseDuration("0", ft)
                #time = traci.trafficlight.getPhaseDuration("0")
                durs = ft

            phase = (phase + 1) % 12

        traci.simulationStep()

        if durs > 0:
            durs -= 1
        steps += 1

    return steps, vehsp

def normalrunreward(ft = 40):
    steps = 0
    phase = 0
    durs = 0
    tots1 = 0
    tots = 0
    totreward = 0
    count = 0
    while traci.simulation.getMinExpectedNumber() > 0:

        if durs <= 0:
            traci.trafficlight.setPhase("0", phase)
            if phase % 2 == 0:
                traci.trafficlight.setPhaseDuration("0", 3)
                durs = 3
                tots1 = 0
                for i in traci.vehicle.getIDList():
                    tots1 += traci.vehicle.getWaitingTime(i)
            else:
                if steps > 3:
                    reward = tots - tots1
                    totreward += reward
                    count += 1

                tots = 0
                for i in traci.vehicle.getIDList():
                    tots += traci.vehicle.getWaitingTime(i)
                traci.trafficlight.setPhaseDuration("0", ft)
                #time = traci.trafficlight.getPhaseDuration("0")
                durs = ft

                # print("time %d" %time)

            # print("{} {} {}".format(steps, phase, durs))
            phase = (phase + 1) % 12

        traci.simulationStep()

        if durs > 0:
            durs -= 1
        steps += 1

    return steps, totreward/count

def trainagent():
    print("-------------train---------------")
    epsilon = 1
    epsilon_min = 0.01
    agent = NNagent.Agent()
    for episode in tqdm.tqdm(range(episodes), ascii=True, unit="episode"):
        totreward = 0
        step = 0
        #run()
        done = False
        generate_routefile(episode % 10)
        if episode < 10:
            traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
            cons, _ = normalrun()
            print("\ncons {}".format(cons))
            shutil.copy("tripinfo.xml", f"tripsdqn/tripinfoprim{episode}.xml")

        traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        phase = 0
        durs = 0
        tots = 0
        current_state = 0
        next_state = 0
        tots1 = 0
        X = np.zeros((6, 24))
        action = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            if durs <= 0:
                if phase % 2 == 0:
                    traci.trafficlight.setPhase("0", phase)
                    traci.trafficlight.setPhaseDuration("0", 3)
                    durs = 3
                    if step > 3:
                        tots1 = 0
                        for i in traci.vehicle.getIDList():
                            tots1 += traci.vehicle.getWaitingTime(i)

                else:
                    if step > 3:
                        reward = (tots - tots1)
                        totreward += reward
                        next_state, X = vehnums(phase, X)
                        agent.update_replay_memory((current_state, action, reward, next_state, done))

                        if PRE_TRAIN_STEPS < step:
                            agent.train(done)
                        current_state = next_state

                    if step <= 3:
                        current_state, X = vehnums(phase, X)

                    if np.random.random() <= epsilon:
                        action = np.random.randint(0, 48)

                    else:
                        action = np.argmax(agent.get_qs(current_state.reshape(1, 144)))
                        print("sugg-time {}".format(action + 2))

                    tots = 0
                    for i in traci.vehicle.getIDList():
                        tots += traci.vehicle.getWaitingTime(i)

                    traci.trafficlight.setPhase("0", phase)
                    durs = action + 2
                    traci.trafficlight.setPhaseDuration("0", action + 2)


                phase = (phase + 1) % 12

            traci.simulationStep()

            if durs > 0:
                durs -= 1
            step += 1

        print("\nepisode - {}, epsilon-{}, steps={}\n".format(episode, epsilon, step))
        shutil.copy("tripinfo.xml", f"tripsdqn/tripinfo-{episode}.xml")
        #current_state = next_state
        done = True
        agent.update_replay_memory((current_state, action, 0, next_state, done))
        agent.train(done)

        """
        if totreward/step > -1:
            if not os.path.isdir('models'):
                os.mkdir('models')
            agent.network.save(f'models/dddqn_traffic_{totreward/step}_{time.time()}.model')
        """
        if (episode + 1) % 500 == 0:
            if not os.path.isdir('models'):
                os.mkdir('models')
            agent.network.save(f'models/dddqn_traffic_{episode + 1}.model')
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay
            epsilon = max(epsilon, epsilon_min)
    agent.network.save(f'models/dddqn_traffic_final.model')

def trainmid():
    print("-------------train---------------")

    agent = NNagent.Agent()
    agent.network.load_weights(f'models/dddqn_traffic_final.model')
    epsilon = 0.01
    episodes = 10000
    rewardspa = []
    for episode in tqdm.tqdm(range(episodes), ascii=True, unit="episode"):

        totreward = 0
        step = 0
        # run()
        done = False
        generate_routefile(episode % 10)
        if episode < 10:
            traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
            cons, _ = normalrun()
            print("\ncons {}".format(cons))
            shutil.copy("tripinfo.xml", f"tripsdqn3/tripinfoprim{episode}.xml")

        traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        phase = 0
        durs = 0
        tots = 0
        current_state = 0
        next_state = 0
        tots1 = 0
        X = np.zeros((6, 24))
        action = 0
        count = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            if durs <= 0:
                if phase % 2 == 0:
                    traci.trafficlight.setPhase("0", phase)
                    traci.trafficlight.setPhaseDuration("0", 3)
                    durs = 3
                    if step > 3:
                        tots1 = 0
                        for i in traci.vehicle.getIDList():
                            tots1 += traci.vehicle.getWaitingTime(i)
                        # tots1 /= max(len(traci.vehicle.getIDList()), 1)
                else:
                    if step > 3:

                        reward = (tots - tots1)
                        """
                        if tots < tots1:
                            reward = -1
                        elif tots > tots1:
                            reward = 1
                        """
                        totreward += reward

                        next_state, X = vehnums(phase, X)
                        agent.update_replay_memory((current_state, action, reward, next_state, done))

                        if PRE_TRAIN_STEPS < step:
                            agent.train(done)
                        current_state = next_state

                    if step <= 3:
                        current_state, X = vehnums(phase, X)

                    if np.random.random() <= epsilon:
                        action = np.random.randint(0, 48)
                        # print("time {}".format(action + 2))
                    else:

                        action = np.argmax(agent.get_qs(current_state.reshape(1, 144)))
                        print("sugg-time {}".format(action + 2))
                    count += 1
                    tots = 0
                    for i in traci.vehicle.getIDList():
                        tots += traci.vehicle.getWaitingTime(i)
                        # tots /= max(len(traci.vehicle.getIDList()), 1)

                    traci.trafficlight.setPhase("0", phase)
                    durs = action + 2
                    traci.trafficlight.setPhaseDuration("0", action + 2)
                    # durs = traci.trafficlight.getPhaseDuration("0")

                phase = (phase + 1) % 12

            traci.simulationStep()

            if durs > 0:
                durs -= 1
            step += 1
        rewardspa.append(totreward/count)
        print("\nepisode - {}, epsilon-{}, steps={}\n".format(episode, epsilon, step))
        shutil.copy("tripinfo.xml", f"tripsdqn3/tripinfo-{episode}.xml")
        # current_state = next_state
        done = True
        agent.update_replay_memory((current_state, action, 0, next_state, done))
        agent.train(done)

        if (episode + 1) % 500 == 0:
            if not os.path.isdir('rewards'):
                os.mkdir('rewards')
            file = open("rewards/reward.bin", "wb")
            pickle.dump(rewardspa, file)
            file.close()
            if not os.path.isdir('models'):
                os.mkdir('models')
            agent.network.save(f'models/dddqn_traffic_20K+{episode + 1}.model')

def test():
    agent = NNagent.Agent()
    flag = 0
    agent.network.load_weights(f"dddqn_traffic_20K+1000.model")
    #agent.network.load_weights(f"models/dddqn_traffic_15000.model")
    st = []
    co = []

    for i in tqdm.tqdm(range(100), ascii=True, unit="episode"):
        generate_routefile(i)
        if i < 100:
            cons = 0
            #traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
            #cons, x = normalrun()
            #co.append(cons)
            # print("\ncons {}".format(cons))
        traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        step = 0
        phase = 0
        durs = 0

        current_state = 0
        X = np.zeros((6, 24))
        
        def something():
            app = QtWidgets.QApplication(sys.argv)
            main = QtWidgets.QFrame()
            main.setGeometry(1200, 400, 450, 120)
            main.setStyleSheet("QLabel{font-size:20px;}")
            l1 = QtWidgets.QLabel("0.00000", main)
            l1.setGeometry(0, 0, 400, 40)
            l2 = QtWidgets.QLabel("South", main)
            l2.setGeometry(0, 40, 300, 40)
            l3 = QtWidgets.QLabel("Yellow", main)
            l3.setGeometry(0, 80, 450, 40)

            def getphasename(phase):
                color = "yellow"
                lanes = {0: "South", 1: "South-North", 2: "North", 3: "West",4: "West-East", 5: "East"}
                if phase % 2 == 0:
                    color = "green"

                return color, lanes[(phase - 1) % 12 // 2]

            def changet():
                l1.setText(f"Time remaining for change: {durs}")

                color, lane = getphasename(phase)
                l2.setText(f"LANE: {lane} activated")
                l3.setText(f"LANE {lane} is running on {color.upper()} light")
            main.show()
            timer = QtCore.QTimer()
            timer.timeout.connect(changet)
            timer.start(1)
            sys.exit(app.exec_())

        if flag == 0:
            thread = threading.Thread(target=something)
            thread.start()
            flag = 1
        

        while traci.simulation.getMinExpectedNumber() > 0:
            if durs <= 0:
                if phase % 2 == 0:
                    traci.trafficlight.setPhase("0", phase)
                    traci.trafficlight.setPhaseDuration("0", 3)
                    durs = 3
                else:
                    current_state, X = vehnums(phase, X)
                    action = np.argmax(agent.get_qs(current_state.reshape(1, 144)))
                    #print(f"sugg-time {action + 2}")
                    traci.trafficlight.setPhase("0", phase)
                    durs = action + 2
                    traci.trafficlight.setPhaseDuration("0", action + 2)

                phase = (phase + 1) % 12

            traci.simulationStep()

            if durs > 0:
                durs -= 1
            step += 1
        st.append(step)
        print("\ncons - {}, steps={}, episode {}\n".format(cons, step, i))
    print(f" avg steps - {np.mean(st)} avg cons - {np.mean(co)}")

def test_integrate():
    agent = NNagent.Agent()
    flag1 = 0
    agent.network.load_weights(f"dddqn_traffic_20K+1000.model")
    #agent.network.load_weights(f"models/dddqn_traffic_15000.model")
    generate_test_integrate(0)

    traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    step = 0
    phase = 0
    durs = 0

    current_state = np.zeros((24,))
    X = np.zeros((6,24))
    
    def something():
        app = QtWidgets.QApplication(sys.argv)
        main = QtWidgets.QFrame()
        main.setGeometry(1200, 400, 450, 120)
        main.setStyleSheet("QLabel{font-size:20px;}")
        l1 = QtWidgets.QLabel("0.00000", main)
        l1.setGeometry(0, 0, 400, 40)
        l2 = QtWidgets.QLabel("South", main)
        l2.setGeometry(0, 40, 300, 40)
        l3 = QtWidgets.QLabel("Yellow", main)
        l3.setGeometry(0, 80, 450, 40)

        def getphasename(phase):
            color = "yellow"
            lanes = {0: "South", 1: "South-North", 2: "North", 3: "West",4: "West-East", 5: "East"}
            if phase % 2 == 0:
                color = "green"

            return color, lanes[(phase - 1) % 12 // 2]

        def changet():
            l1.setText(f"Time remaining for change: {durs}")

            color, lane = getphasename(phase)
            l2.setText(f"LANE: {lane} activated")
            l3.setText(f"LANE {lane} is running on {color.upper()} light")
        main.show()
        timer = QtCore.QTimer()
        timer.timeout.connect(changet)
        timer.start(1)
        sys.exit(app.exec_())

    if flag1 == 0:
            thread = threading.Thread(target=something)
            thread.start()
            flag1 = 1
        
    flag = 1
    np.random.seed(2)
    while traci.simulation.getMinExpectedNumber() > 0:      
        t1 = time.time()  
        if durs == 2 and phase % 2 == 1 and flag < 20:
            current_state1, X = vehnums(phase,X)            
            current_state, counts = get_vehicle_numbers(phase)
            current_state += current_state1            
            for i in range(24):
                type = 0
                if i % 6 == 0:
                    for j in range(counts[i]):
                        type = np.random.choice(["car", "car1", "car2", "car3", "sedan", "sedan1", "sedan2", "sedan3"])
                        traci.vehicle.add(f"{step}{i}{j}0545", f"r{i//6 + 1}o", typeID=type, depart=step+1, departLane="random", departPos=488 - traci.vehicletype.getLength(type) * j)
                elif i % 6 == 1:
                    for j in range(counts[i]):
                        type = np.random.choice(["moped", "moped1", "moped2", "moped3", "motorcycle", "motorcycle1", "motorcycle2", "motorcycle3"])
                        traci.vehicle.add(f"{step}{i}{j}324 ", f"r{i//6 + 1}o", typeID=type, depart=step+1, departLane="random", departPos=488 - traci.vehicletype.getLength(type) * j)
                elif i % 6 == 2:
                    for j in range(counts[i]):
                        type = np.random.choice(["bus", "bus1", "bus2", "bus3"])
                        traci.vehicle.add(f"{step}{i}{j}5756", f"r{i//6 + 1}o", typeID=type, depart=step+1, departLane="random", departPos=488 - traci.vehicletype.getLength(type) * j)
                elif i % 6 == 3:
                    for j in range(counts[i]):
                        type = np.random.choice(["truck", "truck1", "truck2", "truck3"])
                        traci.vehicle.add(f"{step}{i}{j}23442", f"r{i//6 + 1}o", typeID=type, depart=step+1, departLane="random", departPos=488 - traci.vehicletype.getLength(type) * j)
            flag += 1

        elif durs == 2 and phase % 2 == 1 and flag == 20:
            current_state, X = vehnums(phase,X)

        if durs <= 0:
            if phase % 2 == 0:
                traci.trafficlight.setPhase("0", phase)
                traci.trafficlight.setPhaseDuration("0", 3)
                durs = 3
            else:
                #current_state, counts = get_vehicle_numbers(phase)
                #print(np.sum(current_state))
                #print(np.sum(current_state1))
                action = np.argmax(agent.get_qs(current_state.reshape(1, 144)))
                if phase % 12 == 1:
                    print("SN activated", end = "")                
                elif phase % 12 == 3:
                    print("NS and SN activated", end = "")                
                elif phase % 12 == 5:
                    print("NS activated", end = "")                
                elif phase % 12 == 7:
                    print("EW activated", end = "")                
                elif phase % 12 == 9:
                    print("EW and WE activated", end = "")                
                elif phase % 12 == 11:
                    print("WE activated", end = "")                
                
                print(f" : Suggested-time = {action + 2}")
                traci.trafficlight.setPhase("0", phase)
                t2 = time.time()
                #print("Hello",t2-t1)
                durs = action + 2 #40
                traci.trafficlight.setPhaseDuration("0", durs)

            phase = (phase + 1) % 12

        traci.simulationStep()

        if durs > 0:
            durs -= 1
        step += 1
    print("\nsteps=",step)

def mov_img():
    wd = os.getcwd() + "/used"

    files = []
    for (dirpath, dirnames, filenames) in walk(wd):
        files.extend(filenames)
        break

    for f in files:
        shutil.move(wd+'/'+f, os.getcwd()+'/imgs')


if __name__ == "__main__":
    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

    #trainagent()
    #trainmid()
    #test()
    test_integrate()
    mov_img()
    print(end="\n")
    traci.close()
    sys.stdout.flush()
