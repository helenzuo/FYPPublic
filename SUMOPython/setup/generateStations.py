from __future__ import division

import optparse
import os
import random
import sys
import json

import pandas
import traci
from sumolib import checkBinary

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


class Station(object):
    def __init__(self, name, location, longlat, cap, node):
        self.name = name
        self.location = location
        self.longLat = longlat
        self.cap = cap
        self.node = node
        self.outgoing = []
        self.incoming = []
        self.routes = []
        self.otherStations = []
        self.relationDict = dict

    def checkForConnectivity(self, targetStat):
        for i in self.outgoing:
            for j in targetStat.incoming:
                path = traci.sumolib.net.Net.getShortestPath(net, i, j, vClass='bicycle')
                if path[0]:
                    return True
            else:
                continue
            break
        return False

    def update_stations(self, stations):
        for station in stations:
            if not station.checkForConnectivity(self) or not self.checkForConnectivity(station):
                if len(stations) == 1:
                    stations.remove(station)
                    return -1
                return 0
        return 1

    def update_stationRelation(self, stations):
        print(self.name)
        for station in stations:
            if self is not station:
                print(station.name)
                self.update_relation(station)

    def update_relation(self, targetStat):
        route = None
        minCost = float("inf")
        for i in self.outgoing:
            for j in targetStat.incoming:
                path = traci.sumolib.net.Net.getShortestPath(net, i, j, vClass='bicycle')
                if path[0] and path[1] < minCost:
                    minCost = path[1]
                    route = [i.getID(), j.getID()]
        if route is not None:
            self.routes.append([self.name, targetStat.name, i.getID(), j.getID(), minCost])


def getEuclideanDis(coords1, coords2):
    x_diff = coords1[0] - coords2[0]
    y_diff = coords1[1] - coords2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5

# main entry point
if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start(['sumo', "-c", "melbourne.sumocfg"], label="sim0")

    # get net information
    net = traci.sumolib.net.readNet("../net/melbourne_larger.net.xml")

    no_stations = 50
    stations = []
    totalDemand = []
    departureTimes = []

    timeStep = 1
    endSim = int(10800 / timeStep)
    allBikes = []
    ini_full = .95  # percentage occupancy of stations at the start of simulation:
    bikeID = 0
    station_keys = []
    possible_nodes = net._nodes
    while no_stations > 0:
        node = random.choice(possible_nodes)
        if node.getOutgoing() and node.getIncoming():
            x, y = node.getCoord()
            longitude, latitude = traci.sumolib.net.Net.convertXY2LonLat(net, x, y)
            new_st = Station("station " + str(no_stations), [x, y, "xy"], [longitude, latitude], round(random.uniform(10, 30)), traci.sumolib.net.node.Node.getID(node))
            new_st.outgoing = node.getOutgoing()
            new_st.incoming = node.getIncoming()
            result = new_st.update_stations(stations)
            if result == 1:
                stations.append(new_st)
                station_keys.append(new_st.name)
                no_stations -= 1
                possible_nodes = []
                for node in net._nodes:
                    if 500 < getEuclideanDis(node.getCoord(), new_st.location[0:-1]) < 700:
                        if len(stations) > 1:
                            for station in stations[0:-1]:
                                if getEuclideanDis(node.getCoord(), station.location[0:-1]) > 100:
                                    possible_nodes.append(node)
                        else:
                            possible_nodes.append(node)

            elif result == -1:
                no_stations += 1
                possible_nodes = net._nodes
            else:
                possible_nodes.remove(node)

    for station in stations:
        station.update_stationRelation(stations)

    my_df = []
    for station in stations:
        d = {
            'name':station.name,
            'longLat': station.longLat,  # some formula for obtaining values
            'cap': int(station.cap),
            'routes': station.routes,
            'node': station.node
        }
        my_df.append(d)

    with open('../stations.txt', 'w') as f:
        json.dump(my_df, f, ensure_ascii=False)



