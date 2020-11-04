from __future__ import division
from operator import itemgetter
import math
import optparse
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
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
    def __init__(self, name, location, cap):
        self.name = name
        self.location = location
        self.cap = cap
        self.bikes = []
        self.waitingToDock = []
        self.unsatDemand = 0
        self.outgoing = []
        self.incoming = []
        self.neighbourhood = []
        self.otherStations = []
        self.predictedOcc = []  # predicted occupancy at every time step of simulation
        self.estFlow = [0, float, 0, float]
        self.relationDict = dict

    def addOutgoing(self, edge):
        self.outgoing.append(edge)

    def addIncoming(self, edge):
        self.incoming.append(edge)

    def queryBorrow(self, time, destStat, feedback):
        interChangeable = [self] + list(set(self.neighbourhood) - set([destStat]))
        freeBikes = [0] * len(interChangeable)
        for ind in range(len(interChangeable)):
            freeBikes[ind] = interChangeable[ind].predictedOcc[time]
        totFreeBikes = sum(freeBikes)
        if totFreeBikes == 0:  # if no free bikes anywhere at desired time of departure, just go to initial stat
            if feedback == 1:
                self.unsatDemand += 1
            else:
                self.unsatDemand += 1
                # departureTimes.append([time, self, destStat])
            return
        prob = np.array(freeBikes)/totFreeBikes
        choice = np.random.choice(interChangeable, p=prob)
        choice.predictedOcc = np.append(choice.predictedOcc[0:time], (np.array(choice.predictedOcc[time:]) - 1))
        departureTimes.append([time, choice, destStat])

    def queryDock(self, time, startingStat, feedback):
        interChangeable = [self] + list(set(self.neighbourhood) -
                                        set([startingStat]))
        freeDocks = [0] * len(interChangeable)
        no_waiting = [0] * len(interChangeable)
        unsatDem = [0] * len(interChangeable)
        timeArray = [0] * len(interChangeable)
        choice = None
        totalWait = [0] * (len(interChangeable))

        for ind in range(len(interChangeable)):
            travelTime = startingStat.relationDict[interChangeable[ind]].timeTaken
            if interChangeable[ind] is not self:
                walkTime = self.relationDict[interChangeable[ind]].distance/1.5
            else:
                walkTime = 0
            timeInd = int(travelTime + time)
            freeDocks[ind] = interChangeable[ind].cap - interChangeable[ind].predictedOcc[timeInd]
            if freeDocks[ind] > 0:
                waitTime = 0
            else:
                arr = np.array(interChangeable[ind].predictedOcc[timeInd:])
                indexes = np.argwhere(arr < interChangeable[ind].cap)
                if indexes.size != 0:
                    waitTime = indexes[0][0]
                else:
                    waitTime = float("inf")
            totalWait[ind] = waitTime + walkTime + travelTime
            no_waiting[ind] = max(-freeDocks[ind], 0)
            freeDocks[ind] = max(freeDocks[ind], 0)
            unsatDem[ind] = interChangeable[ind].unsatDemand
            timeArray[ind] = timeInd

        totalWait = np.array(totalWait) - totalWait[0]
        if min(totalWait) < 0:
            totalWait = np.array(totalWait) - min(totalWait)
        totFreeDocks = sum(freeDocks)
        totQueue = sum(no_waiting)
        if len(interChangeable) == 1:  # if no neighbouring stations for destination
            choice = self  # go to initially requested dest
        # elif sum(unsatDem) != 0:
        #     dockProb = list(np.array(unsatDem)/sum(unsatDem))
        elif totFreeDocks == 0 and totQueue == 0:  # if full everywhere but no queue anywhere
            maxi = -1
            for ind in range(len(interChangeable)):
                if maxi < interChangeable[ind].unsatDemand:
                    choice = interChangeable[ind]
        elif totFreeDocks == 0:  # full everywhere but bikes waiting exist:
            noQueue = len(no_waiting) - np.count_nonzero(np.array(no_waiting))
            if noQueue > 0:
                dockProb = np.where(np.array(no_waiting) == 0, 1, 0)/noQueue
            else:
                no_waiting = 1 / np.array(no_waiting)
                dockProb = np.array(no_waiting) / sum(no_waiting)
        else:
            temp = np.add(np.array(freeDocks), np.array(unsatDem))
            dockProb = temp / sum(temp)
        if choice is None:
            if feedback == 1:
                choice = np.random.choice(interChangeable, p=dockProb)
            else:
                # extraTime = len(totalWait) - np.count_nonzero(np.array(totalWait))
                # if extraTime > 0:
                #     waitProb = np.where(np.array(totalWait) == 0, 1, 0)/extraTime
                # else:
                #     totalWait = 1 / np.array(totalWait)
                #     waitProb = np.array(totalWait)/sum(totalWait)
                # if sum(totalWait) == 0:
                #     waitProb = [1/len(interChangeable)] * len(interChangeable)

                print(totalWait)
                infff = np.where(np.array(totalWait) == float("inf"), 0, 1)
                if sum(infff) != 0:
                    waitProb = infff/sum(infff)
                else:
                    waitProb = np.array(totalWait)/sum(totalWait)
                print(waitProb)

                w = random.uniform(0, 1)
                p = np.array(waitProb) * w + np.array(dockProb) * (1-w)
                choice = np.random.choice(interChangeable, p=p)

            # choice = interChangeable[totalWait.index(min(totalWait))]
        index = interChangeable.index(choice)
        timeTaken = timeArray[index]
        choice.predictedOcc = np.append(choice.predictedOcc[0:timeTaken],
                                           (np.array(choice.predictedOcc[timeTaken:]) + 1))
        return choice

    def dockBike(self, bike, time):
        if len(self.bikes) < self.cap:
            self.bikes.append(bike)
            self.estFlow[2] += 1
            if time % 600 == 0:
                self.estFlow[3] = 600/self.estFlow[2]
        else:
            self.waitingToDock.append(bike)

    def borrowBike(self, time, dest, feedback):
        # if bike available to be borrowed:
        if len(self.bikes) > 0:
            if feedback == 0:  # if no feedback
                choice = dest
            else:   # use occupancy of bikes @ departure time to assign dest. station
                choice = dest.queryDock(int(time), self, feedback)
            # checking out a bike
            route = self.relationDict[choice].routeName
            traci.vehicle.add(self.bikes[0].id, route, typeID="reroutingType")
            self.bikes[0].lastTrip = [time, self, None, choice]
            self.bikes.remove(self.bikes[0])
        else:
            if feedback == 3:
                self.unsatDemand += 1
        if self.waitingToDock:
            self.bikes.append(self.waitingToDock[0])
            self.waitingToDock.pop(0)

    def checkForConnectivity(self, targetStat, v):
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
            if not station.checkForConnectivity(self, 1) or not self.checkForConnectivity(station, 1):
                if len(stations) == 1:
                    stations.remove(station)
                    return -1
                return 0
        return 1

    def update_stationRelation(self, stations, new):
        for station in stations:
            if self is not station:
                self.update_relation(station, new)
        keys = []
        if new:
            if not self.neighbourhood:
                min_dis = float("inf")
                for otherStation in self.otherStations:
                    if min_dis > otherStation.distance:
                        closest = otherStation.arrive
                self.neighbourhood.append(closest)
            for relation in self.otherStations:
                keys.append(relation.arrive)
            self.relationDict = dict(zip(keys, self.otherStations))  # station object to relation object

    def update_relation(self, targetStat, new):
        minCost = float("inf")
        for i in self.outgoing:
            for j in targetStat.incoming:
                path = traci.sumolib.net.Net.getShortestPath(net, i, j, vClass='bicycle')
                if path[0] and path[1] < minCost:
                    minCost = path[1]
                    route = [i.getID(), j.getID()]
        traci.route.add(self.name + "To" + targetStat.name, route)
        if new:  # generate same trips for all tests
            # # parameters for arrival rate distribution (piecewise linear function)
            min_cyc_dis = 400  # minimum distance that people are going to cycle (instead of walk)
            max_walk_dis = 600  # maximum distance that someone is willing to walk for different dock
            max_rate = 1 / 300  # maximum flow rate
            dis_for_maxrate = 1500  # distance between station where max expected flow rate occurs
            max_dis = 5000  # maximum bound where expected flow rate reaches 0
            rate = 0
            if minCost < max_walk_dis:  # if distance between stations walkable
                self.neighbourhood.append(targetStat)
            if minCost > min_cyc_dis:  # if distance between stations greater than minimum cycling distance:
                if minCost < dis_for_maxrate:
                    rate = (max_rate / (dis_for_maxrate - min_cyc_dis)) * (minCost - min_cyc_dis)
                else:
                    rate = (max_rate / (max_dis - dis_for_maxrate)) * (max_dis - minCost)
                rate = np.random.normal(rate, (rate / 10) ** 0.5, 1)[0]  # output is an array of 1 element
                if rate > 0:  # if demand exists: ie, flow rate > 0
                    # Time of next demand for bike according to poisson dist:
                    nextDemand = 5 * 60
                    while nextDemand < endSim:
                        nextDemand += round(-math.log(1.0 - random.uniform(0, 1)) / rate)
                        totalDemand.append([nextDemand, self, targetStat])
                else:
                    rate = 0
            self.otherStations.append(StationRelation(targetStat, self.name + "To" + targetStat.name, minCost, rate))
        else:
            self.relationDict[targetStat].timeTaken = self.relationDict[targetStat].distance/5.5


class StationRelation(object):
    def __init__(self, arrival, name, distance, rate):
        self.arrive = arrival  # arriving station
        self.routeName = name  # name of routes for traci call
        self.distance = distance
        self.rate = rate  # poisson distribution (lambda rate)
        self.timeTaken = distance/5.5

    def updateLastTime(self, arrivingBike):
        self.timeTaken = arrivingBike.lastTrip[2] - arrivingBike.lastTrip[0]


class Bike(object):
    def __init__(self, bike_no, station):
        self.id = str(bike_no)
        self.lastTrip = [int, object, int, object]


def run2():
    # simulation lasts for 3 hours
    step = 0
    delay = int(5 * 60/timeStep)
    station_occ = [[] for x in range(len(stations))]
    unsat_demand = [[] for x in range(len(stations))]

    while step < endSim:
        traci.simulationStep()

        if step % 600 == 0:
            for station in stations:
                print(step/60, station.name, station.predictedOcc[step:endSim])

        for ind in range(len(stations)):
            station_occ[ind].append(len(stations[ind].bikes) + len(stations[ind].waitingToDock))
            unsat_demand[ind].append(stations[ind].unsatDemand)

        while step + delay >= demand[0][0]:
            depTime, depStat, arrStat = demand[0]
            depStat.queryBorrow(step+delay, arrStat, 1)
            demand.pop(0)

        while departureTimes and step >= departureTimes[0][0]:
            depTime, depStat, arrStat = departureTimes[0]
            depStat.borrowBike(step, arrStat, 1)
            departureTimes.pop(0)

        # if bikes have arrived to destination:
        for arrivedBike in traci.simulation.getArrivedIDList():
            bike = allBikes[int(arrivedBike)]
            bike.lastTrip[2] = step
            depTime, depStat, arrTime, arrStat = bike.lastTrip
            depStat.relationDict[arrStat].updateLastTime(bike)
            arrStat.dockBike(bike, step)

        step += 1

    traci.close()
    sys.stdout.flush()
    return station_occ, unsat_demand

# run with feedback algorithm - no delay
def run3():
    # simulation lasts for 3 hours
    step = 0
    delay = int(5 * 60/timeStep)
    station_occ = [[] for x in range(len(stations))]
    unsat_demand = [[] for x in range(len(stations))]
    while step < endSim:
        traci.simulationStep()

        for ind in range(len(stations)):
            station_occ[ind].append(len(stations[ind].bikes) + len(stations[ind].waitingToDock))
            unsat_demand[ind].append(stations[ind].unsatDemand)

        while step + delay >= demand[0][0]:
            demand[0][1].queryBorrow(step+delay, demand[0][2], 2)
            demand.pop(0)

        while departureTimes and step >= departureTimes[0][0]:
            departureTimes[0][1].borrowBike(step, departureTimes[0][2], 2)
            departureTimes.pop(0)

        # if bikes have arrived to destination:
        for arrivedBike in traci.simulation.getArrivedIDList():
            bike = allBikes[int(arrivedBike)]
            bike.lastTrip[2] = step
            depTime, depStat, arrTime, arrStat = bike.lastTrip
            depStat.relationDict[arrStat].updateLastTime(bike)
            arrStat.dockBike(bike, step)

        step += 1

    traci.close()
    sys.stdout.flush()
    return station_occ, unsat_demand


def getEuclideanDis(coords1, coords2):
    x_diff = coords1[0] - coords2[0]
    y_diff = coords1[1] - coords2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5


def resetSim():
    departureTimes.clear()
    allBikes.clear()
    bikeID = 0
    ini_full = 0.95
    for station in stations:
        station.bikes = []
        station.waitingToDock = []
        station.unsatDemand = 0
        station.update_stationRelation(stations, False)
        for bike in range(int(station.cap * ini_full)):
            new_bike = Bike(bikeID, station)
            station.bikes.append(new_bike)
            allBikes.append(new_bike)
            bikeID += 1
            station.predictedOcc = [len(station.bikes)] * (endSim + 60 * 1000)


# main entry point
if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start(['sumo', "-c", "melbourne.sumocfg"], label="sim1")

    # get net information
    net = traci.sumolib.net.readNet("melbourne.net.xml")

    no_stations = 12
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
            new_st = Station(str(no_stations), [x, y, "xy"], 44)
            new_st.outgoing = node.getOutgoing()
            new_st.incoming = node.getIncoming()
            result = new_st.update_stations(stations)
            if result == 1:
                stations.append(new_st)
                station_keys.append(new_st.name)
                no_stations -= 1
                possible_nodes = []
                for node in net._nodes:
                    if 500 < getEuclideanDis(node.getCoord(), new_st.location[0:-1]) < 800:
                        if len(stations) > 1:
                            for station in stations[0:-1]:
                                if getEuclideanDis(node.getCoord(), station.location[0:-1]) > 200:
                                    possible_nodes.append(node)
                        else:
                            possible_nodes.append(node)

            elif result == -1:
                no_stations += 1
                possible_nodes = net._nodes
            else:
                possible_nodes.remove(node)

    stat_dict = dict(zip(station_keys, stations))

    for station in stations:
        station.update_stationRelation(stations, True)
        for bike in range(int(station.cap * ini_full)):
            new_bike = Bike(bikeID, station)
            station.bikes.append(new_bike)
            allBikes.append(new_bike)
            bikeID += 1
            station.predictedOcc = [len(station.bikes)] * (endSim + 60 * 300)

    for station in stations:
        print("starting Stat:", station.name)
        neigh = []
        for neighbour in station.neighbourhood:
            neigh.append(neighbour.name)
        print("neighbours:", neigh)
        dis = []
        for station in station.otherStations:
            dis.append([station.arrive.name, station.distance, station.rate])
        print("relations:", dis)
    totalDemand = sorted(totalDemand, key=itemgetter(0))

    demand = totalDemand[:]
    station_occ2, unsat_demand2 = run2()

    traci.start(['sumo', "-c", "melbourneFeedback.sumocfg"], label="sim2")
    traci.switch("sim2")
    resetSim()
    demand = totalDemand[:]
    station_occ3, unsat_demand3 = run3()

    time = np.arange(0, 10800)
    # f = plt.figure(1)
    xlim = np.arange(0, 60 * 60 * 3.5, 60 * 60)
    # for i in range(len(stations)):
    #     plt.plot(time, station_occ[i])
    # ax = plt.gca()
    # ax.set_xbound(lower=0)
    # ax.set_ybound(lower=0)
    # plt.xlabel('time (hours)')
    # plt.ylabel('occupancy')
    # plt.legend([station.name for station in stations])
    # plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
    #
    # g = plt.figure(2)
    # for i in range(len(stations)):
    #     plt.plot(time, unsat_demand[i])
    # ax = plt.gca()
    # ax.set_xbound(lower=0)
    # ax.set_ybound(lower=0)
    # plt.xlabel('time (hours)')
    # plt.ylabel('accumulated # unsatisfied demand')
    # plt.legend([station.name for station in stations])
    # plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
    #
    i = plt.figure(3)
    for i in range(len(stations)):
        plt.plot(time, station_occ2[i])
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('time (hours)')
    plt.ylabel('occupancy')
    plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])

    j = plt.figure(4)
    for i in range(len(stations)):
        plt.plot(time, unsat_demand2[i])
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('time (hours)')
    plt.ylabel('accumulated # unsatisfied demand')
    plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])

    k = plt.figure(5)
    for i in range(len(stations)):
        plt.plot(time, station_occ3[i])
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('time (hours)')
    plt.ylabel('occupancy')
    plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])

    l = plt.figure(6)
    for i in range(len(stations)):
        plt.plot(time, unsat_demand3[i])
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('time (hours)')
    plt.ylabel('accumulated # unsatisfied demand')
    plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])

    plt.show()