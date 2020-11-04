from __future__ import division

import math
import optparse
import os
import random
import sys
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import traci
from matplotlib.ticker import PercentFormatter
from sumolib import checkBinary

# check if SUMO_HOME is a path variable in operating system
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Returns the SUMO options (ie: whether to run SUMO UI or just the commandline version)
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

# Station object to be declared for each station in bike share system
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
        self.relationDict = dict

    def addOutgoing(self, edge):
        self.outgoing.append(edge)

    def addIncoming(self, edge):
        self.incoming.append(edge)

    # method call to book ahead and borrow a bike at some future time (depTime)
    def queryBorrow(self, depTime, destStat, feedback):
        # maxWalk = max(random.gauss(600, 100),0)
        maxWalk = 600
        # create a list of possible departure stations to select from based on the maximum walking distance allowed
        interChangeable = [self]
        for station in self.otherStations:
            if station.distance <= maxWalk:
                if station.arrive is not destStat:
                    interChangeable.append(station.arrive)
        # Get number of bikes/docks available at each station, once list of possible departure stations finalised:
        no_interChangeable = len(interChangeable)
        freeBikes = [0] * no_interChangeable
        extraBikes = [0] * no_interChangeable
        ratio = [0] * no_interChangeable

        if feedback == 3:
            # store the fill levels of possible stations into the ratio list
            for ind in range(no_interChangeable):
                ratio[ind] = interChangeable[ind].predictedOcc[depTime] / interChangeable[ind].cap
            # multiplier refers to the average fill level at the possible stations. It will be used to set the "target" fill
            # level of the stations and then to determine the number of "extra bikes" at each station
            multi = np.mean(ratio)

        for ind in range(no_interChangeable):
            if feedback == 1:
                # get the number of bikes docked at each station
                freeBikes[ind] = len(interChangeable[ind].bikes) + len(interChangeable[ind].waitingToDock)
                # multiply station capacity against the target fill level to find "extra bikes"
                extraBikes[ind] = max(freeBikes[ind] - interChangeable[ind].cap, 0)
            elif feedback == 2 or feedback == 5:
                freeBikes[ind] = max(interChangeable[ind].predictedOcc[depTime], 0)
                extraBikes[ind] = max(freeBikes[ind] - interChangeable[ind].cap, 0)
            elif feedback == 3:
                freeBikes[ind] = max(interChangeable[ind].predictedOcc[depTime], 0)
                extraBikes[ind] = max(freeBikes[ind] - interChangeable[ind].cap * multi, 0)

        totFreeBikes = sum(freeBikes) # total number of free bikes across all possible station
        # if no free bikes anywhere at desired time of departure, tell the user this and increment the unsatisfied
        # demand counter by one
        if totFreeBikes == 0:
            self.unsatDemand += 1
            return
        # if there are "extra bikes" available at any station, ie: at least one station has bikes above their optimum
        # fill level, use this value to stochastically assign the user a departing station. Otherwise, just use the
        # number of docked bikes at each station to assign the departing station
        if sum(extraBikes) > 0:
            prob = np.array(extraBikes) / sum(extraBikes)
        else:
            prob = np.array(freeBikes) / totFreeBikes
        choice = np.random.choice(interChangeable, p=prob)

        if feedback == 5:
            choice = interChangeable[np.argmax(freeBikes)]

        # if random.uniform(0, 1) < (pr/2)/(1-pr/2):
        #     choice = self

        if feedback != 0:
            choice.predictedOcc = np.append(choice.predictedOcc[0:depTime], (np.array(choice.predictedOcc[depTime:]) - 1))  # update predicted occ
            departureTimes.append([depTime, choice, destStat, maxWalk])

    # method call to check for docks available at some future time
    def queryDock(self, time, startingStat, feedback, maxWalk):
        # create a list of possible departure stations to select from based on the maximum walking distance allowed
        interChangeable = [self]
        for station in self.otherStations:
            if station.distance <= maxWalk:
                if station.arrive is not startingStat:
                    interChangeable.append(station.arrive)
        # Get number of bikes/docks at each station, once list of possible arrival stations finalised:
        no_interChangeable = len(interChangeable)
        freeDocks = [0] * no_interChangeable # number of docks available
        ratio = [0] * no_interChangeable # fill level
        no_waiting = [0] * no_interChangeable # number of users waiting for dock
        timeArray = [0] * no_interChangeable
        choice = None

        if feedback == 3:
            for ind in range(no_interChangeable):
                travelTime = startingStat.relationDict[interChangeable[ind]].timeTaken
                timeInd = int(travelTime + time)
                ratio[ind] = interChangeable[ind].predictedOcc[timeInd] / interChangeable[ind].cap
            multi = np.mean(ratio)

        for ind in range(no_interChangeable):
            # estimate the time taken to travel from departing station to each possible arrival station
            travelTime = startingStat.relationDict[interChangeable[ind]].timeTaken
            timeInd = int(travelTime + time)
            if feedback == 1:
                freeDocks[ind] = interChangeable[ind].cap - (len(interChangeable[ind].bikes) + len(interChangeable[ind].waitingToDock))
            elif feedback == 2 or feedback == 5:
                freeDocks[ind] = interChangeable[ind].cap - interChangeable[ind].predictedOcc[timeInd]
            elif feedback == 3:
                # use the average fill level of all interchangeable stations as the "target" fill level
                freeDocks[ind] = interChangeable[ind].cap * multi - interChangeable[ind].predictedOcc[timeInd]

            no_waiting[ind] = max(-freeDocks[ind], 0)
            freeDocks[ind] = max(freeDocks[ind], 0)
            timeArray[ind] = timeInd

        totFreeDocks = sum(freeDocks)  # total number of docks avail
        totQueue = sum(no_waiting)  # total number of people waiting to dock
        if no_interChangeable == 1 or (totFreeDocks == 0 and totQueue == 0):  # if no neighbouring stations for destination or full everywhere but no queue anywhere
            choice = self  # go to initially requested station
        elif totFreeDocks == 0:
            # get the number of possible stations that have do not have a queue
            noQueue = no_interChangeable - np.count_nonzero(np.array(no_waiting))
            if noQueue > 0:             # if there exists stations that do not have a wait yet
                # assign probabilities for arrival station equally across those without a wait, since all full anyways
                dockProb = np.where(np.array(no_waiting) == 0, 1, 0) / noQueue
            # if all stations have a wait, assign higher chance of going to station with smaller wait
            else:
                no_waiting = 1 / np.array(no_waiting)
                dockProb = np.array(no_waiting) / sum(no_waiting)
        # docks available:
        else:
            dockProb = np.array(freeDocks) / totFreeDocks
        if choice is None:
            choice = np.random.choice(interChangeable, p=dockProb)
            if feedback == 5:
                choice = interChangeable[np.argmax(freeDocks)]

        # if random.uniform(0, 1) < (pr/2)/(1-pr/2):
        #     choice = self

        if feedback != 0:
            index = interChangeable.index(choice)
            timeTaken = timeArray[index]
            choice.predictedOcc = np.append(choice.predictedOcc[0:timeTaken],
                                        np.array(choice.predictedOcc[timeTaken:]) + 1)

        return choice, timeTaken

    # method is called when the user arrives to dock their bike at this station
    def dockBike(self, bike, time, feedback):
        # extract the estimated arrival time of arriving bicycle
        estArr = bike.lastTrip[4]
        # if bike has arrived earlier than expected, adjust the predicted occupancy information to reflect this
        if feedback != 0 and estArr is not int and estArr > time:
            self.predictedOcc = np.append(self.predictedOcc[0:time],
                                          np.append(np.array(self.predictedOcc[time:estArr]) + 1,
                                                    self.predictedOcc[estArr:]))
        # if there are free docks available, dock the bike. Otherwise, add bike into list of bikes waiting
        self.bikes.append(bike)
        if len(self.bikes) >= self.cap:
            self.waitingToDock.append(bike)

    # method is called when the user arrives to borrow a bike at this station
    def borrowBike(self, time, dest, feedback, maxWalk):
        # if bike available to be borrowed:
        if len(self.bikes) > 0:
            if feedback == 0:
                choice = dest
                self.bikes[0].lastTrip = [time, self, int, dest, int, True]
            else:
                if maxWalk != -1:  # maxWalk == -1 means the rider isn't using the app
                    # at checkout time, confirm the arrival station
                    choice, estArrTime = dest.queryDock(int(time), self, feedback, maxWalk)
                else:
                    choice = dest
            # checking out a bike (creating trip in simulation)
            bike = self.bikes[0]
            self.bikes.remove(bike)
            route = self.relationDict[choice].routeName
            traci.vehicle.add(bike.id, route, typeID="reroutingType")
            if feedback != 0:
                if maxWalk != -1:
                    bike.lastTrip = [time, self, int, choice, estArrTime, True]
                else:
                    bike.lastTrip = [time, self, int, dest, int, True]

        else:
            self.unsatDemand += 1
        # if after borrowing, a dock frees up and there are people waiting to dock, allow the first in line to dock.
        if self.waitingToDock:
            self.waitingToDock.pop(0)

    # ==================================================================================================================
    #                      FOLLOWING METHODS ARE CALLED ONLY ONCE ON SETUP OF SIMULATION
    # ==================================================================================================================
    # this method checks if the new station "connects" to the others -> ie: can't be on a dead end/borders of the map
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

    # called at initialisation to see if new station connects
    def update_stations(self, stations):
        for station in stations:
            if not station.checkForConnectivity(self, 1) or not self.checkForConnectivity(station, 1):
                if len(stations) == 1:
                    stations.remove(station)
                    return -1
                return 0
        return 1

    # this method sets up relationship between this station and the other stations on the map
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

    # this method will initialise station-station parameters like flow rate, distance, routes etc..
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
            # max_walk_dis = 600  # maximum distance that someone is willing to walk for different dock
            max_rate = 1 / 480  # maximum flow rate
            dis_for_maxrate = 1500  # distance between station where max expected flow rate occurs
            max_dis = 5000  # maximum bound where expected flow rate reaches 0
            rate = 0
            # if minCost < max_walk_dis:  # if distance between stations walkable
            #     self.neighbourhood.append(targetStat)
            if minCost > min_cyc_dis:  # if distance between stations greater than minimum cycling distance:
                if minCost < dis_for_maxrate:
                    rate = (max_rate / (dis_for_maxrate - min_cyc_dis)) * (minCost - min_cyc_dis)
                else:
                    rate = (max_rate / (max_dis - dis_for_maxrate)) * (max_dis - minCost)
                rate = np.random.normal(rate, (rate / 5) ** 0.5, 1)[0]  # output is an array of 1 element
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
            self.relationDict[targetStat].timeTaken = self.relationDict[targetStat].distance / 5.5

# Station relation class
class StationRelation(object):
    def __init__(self, arrival, name, distance, rate):
        self.arrive = arrival  # arriving station
        self.routeName = name  # name of routes for traci call
        self.distance = distance
        self.rate = rate  # poisson distribution (lambda rate)
        self.timeTaken = distance / 5.5

    def updateLastTime(self, arrivingBike):
        self.timeTaken = arrivingBike.lastTrip[2] - arrivingBike.lastTrip[0]

# bike class
class Bike(object):
    def __init__(self, bike_no):
        self.id = str(bike_no)
        # [dep Time, dep station, arr time, arr Stat, est arr Time, satisfied?]
        self.lastTrip = [int, object, int, object, int, bool]

# This method runs the simulation. ie: it does all the work of borrowing/docking bikes etc dynamically
def run(feedback):
    # simulation lasts for 3 hours
    step = 0
    delay = int(15 * 60 / timeStep)  # time in advance that users book bike top depart
    station_occ = [[] for x in range(len(stations))]
    unsat_demand = [[] for x in range(len(stations))]
    while step < endSim: #continue until end of simulation

        traci.simulationStep() #increment step in sumo
        # log the station fill levels and unsatisfied demand every step
        for ind in range(len(stations)):
            # check if customers have been waiting to dock for more than 90 seconds at any station to increment unsatisfied customer count
            for bike in stations[ind].waitingToDock:
                if (step - bike.lastTrip[2]) > 90 and bike.lastTrip[5]:
                    bike.lastTrip[5] = False
                    stations[ind].unsatDemand += 1
            station_occ[ind].append((len(stations[ind].bikes) + len(stations[ind].waitingToDock))/stations[ind].cap * 100)
            unsat_demand[ind].append(stations[ind].unsatDemand)

        # user due to book a departure
        while step + delay >= demand[0][0]:
            depTime, depStat, arrStat = demand[0]
            if random.uniform(0,1) < pr:  # rider uses app
                depStat.queryBorrow(step + delay, arrStat, feedback)
            else:  # rider doesn't use app, will leave from desired station and system doesn't know this yet
                departureTimes.append([depTime, depStat, arrStat, -1])
            demand.pop(0)

        # user due to depart
        while departureTimes and step >= departureTimes[0][0]:
            depTime, depStat, arrStat, maxWalk = departureTimes[0]
            depStat.borrowBike(step, arrStat, feedback, maxWalk)  # borrow bike
            if maxWalk == -1:  # doesn't use the app, predicted occ of station will only be updated when the bike is borrowed
                depStat.predictedOcc = np.append(depStat.predictedOcc[0:step], (np.array(depStat.predictedOcc[step:]) - 1))
            departureTimes.pop(0)

        # if bikes have arrived to destination:
        for arrivedBike in traci.simulation.getArrivedIDList():
            bike = allBikes[int(arrivedBike)]
            bike.lastTrip[2] = int(step)
            depTime, depStat, arrTime, arrStat, estTime, satis = bike.lastTrip
            depStat.relationDict[arrStat].updateLastTime(bike)
            arrStat.dockBike(bike, step, feedback)
            if estTime == int:  # not an app user, can only update predicted occupancy now
                arrStat.predictedOcc = np.append(arrStat.predictedOcc[0:step],
                                                 (np.array(arrStat.predictedOcc[step:]) + 1))

        step += 1

    traci.close()
    sys.stdout.flush()
    return station_occ, unsat_demand

# running the simulation without a control method in place. ie: users go where ever they want
def run_control():
    # simulation lasts for 3 hours
    step = 0
    station_occ = [[] for x in range(len(stations))]
    unsat_demand = [[] for x in range(len(stations))]

    while step < endSim:  # run until end of sim
        traci.simulationStep()  # increment step in sumo

        # log the number of unsatisfied customers and fill levels of stations to plot later
        for ind in range(len(stations)):
            # check if customers have been waiting to dock for more than 90 seconds at any station to increment unsatisfied customer count
            for bike in stations[ind].waitingToDock:
                if (step - bike.lastTrip[2]) > 90 and bike.lastTrip[5]:
                    bike.lastTrip[5] = False
                    stations[ind].unsatDemand += 1
            station_occ[ind].append((len(stations[ind].bikes) + len(stations[ind].waitingToDock))/stations[ind].cap * 100)
            unsat_demand[ind].append(stations[ind].unsatDemand)

        # user wants to depart and so borrow a bike
        while step >= demand[0][0]:
            depTime, depStat, arrStat = demand[0]
            depStat.borrowBike(step, arrStat, 0, 0)
            demand.pop(0)

        # if bikes have arrived to destination:
        for arrivedBike in traci.simulation.getArrivedIDList():
            bike = allBikes[int(arrivedBike)]
            bike.lastTrip[2] = int(step)
            depTime, depStat, arrTime, arrStat, estTime, satis = bike.lastTrip
            arrStat.dockBike(bike, step, 0)

        step += 1 # Go to the next time step

    traci.close()
    sys.stdout.flush()
    return station_occ, unsat_demand

# get direct distance between coods
def getEuclideanDis(coords1, coords2):
    x_diff = coords1[0] - coords2[0]
    y_diff = coords1[1] - coords2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5

# method to reset the simulation between different comtrol methods
def resetSim(ini_full):
    departureTimes.clear()
    allBikes.clear()
    bikeID = 0
    ini_full = ini_full
    for station in stations:
        station.bikes = []
        station.waitingToDock = []
        station.unsatDemand = 0
        station.update_stationRelation(stations, False)
        for bike in range(int(station.cap * ini_full)):
            new_bike = Bike(bikeID)
            station.bikes.append(new_bike)
            allBikes.append(new_bike)
            bikeID += 1
            station.predictedOcc = [len(station.bikes)] * (endSim + 60 * 1000)


# main entry point
if __name__ == "__main__":

    pr = .5  # 1 - customer cooperation level

    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start(['sumo', "-c", "melbourne.sumocfg"], label="sim0")

    # get net information
    net = traci.sumolib.net.readNet("melbourne.net.xml")

    no_stations = 10
    stations = []
    totalDemand = []
    departureTimes = []

    timeStep = 1  #1 sec
    endSim = int(10800 / timeStep)
    allBikes = []
    ini_full = .8  # percentage occupancy of stations at the start of simulation:
    bikeID = 0
    station_keys = []
    possible_nodes = net._nodes

    # generating no_stations randomly
    while no_stations > 0:
        node = random.choice(possible_nodes)  # get a random node
        if node.getOutgoing() and node.getIncoming():  # if the node has both incoming and outgoing edges that can be used by bike
            x, y = node.getCoord()  # get the coordinate of the node
            new_st = Station("station " + str(no_stations), [x, y, "xy"], random.uniform(20,40))  # create a new station object (temp)
            new_st.outgoing = node.getOutgoing()
            new_st.incoming = node.getIncoming()
            result = new_st.update_stations(stations)  # this checks if the new station connects with all the other stations
            if result == 1:  # if it does connect
                stations.append(new_st)  #add it to the list of stations
                station_keys.append(new_st.name)  # and the station keys to create dictionary later
                no_stations -= 1  # decrement number of stations that need to be generated
                possible_nodes = []
                for node in net._nodes:  # get all the nodes that are within 500 to 700 metres of the new station (for the next station to be generated)
                    if 500 < getEuclideanDis(node.getCoord(), new_st.location[0:-1]) < 700:
                        if len(stations) > 1: # if there is more than 1 station already generated
                            for station in stations[0:-1]:  # the new station generated must also be at least 100 metres from each of them
                                if getEuclideanDis(node.getCoord(), station.location[0:-1]) > 100:
                                    possible_nodes.append(node)
                        else:
                            possible_nodes.append(node)
            elif result == -1:  # if second station doesn't connect first second one, reselect the first one too
                no_stations += 1
                possible_nodes = net._nodes
            else:  # if 3rd+ station doesn't connect with previous ones, then remove that node from the list of possible stations
                possible_nodes.remove(node)

    stat_dict = dict(zip(station_keys, stations))  # create dict for easier access

    for station in stations:
        station.update_stationRelation(stations, True)  # generate the "demand" for the simulation
        for bike in range(int(station.cap * ini_full)): # load all the bikes into docks of stations
            new_bike = Bike(bikeID)
            station.bikes.append(new_bike)
            allBikes.append(new_bike)
            bikeID += 1
            station.predictedOcc = [len(station.bikes)] * (endSim + 60 * 1000)

    totalDemand = sorted(totalDemand, key=itemgetter(0))
    demand = totalDemand[:]

    # control case
    station_occ, unsat_demand = run_control()
    station_occ = [station_occ]
    unsat_demand = [unsat_demand]

    ############# different control methods to be tested with the same set of demand: #####################
    traci.start(['sumo', "-c", "melbourneFeedback.sumocfg"], label="sim1")
    traci.switch("sim1")
    demand = totalDemand[:]
    resetSim(ini_full)
    station_occ_temp, unsat_demand_temp = run(1)
    station_occ.append(station_occ_temp)
    unsat_demand.append(unsat_demand_temp)

    traci.start(['sumo', "-c", "melbourneFeedback2.sumocfg"], label="sim2")
    traci.switch("sim2")
    resetSim(ini_full)
    demand = totalDemand[:]
    station_occ_temp, unsat_demand_temp = run(2)
    station_occ.append(station_occ_temp)
    unsat_demand.append(unsat_demand_temp)

    traci.start(['sumo', "-c", "melbourneFeedback3.sumocfg"], label="sim3")
    traci.switch("sim3")
    resetSim(ini_full)
    demand = totalDemand[:]
    station_occ_temp, unsat_demand_temp = run(3)
    station_occ.append(station_occ_temp)
    unsat_demand.append(unsat_demand_temp)

    traci.start(['sumo', "-c", "melbourneFeedback4.sumocfg"], label="sim4")
    traci.switch("sim4")
    resetSim()
    demand = totalDemand[:]
    station_occ_temp, unsat_demand_temp = run(5)
    station_occ.append(station_occ_temp)
    unsat_demand.append(unsat_demand_temp)

    # plotting the results
    time = np.arange(0, 10800)
    xlim = np.arange(0, 60 * 60 * 3.5, 60 * 60)
    for sim_no, sim in enumerate(station_occ):
        plt.figure(sim_no)
        for i in range(len(stations)):
            plt.plot(time, station_occ[sim_no][i])
        ax = plt.gca()
        ax.set_xbound(lower=0)
        ax.set_ybound(lower=0)
        plt.xlabel('Time (hours)')
        plt.ylabel('Fill Level')
        plt.legend([station.name for station in stations])
        plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
        ax.yaxis.set_major_formatter(PercentFormatter())
    qos = []
    for sim_no, sim in enumerate(unsat_demand):
        plt.figure(sim_no + len(unsat_demand))
        totalunsat = 0
        for i in range(len(stations)):
            plt.plot(time, unsat_demand[sim_no][i])
            totalunsat += unsat_demand[sim_no][i][-1]
        ax = plt.gca()
        ax.set_xbound(lower=0)
        ax.set_ybound(lower=0)
        plt.xlabel('Time (hours)')
        plt.ylabel('Accumulated number of out-of-stock events')
        plt.legend([station.name for station in stations])
        plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
        print("simulation #: ", sim_no, "QoS average: ", 1 - totalunsat/len(totalDemand))
        qos.append(1 - totalunsat/len(totalDemand))

    qos = np.array(qos)

    ### save results to excel file
    # with open('D:\SUMO\Melbourne 2/det.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(qos)
    #
    # quit()

    plt.show()

