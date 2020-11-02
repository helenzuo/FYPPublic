# =====================================================================================================================
# HELEN ZUO 27807746
# Monash University 2020
# ECSE Final Year Project
# =====================================================================================================================

# we need to import some python modules from the $SUMO_HOME/tools directory
from __future__ import division
import datetime
import json
import math
import multiprocessing as mp
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
import server

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
    def __init__(self, name, location, longlat, cap):
        self.name = name
        self.location = location
        self.longLat = longlat
        self.cap = cap
        self.bikes = []
        self.waitingToDock = []
        self.unsatDemand = 0
        self.otherStations = []
        self.predictedOcc = []  # predicted occupancy at every time step of simulation
        self.routes = []
        self.QR = int
        self.relationDict = dict

    # method call to book ahead and borrow a bike at some future time (depTime)
    def queryBorrow(self, depTime, destStat, isAutomated, customer):
        # create a list of possible departure stations to select from based on the maximum walking distance allowed
        interChangeable = [self]
        for station in self.otherStations:
            if station.distance <= customer.maxWalk:
                # no point having the user start at their "arrival station" for the automated cases.
                # For the client query, we do not set destStat (ie: destState = None) so we can return the bike where
                # we borrowed it
                if station.arrive is not destStat:
                    interChangeable.append(station.arrive)

        # Get number of bikes/docks available at each station, once list of possible departure stations finalised:
        numberInterChangeable = len(interChangeable)  # number of possible stations to depart from
        freeBikes = [0] * numberInterChangeable  # number of free bikes at possible station to be stored in list
        extraBikes = [0] * numberInterChangeable  # number of "extra bikes" to be stored
        ratio = [0] * numberInterChangeable  # fill percentage to be stored

        rowIndex = np.where(self.predictedOcc[:, 0] == depTime)[0][0]
        # store the fill levels of possible stations into the ratio list
        for ind in range(numberInterChangeable):
            ratio[ind] = interChangeable[ind].predictedOcc[rowIndex][1] / interChangeable[ind].cap
        # multiplier refers to the average fill level at the possible stations. It will be used to set the "target" fill
        # level of the stations and then to determine the number of "extra bikes" at each station
        multiplier = np.mean(ratio)

        for ind in range(numberInterChangeable):
            # get the number of bikes docked at each station
            freeBikes[ind] = max(interChangeable[ind].predictedOcc[rowIndex][1], 0)
            # multiply station capacity against the target fill level to find "extra bikes"
            extraBikes[ind] = max(freeBikes[ind] - interChangeable[ind].cap * multiplier, 0)

        totFreeBikes = sum(freeBikes)  # total number of free bikes across all possible station
        # if no free bikes anywhere at desired time of departure, tell the user this and increment the unsatisfied
        # demand counter by one
        if totFreeBikes == 0:
            self.unsatDemand += 1
            return None, interChangeable, None
        # if there are "extra bikes" available at any station, ie: at least one station has bikes above their optimum
        # fill level, use this value to stochastically assign the user a departing station. Otherwise, just use the
        # number of docked bikes at each station to assign the departing station
        if sum(extraBikes) > 0:
            prob = np.array(extraBikes) / sum(extraBikes)
        else:
            prob = np.array(freeBikes) / totFreeBikes

        choice = np.random.choice(interChangeable, p=prob)

        if isAutomated:  # isAutomated is false if this is called by the client through Android app
            choice.predictedOcc[rowIndex:-1, 1] -= 1
            departureTimes.append([depTime, choice, destStat, customer])
        else:
            return choice, interChangeable, freeBikes  # returns the assigned station, list of interchangeable stations and prediectedOcc

    # method call to check for docks available at some future time
    def queryDock(self, time, startingStat, customer, borrowLength=-1, isAutomated=True):
        # create a list of possible departure stations to select from based on the maximum walking distance allowed
        interChangeable = [self]
        for station in self.otherStations:
            if station.distance <= customer.maxWalk:
                if station.arrive is not startingStat:
                    interChangeable.append(station.arrive)
        # Get number of bikes/docks at each station, once list of possible arrival stations finalised:
        numberInterChangeable = len(interChangeable)
        freeDocks = [0] * numberInterChangeable  # number of docks available
        extraDocks = [0] * numberInterChangeable  # number of extra docks available
        ratio = [0] * numberInterChangeable  # fill level
        waiting = [0] * numberInterChangeable  # number of users waiting for dock
        rowIndex = [0] * numberInterChangeable
        choice = None

        travelTime = [0] * numberInterChangeable
        estimateArrivalTime = [0] * numberInterChangeable
        # estimate the time taken to travel from departing station to each possible arrival station
        for ind in range(numberInterChangeable):
            if borrowLength == -1:  # borrowLength = -1 means we are travelling from start station to destination directly
                # this is the last recorded time taken to travel between these two stations
                travelTime[ind] = startingStat.relationDict[interChangeable[ind]].timeTaken
            else:  # an amount of time has been specified by the client
                travelTime[ind] = borrowLength * 60
            # store the estimated arrival time into the time index array
            estimateArrivalTime[ind] = int(travelTime[ind] + time)
            rowIndex[ind] = np.where(self.predictedOcc[:, 0] == estimateArrivalTime[ind])[0][0]  # Find the row for predicted arrival time
            ratio[ind] = interChangeable[ind].predictedOcc[rowIndex[ind]][1] / interChangeable[ind].cap  # predicted fill level of station at time of arrival

        # use the average fill level of all interchangeable stations as the "target" fill level
        multiplier = np.mean(ratio)

        for ind in range(numberInterChangeable):
            freeDocks[ind] = interChangeable[ind].cap - interChangeable[ind].predictedOcc[rowIndex[ind]][1]
            # use the target free dock level to determine the number of "extra docks" at each station
            extraDocks[ind] = max(interChangeable[ind].cap * multiplier - interChangeable[ind].predictedOcc[rowIndex[ind]][1], 0)
            waiting[ind] = max(-freeDocks[ind], 0)  # number of people waiting to dock their bike at station
            freeDocks[ind] = max(freeDocks[ind], 0)

        totFreeDocks = sum(freeDocks)
        totExtraDocks = sum(extraDocks)
        totQueue = sum(waiting)
        # if no neighbouring stations for destination or full everywhere but no queue anywhere, go to initially
        # requested arrival station
        if numberInterChangeable == 1 or (totFreeDocks == 0 and totQueue == 0):
            choice = self
        # if full everywhere but there are people waiting for docks
        elif totFreeDocks == 0:
            # get the number of possible stations that have do not have a queue
            noQueue = numberInterChangeable - np.count_nonzero(np.array(waiting))
            # if there exists stations that do not have a wait yet
            if noQueue > 0:
                # assign probabilities for arrival station equally across those without a wait, since all full anyways
                dockProb = np.where(np.array(waiting) == 0, 1, 0) / noQueue
            # if all stations have a wait, assign higher chance of going to station with smaller wait
            else:
                no_waiting = 1 / np.array(waiting)
                dockProb = np.array(no_waiting) / sum(no_waiting)
        # if extra docks exist somewhere:
        elif totExtraDocks:
            dockProb = np.array(extraDocks) / totExtraDocks
        # if no extra docks anywhere, but there are docks available:
        else:
            dockProb = np.array(freeDocks) / totFreeDocks

        if choice is None:  # would be assigned return station if no interchangeable stations or no docks available anywhere
            choice = np.random.choice(interChangeable, p=dockProb)

        if isAutomated:  # not client query
            index = interChangeable.index(choice)
            choice.predictedOcc[rowIndex[index]:-1, 1] += 1
            return choice, estimateArrivalTime[index]
        else:  # client query
            return choice, estimateArrivalTime, interChangeable, freeDocks

    # method is called when the user arrives to dock their bike at this station
    def dockBike(self, bike, time, customer):
        # extract the estimated arrival time of arriving bicycle
        estArr = bike.lastTrip[4]
        # if bike has arrived earlier than expected, adjust the predicted occupancy information to reflect this
        if estArr > time:
            rowIndex = np.where(self.predictedOcc[:, 0] == estArr)[0][0]
            self.predictedOcc[time:rowIndex, 1] += 1
        bikesInJourney.remove(bike)
        self.bikes.append(bike)
        # if there are free docks available, dock the bike. Otherwise, add bike into list of bikes waiting
        if len(self.bikes) < self.cap:
            bike.state = "Docked"
            bike.checkedOutTo = None
            customer.bikeCheckedOut = None
        else:
            self.waitingToDock.append(bike)
            bike.state = "Waiting To Dock"

    # method is called when the user arrives to borrow a bike at this station - only for automated demand
    def borrowBike(self, time, dest, customer):
        # if bike(s) are available to be borrowed:
        if len(self.bikes) > 0:
            # at checkout time, confirm the arrival station
            choice, estArrTime = dest.queryDock(int(time), self, customer)
            # checking out a bike (creating trip in simulation)
            route = self.relationDict[choice].routeName
            traci.vehicle.add(self.bikes[0].id, route, typeID="reroutingType")
            self.bikes[0].lastTrip = [time, self, int, choice, estArrTime, True]
            self.bikes[0].state = "Riding"
            self.bikes[0].checkedOutTo = customer
            customer.bikeCheckedOut = self.bikes[0]
            bikesInJourney.append(self.bikes[0])
            # pop the bike from the list of available bikes
            self.bikes.remove(self.bikes[0])
        else:
            self.unsatDemand += 1
        # if after borrowing, a dock frees up and there are people waiting to dock, allow the first in line to dock.
        if self.waitingToDock:
            self.waitingToDock[0].state = "Docked"
            self.waitingToDock[0].checkedOutTo.bikeCheckedOut = None
            self.waitingToDock[0].checkedOutTo = None
            self.waitingToDock.pop(0)

    # method is called when the APP user arrives to borrow a bike at this station
    def borrowBikeApp(self, customer, time):
        if len(self.bikes) > 0:
            # Don't create a trip with simulation for client since we don't know where they are travelling yet...
            self.bikes[0].state = "App check out"
            self.bikes[0].checkedOutTo = customer
            self.bikes[0].lastTrip[0] = time
            self.bikes[0].lastTrip[1] = self
            customer.bikeCheckedOut = self.bikes[0]
            bikesInJourney.append(self.bikes[0])
            self.bikes.remove(self.bikes[0])
            # if after borrowing, a dock frees up and there are people waiting to dock, allow the first in line to dock.
            if self.waitingToDock:
                self.waitingToDock[0].state = "Docked"
                self.waitingToDock[0].checkedOutTo.bikeCheckedOut = None
                self.waitingToDock[0].checkedOutTo = None
                self.waitingToDock.pop(0)
            return True, customer  # return that the client was able to borrow a bike
        self.unsatDemand += 1
        return False, customer  # return that the client was unable to borrow a bike

    # method is called when the APP user arrives to dock their bike at this station but didn't book it
    def dockBikeWrong(self, bike, time, customer):
        # extract the estimated arrival time of arriving bicycle
        estArr = bike.lastTrip[4]
        # if bike has arrived earlier than expected, adjust the predicted occupancy information of
        # the correct station to reflect
        if estArr is not int and estArr > time:
            rowIndex = np.where(customer.arriveAt.predictedOcc[:, 0] == estArr)[0][0]
            customer.arriveAt.predictedOcc[time:rowIndex, 1] += 1

        # Dock the bike at the station and update bike/customer objects
        bikesInJourney.remove(bike)
        self.bikes.append(bike)
        bike.state = "Docked"
        bike.checkedOutTo = None
        customer.bikeCheckedOut = None

    # ==================================================================================================================
    #                      FOLLOWING METHODS ARE CALLED ONLY ONCE ON SETUP OF SIMULATION
    # ==================================================================================================================
    # this method sets up relationship between this station and the other stations on the map
    def update_stationRelation(self):
        # for station in stations:
        for route in self.routes:
            traci.route.add(route[0] + "To" + route[1], [route[2], route[3]])
            self.update_relation(stat_dict[route[1]], route[4])
        dictKeys = []
        for relation in self.otherStations:
            dictKeys.append(relation.arrive)
        # station object to relation object
        self.relationDict = dict(zip(dictKeys, self.otherStations))

    # this method will initialise station-station parameters like flow rate, distance, routes etc..
    def update_relation(self, targetStat, minCost):
        min_cyc_dis = 500  # minimum distance that people are going to cycle (instead of walk)
        max_rate = 1 / 1000  # maximum flow rate
        dis_for_maxrate = 1500  # distance between station where max expected flow rate occurs
        max_dis = 5000  # maximum bound where expected flow rate reaches 0
        rate = 0
        if min_cyc_dis < minCost < max_dis:  # if distance between stations greater than minimum cycling distance:
            if minCost < dis_for_maxrate:
                rate = (max_rate / (dis_for_maxrate - min_cyc_dis)) * (minCost - min_cyc_dis)
            else:
                rate = (max_rate / (max_dis - dis_for_maxrate)) * (max_dis - minCost)
            rate = np.random.normal(rate, (rate / 15) ** 0.5, 1)[0]  # output is an array of 1 element
            rate = min(rate, 1/600)
            if rate > 0:  # if demand exists: ie, flow rate > 0
                # Time of next demand for bike according to poisson dist:
                nextDemand = 300  # Customer to arrive 5 minutes after the start of simulation to begin with
                # (so system can collect the data it needs to set itself up)
                while nextDemand < endSim:
                    nextDemand += round(-math.log(1.0 - random.uniform(0, 1)) / rate)
                    customerId = len(cust_dict)
                    newCustomer = Customer(customerId, "dummy", max(random.gauss(1000, 100), 500))
                    # customerDatabase.append(newCustomer)
                    cust_dict.update({str(customerId): newCustomer})
                    totalDemand.append([nextDemand, self, targetStat, newCustomer])
            else:
                rate = 0
        self.otherStations.append(StationRelation(targetStat, self.name + "To" + targetStat.name, minCost, rate))


class StationRelation(object):
    def __init__(self, arrival, name, distance, rate):
        self.arrive = arrival  # arriving station
        self.routeName = name  # name of routes for traci call
        self.distance = distance * 1.25
        self.rate = rate  # poisson distribution (lambda rate)
        self.timeTaken = self.distance / 4

    def updateLastTime(self, arrivingBike):
        self.timeTaken = arrivingBike.lastTrip[2] - arrivingBike.lastTrip[0]


class Bike(object):
    def __init__(self, bike_no, state):
        self.id = str(bike_no)
        # [dep Time, dep station, arr time, arr Stat, est arr Time, satisfied?]
        self.lastTrip = [int, Station, int, Station, int, True]
        self.state = state
        self.checkedOutTo = Customer



class Customer(object):
    def __init__(self, id, name, maxWalk):
        self.id = id
        self.maxWalk = maxWalk
        self.name = name
        self.journeyHistory = []
        self.bikeCheckedOut = Bike
        self.departFrom = Station
        self.arriveAt = Station


def passStationInfoToClient():
    output = ""
    for station in stations:
        output += station.name + "#" + str(len(station.bikes)) + "#"
    return output

# This method runs the simulation. ie: it does all the work of borrowing/docking bikes etc dynamically
def run():
    step = 0  # initialise to step
    station_occ = [[] for x in range(len(stations))]
    unsat_demand = [[] for x in range(len(stations))]

    now = datetime.datetime.now()
    startTime = now.hour * 60 + now.minute  # time of day in minutes at the start of the simulation

    # This simulation only allows one client to interface with the simulation at once
    user = Customer

    pred_graph = []
    while step < endSim:  #continue until end of simulation
        # DynamicInfo updated every step of simulation according to current station occupancy levels
        # This is sent to the client when they request it
        dynamicInfo = []
        for station in stations:
            d = {'occ': int(len(station.bikes)),
                 'id': station.name
                 }
            dynamicInfo.append(d)
        stationDynamic.value = json.dumps(dynamicInfo)

        thread.join(1)  # pause sumo simulation thread for 1s to allow server to listen to client (blocks the calling thread)


        # Have simulation respond to requests made by user through the android app here:
        if incoming.value != "":
            msg = json.loads(incoming.value)  # unpack incoming json string into dictionary format
            incoming.value = ""  # reset the incoming.value (shared variable with the server thread)
            # this is essentially called once when the user first logs in, so that the simulation knows which user has
            # connected with the server
            if "name" in msg:
                if msg['username'] not in cust_dict:  # if first time the user has logged in during current simulation
                    user = Customer(msg['username'], msg['name'], 0)  # create a Customer Object for this user
                    cust_dict.update({msg['username']: user})  # append the user to the dictionary
                    outgoing.value = "serverRestart"
                else: # If not first time the user has logged in during the current simulation
                    user = cust_dict.get(msg['username']) # get the appropriate Customer Object from dictionary
                    outgoing.value = "serverContinue"
            elif msg["key"] == "queryDepart":  # The client wants to book a bike
                departingStation = stat_dict[msg["id"]]
                user.maxWalk = msg["distance"]
                now = datetime.datetime.now()
                nowInMin = now.hour * 60 + now.minute  # current time in minutes
                departAt = (msg["time"] - nowInMin) * 60 + step  # depart time in terms of steps (for the simulation)
                assignedDeparture, interChangeable, freeBikes = departingStation.queryBorrow(departAt, None, False, user)
                if assignedDeparture is None:  # no bikes available anywhere within walkable distance!
                    print("No bikes available")
                    outgoing.value = "empty"  # let the client know all stations are prediected to be empty
                else:
                    departQueryReturn = []  # This list contains the interchangeable stations and their predicted
                    # occupancies as well as which station has been "assigned" to the user
                    for ind in range(len(interChangeable)):
                        d = {
                            'occ': int(len(interChangeable[ind].bikes)),
                            'id': interChangeable[ind].name,
                            'assigned': 0,
                            'predictedOcc': int(freeBikes[ind])
                        }
                        if interChangeable[ind] == assignedDeparture:
                            d['assigned'] = 1
                        departQueryReturn.append(d)
                    outgoing.value = json.dumps(departQueryReturn)  # sends this info back to the client
            elif msg["key"] == "confirmDepartingStation":  # client has selected a station that they would like to depart from
                now = datetime.datetime.now()
                nowInMin = now.hour * 60 + now.minute
                departAt = (msg["time"] - nowInMin) * 60 + step
                rowIndex = np.where(stat_dict[msg['id']].predictedOcc[:, 0] == departAt)[0][0]  # update the predicted occupancies of the selected station
                stat_dict[msg['id']].predictedOcc[rowIndex:-1, 1] -= 1
                user.departFrom = stat_dict[msg['id']]  # update the Customer Object to reflect this
            elif msg["key"] == "QRScanned":  # User has scanned a QR codes
                if int(msg["id"]) == user.departFrom.QR:  # if scanned QR code matches the station booked
                    outgoing.value = "success"
                    result, user = user.departFrom.borrowBikeApp(user, step)  # borrow the bike from station
                    if not result:  # If the station was empty
                        outgoing.value = "empty"  # Let the user know he/she scanned the QR code of an empty dock
                else:  # QR code scanned doesn't match the booked station
                    outgoing.value = qr_dict[int(msg['id'])].name  # Let the user know they scanned QR code of incorrect station
            elif msg["key"] == "queryArrival":  # Client wants to book a return dock
                arrivingStation = stat_dict[msg["id"]]
                user.maxWalk = msg["distance"]
                assignedArrival, estArr, interChangeable, freeDocks = arrivingStation.queryDock(step, user.departFrom, user, msg['time'], False)
                arrivalQueryReturn = []  # This list contains the interchangeable stations and their predicted
                # occupancies as well as which station has been "assigned" to the user and estimated time of arrival
                # if the user is travelling directly there
                for ind in range(len(interChangeable)):
                    d = {
                        'occ': len(interChangeable[ind].bikes),
                        'id': interChangeable[ind].name,
                        'assigned': 0,
                        'predictedDocks': int(freeDocks[ind]),
                        'estArr': startTime + int(estArr[ind] / 60)   # est time of arrival at destination
                    }
                    if interChangeable[ind] == assignedArrival:
                        d['assigned'] = 1
                    arrivalQueryReturn.append(d)
                outgoing.value = json.dumps(arrivalQueryReturn)  # send this info back to the client
            elif msg["key"] == "confirmArrivalStation":  # Client has selected a station to reserve dock at
                now = datetime.datetime.now()
                nowInMin = now.hour * 60 + now.minute
                arriveAt = (msg["time"] - nowInMin) * 60 + step
                rowIndex = np.where(stat_dict[msg['id']].predictedOcc[:, 0] == arriveAt)[0][0]  # updated predicted occupancy levels of selected station to reflect
                user.bikeCheckedOut.lastTrip[3] = stat_dict[msg['id']]
                user.bikeCheckedOut.lastTrip[4] = int(arriveAt)  # update user for estimated time of arrival
                stat_dict[msg['id']].predictedOcc[rowIndex:-1, 1] += 1
                user.arriveAt = stat_dict[msg['id']]
            elif msg["key"] == "cancelDeparture":  # client wants to cancel booked bike
                departAt = (msg["time"] - startTime) * 60
                if step < departAt:  # if current simulation step before booked departure time update the predicted occupancy of booked station
                    rowIndex = np.where(stat_dict[msg['id']].predictedOcc[:, 0] == departAt)[0][0]
                    stat_dict[msg['id']].predictedOcc[rowIndex:-1, 1] += 1
                user.departFrom = Station
            elif msg["key"] == "cancelArrival":  # client wants to cancel booked dock
                arriveAt = (msg["time"] - startTime) * 60
                if step < arriveAt:  # if current simulation step before booked arrival time update the predicted occupancy of booked station
                    rowIndex = np.where(stat_dict[msg['id']].predictedOcc[:, 0] == arriveAt)[0][0]
                    stat_dict[msg['id']].predictedOcc[rowIndex:-1, 1] -= 1
                user.arriveAt = Station
            elif msg["key"] == "docked":  # if the bike has been docked
                user.bikeCheckedOut.lastTrip[2] = step  # update the user's arrival time and arrival station
                user.bikeCheckedOut.lastTrip[3] = stat_dict[msg['id']]
                prevBike = user.bikeCheckedOut
                if user.arriveAt is not None and stat_dict[msg['id']] == user.arriveAt: # if docked at the station booked
                    stat_dict[msg['id']].dockBike(user.bikeCheckedOut, step, user)
                else:  # if docked at the "wrong" station
                    stat_dict[msg['id']].dockBikeWrong(user.bikeCheckedOut, step, user)

                depTimeInSec = startTime * 60 + prevBike.lastTrip[0]
                arrTimeInSec = startTime * 60 + prevBike.lastTrip[2]
                date = datetime.datetime.today()
                date_str = date.strftime('%A') + ", " + date.strftime("%d") + " " + date.strftime("%B") + " " + date.strftime("%Y")  # date of trip
                d = {  # dictionary containing the trip details to be stored in the database on the server thread
                    'username': user.id,
                    'startStation': prevBike.lastTrip[1].name,
                    'endStation': msg['id'],
                    'date': date_str,
                    'startTime': depTimeInSec/60,
                    'endTime': arrTimeInSec/60,
                    'bike': prevBike.id,
                    'duration': (arrTimeInSec - depTimeInSec) # in seconds
                }
                outgoing.value = json.dumps(d)

        # ===============================================================================
        # Simulation for the automated users generated at initialisation below:
        # ===============================================================================
        delay = int(random.uniform(60, 300))  # people will book bikes at different times ahead of checking them out
        while step + delay >= demand[0][0]:
            depTime, depStat, arrStat, customer = demand[0]
            depStat.queryBorrow(demand[0][0], arrStat, True, customer)  # "books" a bike and assumes the rider will abide
            demand.pop(0)

        # if customer due to leave (based on their booking)
        while departureTimes and step >= departureTimes[0][0]:
            depTime, depStat, arrStat, customer = departureTimes[0]
            depStat.borrowBike(departureTimes[0][0], arrStat, customer)  # borrow a bike and "books" a dock at arrival station
            departureTimes.pop(0)

        # if bikes have arrived to destination:
        for arrivedBike in traci.simulation.getArrivedIDList():
            bike = allBikes[int(arrivedBike)]
            bike.lastTrip[2] = step  # arrival time
            _, depStat, _, arrStat, _, _ = bike.lastTrip
            depStat.relationDict[arrStat].updateLastTime(bike)  # update the time taken from dest->arrival in the system for next customer
            arrStat.dockBike(bike, step, bike.checkedOutTo) # dock the bike at the appropriate station

        # update the predicted occupancy to account for bikes which are taking longer than expected to arrive
        # exclude the client borrowed one..
        for bike in bikesInJourney:
            if bike.state != "App check out":
                if bike.lastTrip[4] < step:
                    bike.lastTrip[3].predictedOcc[0][1] -= 1

        for ind in range(len(stations)):
            # check if customers have been waiting to dock for more than 90 seconds at any station to increment unsatisfied customer count
            for bike in stations[ind].waitingToDock:
                if (step - bike.lastTrip[2]) > 90 and bike.lastTrip[5]:
                    bike.lastTrip[5] = False
                    stations[ind].unsatDemand += 1
            # log the necessary information about each station
            station_occ[ind].append((len(stations[ind].bikes) + len(stations[ind].waitingToDock)) / stations[ind].cap * 100)
            unsat_demand[ind].append(stations[ind].unsatDemand)

            # update the predicted occupancy (ie: extend it out for one more second)
            stations[ind].predictedOcc = np.reshape(np.delete(stations[ind].predictedOcc, 0, 0), (-1, 2))
            stations[ind].predictedOcc = np.append(stations[ind].predictedOcc, [stations[ind].predictedOcc[-1][0] + 1, stations[ind].predictedOcc[-1][1]])
            stations[ind].predictedOcc = np.reshape(stations[ind].predictedOcc, (-1, 2))



        # Go to the next time step
        step += 1
        traci.simulationStep()

    traci.close()
    sys.stdout.flush()

    return station_occ, unsat_demand

# main entry point for Python Script
if __name__ == "__main__":

    # run simulation using SUMO GUI or not
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # traci starts sumo as a subprocess and then this script connects and runs
    sumo = [sumoBinary, "-c", "melbourne.sumocfg"]
    traci.start(sumo)

    # get net information
    net = traci.sumolib.net.readNet("melbourne_larger.net.xml")

    # empty lists to store station/customer/demand information to be generated
    stations = []
    station_keys = []
    qr_keys = []
    customerDatabase = []

    totalDemand = []
    departureTimes = []
    allBikes = []
    bikesInJourney = []
    bikeID = 0
    ini_full = .8  # percentage occupancy of stations at the start of simulation
    endSim = int(10800)  # length of simulation in seconds (3 hrs)

    with open('stations.txt') as dataFile:  # we keep the stations static (generated in another script and stored in stations.txt)
        dataLoaded = json.load(dataFile)  # extract the necessary attributes from txt file (list of objects)
        for ind in range(len(dataLoaded)):
            station = dataLoaded[ind].get("name")
            longlat = dataLoaded[ind].get("longLat")
            cap = 44#int(dataLoaded[ind].get("cap") * 2)
            node = traci.sumolib.net.Net.getNode(net, dataLoaded[ind].get("node"))  # get the node corresponding to the node id in the json object
            x, y = node.getCoord()  # get the x,y coordinates of the node
            longitude, latitude = traci.sumolib.net.Net.convertXY2LonLat(net, x, y)  # get the latitude and longitude of the node
            new_st = Station(station, [x, y, "xy"], longlat, cap)  # create Station Object using json object
            new_st.routes = dataLoaded[ind].get("routes")  # routes = id:station x to station y, edges:(start, end)
            new_st.QR = dataLoaded[ind].get("QR")  # QR code of the station to borrow bike with
            stations.append(new_st)  # append the station to the station list
            station_keys.append(new_st.name)  # keys for easier access using dictionaries
            qr_keys.append(new_st.QR)

    stat_dict = dict(zip(station_keys, stations))  # create dictionary for easier access of station objects
    qr_dict = dict(zip(qr_keys, stations))  # create dictionary for easier access of stations according to qr codes
    cust_dict = dict()  # initialise an empty dict for customers

    predictionInterval = 60 * 60 * 3  # predict occupancy for up to 2 hours in advance
    tempIndexArray = [*range(predictionInterval)]  # time index of predicted occupancies
    for station in stations:
        station.update_stationRelation()  # generate the "demand" for the simulation
        for bike in range(int(station.cap * ini_full)):  # create bikes and store them at the stations
            new_bike = Bike(bikeID, "Docked")
            station.bikes.append(new_bike)
            allBikes.append(new_bike)
            bikeID += 1
        station.predictedOcc = [len(station.bikes)] * predictionInterval  # initialise predicted occupancy of stations
        station.predictedOcc = np.column_stack((tempIndexArray, station.predictedOcc))

    totalDemand = sorted(totalDemand, key=itemgetter(0))  # sort the generated demand of all stations by time
    demand = totalDemand[:]

    # initialise station info that is to be sent to the client when they connect
    staticInfo = []
    dynamicInfo = []
    for station in stations:
        static = {
            'lat': float(station.longLat[1]), # lat and long used to get address in app
            'long': float(station.longLat[0]),
            'cap': int(station.cap),
            'id': station.name
        }
        dynamic = {
            'occ': int(len(station.bikes)),
            'id': station.name
             }
        staticInfo.append(static)
        dynamicInfo.append(dynamic)
    stationInfoStatic = json.dumps(staticInfo)  # convert to json string
    stationInfoDynamic = json.dumps(dynamicInfo)

    with mp.Manager() as manager:  # multiprocessing manager
        stationStatic = manager.Value('c', stationInfoStatic)
        stationDynamic = manager.Value('c', stationInfoDynamic)
        incoming = manager.Value('c', "")
        outgoing = manager.Value('c', "")
        thread = mp.Process(target=server.server, args=(stationStatic, stationDynamic, incoming, outgoing))  # create thread (server)
        print("Launching the server.")
        thread.start()  # start the thread
        print("The server has been launched")

        station_occ, unsat_demand = run()  # run the simulation

    # plotting the results at end of simulation (the fill levels of each station and the # of unsatisfied customers)
    time = np.arange(0, endSim)
    xlim = np.arange(0, 60 * 60 * 3.5, 60 * 60)
    plt.figure(1)
    for i in range(len(stations)):
        plt.plot(time, station_occ[i])
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Fill Level')
    # plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
    ax.yaxis.set_major_formatter(PercentFormatter())
    qos = []
    plt.figure(2)
    totalunsat = 0
    for i in range(len(stations)):
        plt.plot(time, unsat_demand[i])
        totalunsat += unsat_demand[i][-1]
    ax = plt.gca()
    ax.set_xbound(lower=0)
    ax.set_ybound(lower=0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Accumulated number of out-of-stock events')
    # plt.legend([station.name for station in stations])
    plt.xticks(xlim, [str(n).zfill(2) for n in np.arange(0, 3.5, 1)])
    qos.append(1 - totalunsat / len(totalDemand))

    plt.show()


