# =====================================================================================================================
# HELEN ZUO 27807746
# Monash University 2020
# ECSE Final Year Project
# =====================================================================================================================

import json
import socket
import time
import pandas as pd
import numpy as np

# This is the "server" that will talk to the android app
def server(stationStatic, stationDynamic, incomingMsg, outgoingMsg):
    # size of buffer and backlog
    buffer = 4096  # value should be a relatively small power of 2, e.g. 4096
    backlog = 1  # tells the operating system to keep a backlog of 1 connection;
    # this means that you can have at most 1 client waiting while the server is handling the current client;
    # the operating system will typically allow a maximum of 5 waiting connections; to cope with this,
    # busy servers need to generate a new thread to handle each incoming connection so that it can quickly
    # serve the queue of waiting clients

    # create a socket: AF_INET = IPv4 socket family; SOCK_STREAM = TCP socket type
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket to an address and port
    host = '192.168.20.9'  # LAN IP Address (to allow access via phone)
    port = 8080  # reserve a port for the service (i.e. a large number less than 2^16);
    # the call will fail if some other application is already using this port number on the same machine
    server_socket.bind((host, port))  # binds the socket to the hostname and port number

    # listen for incoming connections
    server_socket.listen(backlog)

    loggedIn = False

    # Retrieve the database for users and trips from Excel
    users_df = pd.read_excel(r'D:\Google Drive\Uni\FYP\ClientApp\users.xlsx', sheet_name='Users', converters={'mobile': str,'dob': str})
    users_df = users_df.replace(np.nan, 'None', regex=True)
    trips_df = pd.read_excel(r'D:\Google Drive\Uni\FYP\ClientApp\trips.xlsx', sheet_name='Trips')

    while True:
        # passively accept TCP client connections; the call returns a pair of arguments: client is a new
        # Socket object used to communicate with the client and address is the address of the client
        client_socket, address = server_socket.accept()
        client_socket.setblocking(True)  # blocks the thread at rcv call

        packet = str(client_socket.recv(buffer), encoding='utf-8')  # receive client data into buffer
        obj = json.loads(packet)  # unpack the json string into a dictionary

        if "newUser" in obj and not loggedIn:  # If pinging from "sign-up fragment"
            if obj["newUser"] == "signUp":
                result, users_df = checkForUserSignUp(packet, users_df)
                if result == 0:  # both email and user exists
                    write_utf8("both", client_socket)
                elif result == -1:  # email exists
                    write_utf8("email", client_socket)
                elif result == -2:  # username exists
                    write_utf8("username", client_socket)
                else:
                    write_utf8("received", client_socket)
            elif obj["newUser"] == "logIn":  # If pinging from "log-in fragment"
                result, user = checkFoUserLogin(packet, users_df)
                if result:
                    write_utf8(user, client_socket)
                else:
                    write_utf8("fail", client_socket)
            elif obj['newUser'] == "loggedIn":  # After reaching the "MainActivity", app pings again to retrieve info
                # Time that user has logged in/signed up as seen from the server
                start_time = time.strftime('%d %b %Y at %H:%M:%S')
                init_time = str(start_time)  # convert connection time to a string
                print(obj['name'], ' has made a connection with', address, 'on', init_time + '.')
                incomingMsg.value = packet  # to share this info with the main thread (which user has just logged in)
                while outgoingMsg.value == "":
                    pass

                d = {'staticInfo': stationStatic.value,
                     'dynamicInfo': stationDynamic.value,
                     'tripInfo': trips_df.to_json(orient="records"),
                     'server':outgoingMsg.value
                }
                write_utf8(json.dumps(d), client_socket)  # Send the relevant station/trip info to client
                loggedIn = True
                outgoingMsg.value = ""
        else:
            if "key" not in obj:  # update user profile of connected client
                users_df = updateUserInfo(packet, users_df)
                client_socket.close()
            else:
                key = obj["key"]
                if key == "quit":   # client has closed the app or returned to the login page
                    print("Disconnecting client... ")
                    client_socket.close()  # Close the connection with the client
                    loggedIn = False
                incomingMsg.value = packet
                # these require info to be sent back to the client (info retrieved from main thread)
                if key == "queryDepart" or key == "QRScanned" or key == "queryArrival" or key == "waitingToDock":
                    while True:
                        if outgoingMsg.value != "":  # outgoingMsg.value is set in the main thread and is sent to the client
                            write_utf8(outgoingMsg.value, client_socket)
                            outgoingMsg.value = ""
                            break
                elif key == "confirmDepartingStation" or key == "confirmArrivalStation" or key == "cancelDeparture" or \
                        key == "cancelArrival":
                    client_socket.close()  # the socket is closed after every msg on the client end too
                elif key == "refresh":  # the user has refreshed their screen
                    d = {'stationInfo': stationDynamic.value,
                         'tripInfo': trips_df.to_json(orient="records")}
                    write_utf8(json.dumps(d), client_socket)
                elif key == "checkDock":  # client pings to see if bike has been docked periodically
                    with open('dock.txt', 'r') as dockText:  # where the bike has been docked is stored in a txt file at the moment
                        inp = dockText.read()
                        text_file = open('dock.txt', 'w')  # overwrite the entry after it has been read
                        text_file.write("")
                        text_file.close()
                        if inp != "":
                            d = {
                                "key": "docked",
                                "id": str(inp).lower().strip()  # lowercase and trim the string
                            }
                            incomingMsg.value = json.dumps(d)
                            while outgoingMsg.value == "":  # wait for the main thread to update station occ to account for bike being docked
                                pass
                            outgoingObj = json.loads(outgoingMsg.value)
                            write_utf8(outgoingObj['endStation'], client_socket)  # let the client know that the bike has been docked (and the station it has been docked at)
                            trips_df = addTripToDf(outgoingObj, trips_df)  # add completed trip to our database
                            outgoingMsg.value = ""
                        else:
                            write_utf8("", client_socket)

# call to send string messages to java (will encode into UTF-8)
def write_utf8(s, sock):
    sock.sendall(bytes(s, encoding="utf-8"))
    sock.close()

# call to update user info in database and save into excel spreadsheet
def updateUserInfo(user_string, df):
    user = json.loads(user_string)
    row = df["username"] == user["username"]  # get the row where username matches
    df.loc[row, 'email'] = user['email']
    df.loc[row, 'name'] = user['name']
    df.loc[row, 'mobile'] = user['mobile']
    df.loc[row, 'gender'] = user['gender']
    df.loc[row, 'dob'] = user['dob']
    df['favStations'] = df['favStations'].astype('object')
    df.at[df[df['username'] == user["username"]].index[0], 'favStations'] = json.dumps(user['favStations'])  # save station list as json string
    df.to_excel(r'D:\Google Drive\Uni\FYP\ClientApp\users.xlsx', sheet_name='Users', index=False)
    return df

# checks if username/email and password entered match a user in our database
def checkFoUserLogin(user_string, df):
    user = json.loads(user_string)
    if "email" in user:  # if email entered by client, look for the row in database that matches the email address
        key = "email"
    else:  # if username entered by client, look for the row in the database that matches the username
        key = "username"
    if not df.empty and user[key] in set(pd.Series(list(df[key]))):    # If the database is not empty and contains the entered username or email
        row = df.loc[df[key] == user[key]]
        if user['password'] == row['password'].item():  # check if the passwords match too
            d = {
                'email': row["email"].item(),
                'name': row["name"].item(),
                'username': row["username"].item(),
                'password': row["password"].item(),
                'mobile': row["mobile"].item(),
                'dob': row["dob"].item(),
                'gender': row["gender"].item(),
                'favStations': json.loads(row["favStations"].item()),
            }
            return True, json.dumps(d)  # if logins match, then return user info to the client
    return False, None  # either username and/or passwords do not match the database info

# Checks if the username/email already exist in database. If not, store the new user into database
def checkForUserSignUp(user_string, df_user):
    user = json.loads(user_string)
    if not df_user.empty:  # if the database is not empty
        if user["email"] in list(df_user.email.values) and user["username"] in list(df_user.username.values):  # both username and emails already exist in database
            return 0, df_user
        if user["email"] in list(df_user.email.values):  # email already associated with a user
            return -1, df_user
        if user["username"] in list(df_user.username.values):  # username taken by someone else
            return -2, df_user
    # Save the new user into database if username and email are valid
    d = {
        'email': user["email"],
        'name': user["name"],
        'username': user["username"],
        'password': user["password"],
        'gender': 2,
        'mobile': "None",
        'dob': "None",
        'favStations': None
    }

    if not df_user.empty:
        df_user = df_user.append(d, ignore_index=True)  # append to user database
    else:
        df_user = pd.DataFrame(d, index=[0])  # create a new user database if empty

    df_user.to_excel(r'D:\Google Drive\Uni\FYP\ClientApp\users.xlsx', sheet_name='Users', index=False)  # store into excel spreadsheet
    return 1, df_user


# Adds a COMPLETED trip into the trips database
def addTripToDf(trip, df_trip):
    d = {  # convert relevant info into dictionary format to be appended to pandas dataframe
        'username': trip['username'],
        'startStation': trip['startStation'],
        'endStation': trip['endStation'],
        'date': trip['date'],
        'startTime': trip['startTime'],
        'endTime': trip['endTime'],
        'bike': trip['bike'],
        'duration': trip['duration']
    }

    if not df_trip.empty:
        df_trip = df_trip.append(d, ignore_index=True)  # append to trips database
    else:
        df_trip = pd.DataFrame(d, index=[0])  # create a new trips database if empty

    df_trip.to_excel(r'D:\Google Drive\Uni\FYP\ClientApp\trips.xlsx', sheet_name='Trips', index=False)  # store into excel spreadsheet

    return df_trip