import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pymongo import MongoClient
from pprint import pprint
import time


CONNECTION_STRING ="mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(CONNECTION_STRING)
db=client.admin
# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
#pprint(serverStatusResult)


def get_area_data(area_name: str):
    date_val = []

    floor = df[(df["area_code"] == area_name)]

    for t in floor.timestamp:
        ts = int(t / 1000)
        time_value = datetime.fromtimestamp(ts)
        date_val.append(time_value)

    date_val = sorted(date_val)
    return date_val, floor


if __name__ == '__main__':
    # Get the database
    db_name = client.get_database("venezia-project")
    db_collection=db_name["totals-per-area"]

    totals=db_collection.count_documents({})

    print("TOTALE ELEMENTI",totals)

    start = time.time()
    print("START TIME: ", start)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    df = pd.read_csv("./data/Museo M9/totals_per_area.csv")
    timestamp = df.timestamp
    areaCode = df.area_code
    totals = df.totals
    alarms = df.alarms
    date = df.date



    end = time.time()
    print("END TIME:",end)
    print(end - start)
    # print(df.head(5))
    # print(timestamp)
    # print(areaCode)
    # print(totals)
    # print(alarms)
    # print(date)