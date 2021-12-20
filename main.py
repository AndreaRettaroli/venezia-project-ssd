from datetime import datetime

import pandas as pd
import pymongo
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pymongo import MongoClient
from config import CONNECTION_STRING


def connect_to_db():
    client = MongoClient(CONNECTION_STRING)

    # db = client.admin
    # serverStatusResult = db.command("serverStatus")
    # pprint(serverStatusResult)

    return client


def convert_timestamp(df):
    date = []

    for t in df.timestamp:
        t = int(t)
        ts = int(t / 1000)
        time_value = datetime.fromtimestamp(ts)
        date.append(time_value)

    return date


if __name__ == '__main__':
    # Database connection
    mongo_client = connect_to_db()
    db_name = mongo_client.get_database("venezia-project")
    total_per_area_collection = db_name["totals-per-area"]

    db_sorted = mongo_client.get_database("venezia-sorted")
    total_per_area_sorted_collection = db_sorted["totals-per-area-sorted"]

    # Find by area code
    cursors_floor_1 = total_per_area_collection.find({"area_code": "floor_1"})
    cursors_floor_2 = total_per_area_collection.find({"area_code": "floor_2"})
    cursors_floor_3 = total_per_area_collection.find({"area_code": "floor_3"})

    # Convert to dataframe
    df_floor1 = pd.DataFrame(list(cursors_floor_1))
    df_floor1["totals"] = df_floor1["totals"].astype("int32")
    df_floor1 = df_floor1.sort_values(by="timestamp", ascending=True)
    # df_floor1 = df_floor1.iloc[1::300, :]  # Ogni 10 minuti
    df_floor2 = pd.DataFrame(list(cursors_floor_2))
    df_floor2["totals"] = df_floor2["totals"].astype("int32")
    df_floor2 = df_floor2.sort_values(by="timestamp", ascending=True)
    # df_floor2 = df_floor2.iloc[1::1800, :]
    df_floor3 = pd.DataFrame(list(cursors_floor_3))
    df_floor3["totals"] = df_floor3["totals"].astype("int32")
    df_floor3 = df_floor3.sort_values(by="timestamp", ascending=True)
    # df_floor3 = df_floor3.iloc[1::300, :]
    # Convert timestamp to date and add a column to dataframe
    date_val_floor_1 = convert_timestamp(df_floor1)
    df_floor1["date"] = date_val_floor_1

    date_val_floor_2 = convert_timestamp(df_floor2)
    df_floor2["date"] = date_val_floor_2

    date_val_floor_3 = convert_timestamp(df_floor3)
    df_floor3["date"] = date_val_floor_3

    # Get month's range
    df_floor1["date"] = pd.to_datetime(df_floor1.date)
    mask = (df_floor1.date > '2021-9-1') & (df_floor1.date <= '2021-10-1')
    df_floor1 = df_floor1.loc[mask]

    df_floor2["date"] = pd.to_datetime(df_floor2.date)
    mask = (df_floor2.date > '2021-8-1') & (df_floor2.date <= '2021-10-1')
    df_floor2 = df_floor2.loc[mask]

    # Plot
    figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df_floor1.date, df_floor1.totals, color="green", label="Floor_1")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df_floor2.date, df_floor2.totals, color="red", label="Floor_2")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(df_floor3.date, df_floor3.totals, color="blue", label="Floor_3")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(df_floor1.date, df_floor1.totals, label="Floor 1")
    plt.legend()

    plt.plot(df_floor2.date, df_floor2.totals, label="Floor 2")
    plt.legend()

    plt.plot(df_floor3.date, df_floor3.totals, label="Floor 3")
    plt.legend()
    plt.show()