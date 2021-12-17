from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pymongo import MongoClient


def connect_to_db(username: str = "admin", password: str = "admin", host: str = "localhost", port: int = 27017):
    connection_string = f"mongodb://{username}:{password}@{host}:{port}/?" \
                        f"readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"

    client = MongoClient(connection_string)

    # db = client.admin
    # serverStatusResult = db.command("serverStatus")
    # pprint(serverStatusResult)

    return client


def get_area_data(df):
    date = []

    for t in df.timestamp:
        t = int(t)
        ts = int(t / 1000)
        time_value = datetime.fromtimestamp(ts)
        date.append(time_value)

    return sorted(date)


if __name__ == '__main__':
    # Database connection
    mongo_client = connect_to_db()
    db_name = mongo_client.get_database("venezia-project")
    total_per_area_collection = db_name["totals-per-area"]

    # Find by area code
    cursors_floor_1 = total_per_area_collection.find({"area_code": "floor_1"})
    cursors_floor_2 = total_per_area_collection.find({"area_code": "floor_2"})
    cursors_floor_3 = total_per_area_collection.find({"area_code": "floor_3"})
    cursors_front_desk = total_per_area_collection.find({"area_code": "front_desk"})

    # Convert to dataframe
    df_floor1 = pd.DataFrame(list(cursors_floor_1))
    df_floor1["totals"] = df_floor1["totals"].astype("int32")

    df_floor2 = pd.DataFrame(list(cursors_floor_2))
    df_floor2["totals"] = df_floor2["totals"].astype("int32")

    df_floor3 = pd.DataFrame(list(cursors_floor_3))
    df_floor3["totals"] = df_floor3["totals"].astype("int32")

    df_front_desk = pd.DataFrame(list(cursors_front_desk))
    df_front_desk["totals"] = df_front_desk["totals"].astype("int32")

    date_val_floor_1 = get_area_data(df_floor1)
    date_val_floor_2 = get_area_data(df_floor2)
    date_val_floor_3 = get_area_data(df_floor3)
    date_val_front_desk = get_area_data(df_front_desk)

    # Plot
    figure(figsize=(15, 8))
    plt.subplot(3, 2, 1)
    plt.plot(date_val_floor_1, df_floor1.totals, color="green", label="Floor_1")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(date_val_floor_2, df_floor2.totals, color="red", label="Floor_2")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(date_val_floor_3, df_floor3.totals, color="blue", label="Floor_3")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(date_val_front_desk, df_front_desk.totals, color="black", label="Front_desk")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(date_val_floor_1, df_floor1.totals, label="Floor 1")
    plt.legend()

    plt.plot(date_val_floor_2, df_floor2.totals, label="Floor 2")
    plt.legend()

    plt.plot(date_val_floor_3, df_floor3.totals, label="Floor 3")
    plt.legend()

    plt.plot(date_val_front_desk, df_front_desk.totals, label="Front_desk")
    plt.legend()
    plt.show()