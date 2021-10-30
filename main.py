import os
import json
import urllib.request
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    df = pd.read_csv(
        "../venezia-project-ssd/data/Museo M9/totals_per_area.csv")
    timestamp = df.timestamp
    areaCode = df.area_code
    totals = df.totals
    alarms = df.alarms
    date = df.date

    print(timestamp)
    print(areaCode)
    print(totals)
    print(alarms)
    print(date)
    print("fine")
