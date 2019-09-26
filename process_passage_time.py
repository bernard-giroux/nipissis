#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import re
import numpy as np
import pandas as pd

MILEAGE_SITES = (27.4, 29.9, 30.4)
MILEAGE_MAI = 128.1

COLUMNS = ['Train', 'Direction', 'MPH', 'Cars', 'Site 1', 'Site 2', 'Site 3']


trains = pd.read_pickle('./train_data.pkl')

# create datetime objects for dates stored in strings
for n in range(trains.shape[0]):
    text = trains.loc[n, 'Arrival Mai']
    if text is not None:
        date_re = r'(\d+) (.+) (\d+):(\d+)'
        tmp = re.findall(date_re, text)
        day = int(tmp[0][0])
        if 'ul' in tmp[0][1]:
            month = 7
        else:
            month = 8
        hour = int(tmp[0][2])
        minute = int(tmp[0][3])
        trains.loc[n, 'Arrival Mai'] = datetime.datetime(2019, month, day,
                                                         hour, minute)

    text = trains.loc[n, 'Arrival Yard']
    if text is not None:
        date_re = r'(\d+) (.+) (\d+):(\d+)'
        tmp = re.findall(date_re, text)
        day = int(tmp[0][0])
        if 'ul' in tmp[0][1]:
            month = 7
        else:
            month = 8
        hour = int(tmp[0][2])
        minute = int(tmp[0][3])
        trains.loc[n, 'Arrival Yard'] = datetime.datetime(2019, month, day,
                                                          hour, minute)

    text = trains.loc[n, 'Log Date']
    if text is not None:
        date_re = r'(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)'
        tmp = re.findall(date_re, text)
        day = int(tmp[0][2])
        hour = int(tmp[0][3])
        trains.loc[n, 'Log Date'] = datetime.datetime(2019, 8, day,
                                                      hour, minute)


# %%
#
# Passage times at Mai that are earlier than log date are actual time (not
#  estimated)
#
# Multiple entries for the same train:
#        We should use entrye with mileage closest to the sites
#        But before we must make sure it is the same trip!
#

def get_passage_times(entry):
    sites = [None, None, None]
    direction = entry['Direction']
    cars = entry['Cars']
    speed = entry['MPH']
    # we want to store the speed value found in the logs, but we need a
    # value to compute dt
    v = speed if speed else 35
    # TODO: instead of using 35 we should probably compute the average
    #       speed for trains with same number of cars
    if entry['Direction'] == 'southbound':
        if entry['Arrival Mai'] is not None:
            if entry['Log Date'] > entry['Arrival Mai']:
                # we passed Mai, we use actual mileage
                for n in range(3):
                    dist = entry['Mileage'] - MILEAGE_SITES[n]
                    dt = dist / v
                    # mileage positif, on n'est pas encore passé
                    sites[n] = entry['Log Date'] + datetime.timedelta(hours=dt)
            else:
                # perhaps more accurate to use estimated passage time at Mai
                for n in range(3):
                    dist = MILEAGE_MAI - MILEAGE_SITES[n]
                    dt = dist / v
                    sites[n] = entry['Arrival Mai'] + datetime.timedelta(hours=dt)
        else:
            for n in range(3):
                dist = MILEAGE_SITES[n]
                dt = dist / v
                sites[n] = entry['Arrival Yard'] - datetime.timedelta(hours=dt)
    else:  # northbound
        if entry['Arrival Mai'] is not None and entry['Mileage'] > MILEAGE_MAI:
            # on a dépassé Mai, on utilise l'heure de passage à Mai
            for n in range(3):
                dist = MILEAGE_MAI - MILEAGE_SITES[n]
                dt = dist / v
                sites[n] = entry['Arrival Mai'] - datetime.timedelta(hours=dt)
        else:
            for n in range(3):
                dist = entry['Mileage'] - MILEAGE_SITES[n]
                dt = dist / v
                sites[n] = entry['Log Date'] - datetime.timedelta(hours=dt)
    return direction, cars, speed, sites


passage_times = pd.DataFrame(columns=COLUMNS)

for name in np.unique(trains.loc[:, 'Train']):
    if name == 'MN0023N':
        # this train does not move from mileage 224
        continue

    entries = trains[trains['Train'] == name]
    entries = entries.reset_index()

    if len(entries) > 1:
        ind = np.argmin(np.abs(entries['Mileage'].values - MILEAGE_SITES[0]))
        direction, cars, speed, sites = get_passage_times(entries.loc[ind, :])
    else:
        direction, cars, speed, sites = get_passage_times(entries.loc[0, :])

    passage_times = passage_times.append(
        pd.DataFrame(
            [[name, direction, speed, cars, sites[0], sites[1], sites[2]]],
            columns=COLUMNS,
        ),
    )

passage_times = passage_times.reset_index(drop=True)

with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None
        ):
    print(passage_times)

passage_times.to_pickle('passage_times.pkl')
