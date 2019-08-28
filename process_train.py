#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from PIL import Image, ImageOps
import pandas as pd
import pytesseract


#  Notes sur les fichiers logs
#
#  La couleur des flèches indique si le train est
#       - arrêté (rouge)
#       - en marche (vert)
#
#  Dates d'arrivée à Yard (destination) ou Mai
#       - estimée (italique bleu)
#       - réelle (noir)
#
#  La distance à destination au nord semble être le mile 224
#


if os.name == 'nt':
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

LINES_RECTANGLES = (
    (703, 385, 1920, 415),
    (703, 415, 1920, 445),
    (703, 445, 1920, 475),
    (703, 475, 1920, 505),
    (703, 505, 1920, 535),
    (703, 535, 1920, 565),
    (703, 565, 1920, 595),
    (703, 595, 1920, 625),
    (703, 625, 1920, 655),
)
ICONS_RECTANGLES = (
    (671, 386, 703, 415),
    (671, 416, 703, 445),
    (671, 446, 703, 475),
    (671, 476, 703, 505),
    (671, 506, 703, 535),
    (671, 536, 703, 565),
    (671, 566, 703, 595),
    (671, 596, 703, 625),
    (671, 626, 703, 655),
)
TRAIN_RECTANGLES = (
    (705, 385, 788, 415),
    (705, 415, 788, 445),
    (705, 445, 788, 475),
    (705, 475, 788, 505),
    (705, 505, 788, 535),
    (705, 535, 788, 565),
    (705, 565, 788, 595),
    (705, 595, 788, 625),
    (705, 625, 788, 655),
)
MILEAGE_MPH_CARS_RECTANGLES = (
    (860, 385, 1128, 415),
    (860, 415, 1128, 445),
    (860, 445, 1128, 475),
    (860, 475, 1128, 505),
    (860, 505, 1128, 535),
    (860, 535, 1128, 565),
    (860, 565, 1128, 595),
    (860, 595, 1128, 625),
    (860, 625, 1128, 655),
)
ARRIVALS_RECTANGLES = (
    (1125, 385, 1618, 415),
    (1125, 415, 1618, 445),
    (1125, 445, 1618, 475),
    (1125, 475, 1618, 505),
    (1125, 505, 1618, 535),
    (1125, 535, 1618, 565),
    (1125, 565, 1618, 595),
    (1125, 595, 1618, 625),
    (1125, 625, 1618, 655),
)
LOG_DATE_RECTANGLE = (1493, 30, 1738, 53)

COLUMNS = ['Train', 'Direction', 'Mileage', 'MPH', 'Cars', 'Arrival Mai',
           'Arrival Yard', 'Log Date']


def get_images():
    images_paths = os.listdir('.')
    images_paths = [image for image in images_paths if 'png' in image]
#    images_paths = [image for image in images_paths if 'TrainCircuit_20190806' in image]
    images = [Image.open(image_path) for image_path in sorted(images_paths)]
    return images


def get_lines(images):
    lines = []
    for image in images:
        for rectangle in LINES_RECTANGLES:
            lines.append(image.crop(rectangle))
    return lines


def get_ref_icons():
    image = Image.open('./TrainCircuit_20190806072539.png')
    icon_up = ImageOps.posterize(image.crop(ICONS_RECTANGLES[0]).convert('L'), 1)
    icon_down = ImageOps.posterize(image.crop(ICONS_RECTANGLES[5]).convert('L'), 1)
    icon_circle = ImageOps.posterize(image.crop(ICONS_RECTANGLES[4]).convert('L'), 1)
    image = Image.open('./TrainCircuit_20190806163547.png')
    icon_down2 = ImageOps.posterize(image.crop(ICONS_RECTANGLES[5]).convert('L'), 1)
    return icon_up, icon_down, icon_down2, icon_circle


def get_icons(images):
    icons = []
    for image in images:
        for rectangle in ICONS_RECTANGLES:
            icons.append(ImageOps.posterize(image.crop(rectangle).convert('L'), 1))
    return icons


def get_images_rectangles(images):
    icons = []
    trains = []
    mmcs = []
    arrivals = []
    log_dates = []
    for image in images:
        log_date = image.crop(LOG_DATE_RECTANGLE)
        for rectangle in ICONS_RECTANGLES:
            icons.append(ImageOps.posterize(image.crop(rectangle).convert('L'), 1))
            log_dates.append(log_date)
        for rectangle in TRAIN_RECTANGLES:
            trains.append(image.crop(rectangle))
        for rectangle in MILEAGE_MPH_CARS_RECTANGLES:
            mmcs.append(image.crop(rectangle))
        for rectangle in ARRIVALS_RECTANGLES:
            arrivals.append(image.crop(rectangle))
    return icons, trains, mmcs, arrivals, log_dates


def get_train_data(images):
    icons, trains, mmcs, arrivals, log_dates = get_images_rectangles(images)
    data = zip(icons, trains, mmcs, arrivals, log_dates)
    icon_up, icon_down, icon_down2, icon_circle = get_ref_icons()
    trains = pd.DataFrame(columns=COLUMNS)
    for icon, train, mmc, arrival, log_date in data:

        name = pytesseract.image_to_string(train)
        if not name or name == 'WK':
            continue
        if name[-1] == '.':
            name = name[:-1]

        logdate = pytesseract.image_to_string(log_date, config='--psm 7')

        info = pytesseract.image_to_string(arrival, config='--psm 7')
        arrivals_re = r'(\d+\s+(.?ul|Aug)\s+\d+:\d+)'
        arrivals = re.findall(arrivals_re, info)
        if len(arrivals) == 0:
            # print(name, 'arrivals: ', info)
            # arrival.show()
            continue
        if len(arrivals) == 2:
            mai = arrivals[0][0]
            yard = arrivals[1][0]
        else:
            mai = None
            yard = arrivals[0][0]

        info = pytesseract.image_to_string(mmc)
        mmc_re = r'((\d+\.\d+)|(\d+))\s+\|?\s*(\d+)\s*\|?\s*(\d+)\s*\(\d+'
        tmp = re.findall(mmc_re, info)
        if len(tmp) == 0:
            # no cars
            mmc_re = r'((\d+\.\d+)|(\d+))\s+\|?\s*(\d+)'
            tmp = re.findall(mmc_re, info)
            if len(tmp) == 0:
                # print('\nmmc: ', name, info)
                # mmc.show()
                continue
            else:
                tmp = tmp[0]
        else:
            tmp = tmp[0]

        if tmp[1]:
            # a float was captured
            mileage = float(tmp[1])
        else:
            # an int was captured (the dot was missed, we must divide by 10)
            mileage = float(tmp[2])/10.0
#            print(name, info)
#            mmc.show()
        speed = int(tmp[3]) if tmp[3] else None
        if len(tmp) == 5:
            cars = int(tmp[4]) if tmp[4] else None
        else:
            cars = None

        if icon == icon_up:
            direction = 'northbound'
        elif icon == icon_down or icon == icon_down2:
            direction = 'southbound'
        else:
            direction = ''

        trains = trains.append(
            pd.DataFrame(
                [[name, direction, mileage, speed, cars, mai, yard, logdate]],
                columns=COLUMNS,
            ),
        )
        trains = trains.dropna(how='all')

    trains = trains.reset_index(drop=True)
    return trains


def get_train_data2(lines, icons):
    icon_up, icon_down, icon_circle = get_ref_icons()
    data = zip(lines, icons)
    trains = pd.DataFrame(columns=COLUMNS)
    for line, icon in data:
        info = pytesseract.image_to_string(line)
        print(info)

        arrivals_re = r'([0-9]{2} (Jul|Aug) [0-9]{2}:[0-9]{2})'
        arrivals = re.findall(arrivals_re, info)
        arrivals = [a[0] for a in arrivals] if len(arrivals) == 2 else [None, None]

        mileage_re = r'([0-9]+\.[0-9]+)'
        mileage = re.findall(mileage_re, info)
        mileage = float(mileage[0]) if mileage else None

        cars_re = r'([0-9]+ \([0-9]+\))'
        cars = re.findall(cars_re, info)
        cars = cars[0] if cars else None

        speed = re.findall(f'{mileage_re}.([0-9]+).{cars_re}', info)
        speed = int(speed[0][1]) if speed else None

        name_re = r'^[^A-z0-9]*([A-z0-9]*)'
        name = re.findall(name_re, info)
        print(name)
        if not name:
            continue
        name = name[0].replace('O', '0').replace('o', '0')

        if icon == icon_up:
            direction = 'northbound'
        elif icon == icon_down:
            direction = 'southbound'
        else:
            direction = ''

        trains = trains.append(
            pd.DataFrame(
                [[name, direction, mileage, speed, cars, *arrivals]],
                columns=COLUMNS,
            ),
        )
        trains = trains.dropna(how='all')

    trains = trains.drop_duplicates(subset='Train')
    trains = trains.reset_index(drop=True)
    return trains


def manual_corrections(train_data):
    train_data.loc[train_data.index == 3, 'Cars'] = 240
    train_data.loc[train_data.index == 12, 'Cars'] = 11
    train_data.loc[train_data.index == 18, ['Mileage', 'MPH', 'Cars']] = 39.9, 41, 50
    train_data.loc[train_data.index == 36, ['Mileage', 'MPH', 'Cars']] = 32.7, 35, 42
    train_data.loc[train_data.index == 39, ['Mileage', 'MPH', 'Cars']] = 186.7, 0, 160
    train_data.loc[train_data.index == 51, ['Mileage', 'MPH', 'Cars']] = 149.1, 0, 42
    train_data.loc[train_data.index == 54, ['Mileage', 'MPH', 'Cars']] = 12.0, 14, None
    train_data.loc[train_data.index == 61, ['Mileage', 'MPH', 'Cars']] = 66.5, 18, 240
    train_data.loc[train_data.index == 80, ['Mileage', 'MPH', 'Cars']] = 155.3, 15, 160
    train_data.loc[train_data.index == 93, 'Cars'] = 163
    train_data.loc[train_data.index == 97, ['Mileage', 'MPH', 'Cars']] = 30.7, 14, 164
    train_data.loc[train_data.index == 100, ['Mileage', 'MPH', 'Cars']] = 77.9, 12, 163
    train_data.loc[train_data.index == 111, ['Mileage', 'MPH', 'Cars']] = 45.2, 35, 240
    train_data.loc[train_data.index == 112, 'Cars'] = 163
    train_data.loc[train_data.index == 113, ['Mileage', 'MPH', 'Cars']] = 156.7, 15, 160
    train_data.loc[train_data.index == 122, ['MPH', 'Cars']] = 0, 163
    new_rows = [
        ['PH0706C', 'southbound', 26.3, 27, 160, '05 Aug 08:35', '06 Aug 08:01', '2019-08-06 07:25:39'],
        ['FCS0120X', 'southbound', 63.7, 26, 108, '06 Aug 18:56', '06 Aug 22:47', '2019-08-06 20:50:50'],
        ['MS0022K', 'southbound', 169.4, 11, 163, '07 Aug 18:05', '07 Aug 23:05', '2019-08-07 16:35:47'],
        ['PH0715S', 'southbound', 113.1, 34, 160, '07 Aug 20:01', '08 Aug 00:33', '2019-08-07 20:50:50'],
        ['PH0716C', 'southbound', 70.3, 11, 160, '08 Aug 00:08', '08 Aug 09:28', '2019-08-08 07:25:39'],
        ['PH0719B', 'southbound', 18.0, 15, 160, '08 Aug 15:47', '08 Aug 21:12', '2019-08-08 20:50:50'],
        ['PH0720A', 'southbound', 13.6, 14, 160, '09 Aug 01:47', '09 Aug 07:29', '2019-08-09 07:25:39'],
        ['PH0722D', 'southbound', 15.7, 0, 160, '09 Aug 06:02', '09 Aug 20:59', '2019-08-09 20:50:50'],
        ['Ph0725R', 'southbound', 55.5, 0, 160, '10 Aug 03:11', '06 Aug 08:51', '2019-08-10 07:25:35'],
        ['PH0726F', 'southbound', 75.0, 25, 160, '10 Aug 14:20', '10 Aug 18:55', '2019-08-10 16:40:43'],
        ['PH0726F', 'southbound', 37.6, 15, 160, '10 Aug 14:20', '10 Aug 21:49', '2019-08-10 20:50:46'],
        ['PH0728P', 'southbound', 78.7, 20, 160, '11 Aug 05:13', '11 Aug 09:48', '2019-08-11 07:25:35'],
        ['A0016A', 'southbound', 154.4, 24, 124, '11 Aug 17:34', '11 Aug 22:40', '2019-08-11 16:40:42'],
        ['MS0025K', 'southbound', 130.3, 15, 163, '11 Aug 20:53', '12 Aug 02:04', '2019-08-12 20:55:46'],
    ]
    for i, row in enumerate(new_rows):
            columns = {
                'Train': row[0],
                'Direction': row[1],
                'Mileage': row[2],
                'MPH': row[3],
                'Cars': row[4],
                'Arrival Mai': row[5],
                'Arrival Yard': row[6],
                'Log Date': row[7],
            }
            new_rows[i] = columns
    new_rows = pd.DataFrame(new_rows, columns=COLUMNS)

    return train_data.append(new_rows, ignore_index=True)


def main():
    images = get_images()
    train_data = get_train_data(images)

    train_data['Train'] = train_data['Train'].str.upper()
    train_data['Train'] = train_data['Train'].str.replace('O', '0')
    train_data['Train'] = train_data['Train'].str.replace('I', '1')
    train_data = manual_corrections(train_data)

    with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
            ):
        print(train_data)

    train_data.to_pickle('train_data.pkl')


if __name__ == '__main__':
    main()
