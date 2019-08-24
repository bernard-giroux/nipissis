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
            print(name, 'arrivals: ', info)
            arrival.show()
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
                print('\nmmc: ', name, info)
                mmc.show()
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


def main():
    images = get_images()
    train_data = get_train_data(images)

    with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None
            ):
        print(train_data)
    train_data.to_pickle('train_data.pkl')


if __name__ == '__main__':
    main()
