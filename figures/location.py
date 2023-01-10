# -*- coding: utf-8 -*-

from os.path import join
import xml.etree.ElementTree as ET

import pyproj
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import proplot as pplt

from catalog import catalog, Metadata, Figure

RIVER = np.array(
    [
        [
            6, 8, 9.1, 9.8, 10.1, 10.3, 10.4, 10.5, 10.8, 11.3, 12,
            13.8, 15.7, 16, 16.3, 16.6, 16.8, 17.4, 18.5, 19.5, 20.2, 20.8,
            21.1, 21.5, 22.1, 21.7, 18.5, 18, 18.5, 19, 20.3, 21, 21.2,
            21, 20.5, 19.2, 27.5, 28.2, 29, 29.5,
        ],
        np.arange(22, 62, 1),
    ]
)
RIVER = np.insert(RIVER, -4, [20, 58], axis=1)
RIVER = np.insert(RIVER, -4, [26.5, 57.5], axis=1)
RIVER = np.insert(RIVER, 26, [21.5, 47.5], axis=1)
RIVER += [[7050], [55950]]
RIVER *= 100


class Location_(Metadata):
    def generate(self):
        coords = get_raw_coordinates()
        coords = filter_keys(coords)

        railway = coords['Voie ferrée']
        geophones = np.concatenate(
            [coord for key, coord in coords.items() if 'G' in key]
        )
        distances = closest_distances(geophones, railway)
        satellite = plt.imread(join(catalog.dir, "satellite.png"))

        self["railway"] = railway
        self["geophones"] = geophones
        self["distances"] = distances
        self["satellite"] = satellite


def get_raw_coordinates():
    utm = pyproj.Proj('+init=EPSG:32619')
    wgs84 = pyproj.Proj('+init=EPSG:4326')

    tree = ET.parse('Nipissis.kml')
    prefix = './/{http://www.opengis.net/kml/2.2}'
    placemarks = tree.findall(prefix + 'Placemark')

    coords = {}

    for attributes in placemarks:
        name = attributes.find(prefix + 'name')
        if 'Voie ferrée' in name.text:
            ls = attributes.find(prefix + 'LineString')
            coord = ls.find(prefix + 'coordinates')
            tmp = coord.text.replace('\n', '').replace('\t', '')
            tmp = tmp.split(' ')
            if '' in tmp:
                tmp.remove('')

            coords[name.text] = np.empty([len(tmp), 3])
            for i, t in enumerate(tmp):
                lon, lat, z = t.split(',')
                x, y, z = pyproj.transform(wgs84, utm, lon, lat, z)
                coords[name.text][i] = x, y, z
        else:
            pt = attributes.find(prefix + 'Point')
            coord = pt.find(prefix + 'coordinates')
            lon, lat, z = coord.text.split(',')
            x, y, z = pyproj.transform(wgs84, utm, lon, lat, z)
            coords[name.text] = x, y, z

    return coords


def filter_keys(coords):
    tags = [tag[:4] for tag in coords.keys() if tag != 'Voie ferrée']
    for tag in np.unique(tags):
        keys = [key for key in coords.keys() if tag in key]
        keys.sort()
        coords[tag] = np.empty([len(keys), 3])
        for i, key in enumerate(keys):
            coords[tag][i] = coords[key]
            del coords[key]

    return coords


def distance_point_to_line(point, line):
    p3 = point
    p1, p2 = line
    return norm(np.cross(p2-p1, p1-p3)) / norm(p2-p1)


def closest_distances(points, lines):
    distances = np.full(len(points), np.inf)
    for i, p3 in enumerate(points):
        for p1, p2 in zip(lines[:-1], lines[1:]):
            # https://math.stackexchange.com/q/2250212
            t = -np.dot(p1-p3, p2-p1) / norm(p2-p1)**2
            if t >= 0 and t <= 1:
                distance = distance_point_to_line(p3, (p1, p2))
            else:
                d1 = norm(p1-p3)
                d2 = norm(p2-p3)
                distance = d1 if d1 < d2 else d2
            if distance < distances[i]:
                distances[i] = distance

    return distances


class Location(Figure):
    Metadata = Location_

    def plot(self, data, show_river=False):
        railway = data["railway"]
        geophones = data["geophones"]
        distances = data["distances"]
        satellite = data["satellite"]

        print("DISTANCES SITE FROM RAILWAY")
        for start, end in [[0, 24], [24, 48], [48, 72]]:
            print(np.mean(distances[start:end]))

        _, ax = pplt.subplots(figsize=[3.33, 5])
        ax.plot(
            *railway.T[:2],
            c='w',
            label="Railway",
        )
        if show_river:
            ax.plot(
                *RIVER,
                c='cyan',
                label="River",
                zorder=100,
            )
        ax.scatter(
            *geophones.T[:2],
            s=.5,
            c='orange',
            label="Geophones",
        )
        satellite = satellite[10:-50, 38:-24]
        xlim = [705800, 708000]
        ylim = [5597200, 5601100]
        ax.imshow(satellite, zorder=-1, extent=[*xlim, *ylim])
        ax.set_aspect('equal')
        for ticks in [plt.xticks, plt.yticks]:
            loc, _ = ticks()
            ticks(loc[1::2].astype(int))
        ax.format(
            xlabel="Easting (UTM)",
            ylabel="Northing (UTM)",
            xlim=xlim,
            ylim=ylim,
            xformatter='{x:d}',
            yformatter='{x:d}',
        )
        ax.legend(loc='lower right', ncol=1)


catalog.register(Location)
