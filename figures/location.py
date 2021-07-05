# -*- coding: utf-8 -*-

from os.path import join
import xml.etree.ElementTree as ET

import pyproj
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

from catalog import catalog, Metadata, Figure


class Location_(Metadata):
    def generate(self):
        coords = get_raw_coordinates()
        coords = filter_keys(coords)

        railway = coords['Voie ferrée']
        geophones = np.concatenate(
            [coord for key, coord in coords.items() if 'G' in key]
        )
        hydrophones = np.concatenate(
            [coord for key, coord in coords.items() if 'H' in key]
        )
        distances = closest_distances(
            np.concatenate([geophones, hydrophones]),
            railway,
        )
        satellite = plt.imread(join(catalog.dir, "satellite.png"))

        self["railway"] = railway
        self["geophones"] = geophones
        self["hydrophones"] = hydrophones
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

    def plot(self, data):
        railway = data["railway"]
        geophones = data["geophones"]
        # hydrophones = data["hydrophones"]
        distances = data["distances"]
        satellite = data["satellite"]

        print("DISTANCES SITE FROM RAILWAY")
        for start, end in [[0, 24], [24, 48], [48, 72]]:
            print(np.mean(distances[start:end]))

        plt.plot(
            *railway.T[:2],
            c='w',
            label="Railway",
        )
        plt.scatter(
            *geophones.T[:2],
            s=.5,
            c='tab:orange',
            label="Geophones",
        )
        # plt.scatter(*hydrophones.T[:2], s=.5, c='tab:blue')
        extent = [*plt.xlim(), *plt.ylim()]
        plt.imshow(satellite, zorder=-1, extent=extent)
        plt.gca().set_aspect('equal')
        for ticks in [plt.xticks, plt.yticks]:
            loc, _ = ticks()
            ticks(loc[1::2])
        plt.xlabel("Easting (UTM)")
        plt.ylabel("Northing (UTM)")
        plt.legend(loc='lower right')


catalog.register(Location)
