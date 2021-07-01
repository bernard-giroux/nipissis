from os.path import join, curdir, exists

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from h5py import File
from inflection import underscore


class Catalog(list):
    dir = 'figures'

    def register(self, figure):
        if type(figure) is type:
            figure = figure()
        figure.reldir = join(self.dir, figure.reldir)
        self.append(figure)

    def draw_all(self):
        for figure in self:
            figure.generate()
            figure.save()


class Metadata(File):
    def __init__(self, path, *args, **kwargs):
        path = '.'.join(path.split('.')[:-1])  # Remove extension.
        path = path + '.meta'
        is_not_generated = not exists(path)
        super().__init__(path, 'a', *args, **kwargs)
        if is_not_generated:
            self.generate()

    def generate(self):
        raise NotImplementedError

    def __setitem__(self, key, value):
        try:
            del self[key]
        except KeyError:
            pass
        super().__setitem__(key, value)


class Figure(Figure):
    Metadata = Metadata
    reldir = ""

    @property
    def filename(self):
        return underscore(type(self).__name__) + '.pdf'
        # return catalog.index(self)

    @property
    def filepath(self):
        return join(self.reldir, self.filename)

    def generate(self, *args, **kwargs):
        with self.Metadata(self.filepath) as data:
            self.plot(data)

    def save(self):
        plt.savefig(self.filepath)
        plt.show()
        plt.close()

    def plot(self, data):
        raise NotImplementedError


catalog = Catalog()
