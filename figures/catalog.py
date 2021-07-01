from os.path import join, curdir, exists

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from h5py import File


class Catalog(list):
    def __init__(self, *args, figures_dir=curdir):
        self.figures_dir = figures_dir
        super().__init__(*args)

    def register(self, figure):
        figure.reldir = join(self.figures_dir, figure.reldir)
        self.append(figure)

    def draw_all(self):
        for figure in self:
            figure.draw()
            figure.save(figures_dir=self.figures_dir)


class Metadata(File):
    def __init__(self, path, *args, **kwargs):
        path = '.'.join(path.split('.')[:-1])  # Remove extension.
        self.path = path + '.meta'
        if not exists(self.path):
            self.generate()
        super().__init__(path, 'r+', *args, **kwargs)

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

    def __init__(self, metadata, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def filename(self):
        return type(self).__name__ + '.pdf'
        # return catalog.index(self)

    @property
    def filepath(self):
        return join(self.reldir, self.filename)

    def draw(self, *args, **kwargs):
        with self.Metadata(self.filepath) as data:
            self.plot(data)
            super().draw(*args, **kwargs)

    def save(self):
        plt.savefig(self.filepath)
        plt.show()
        plt.close()

    def plot(self, data):
        raise NotImplementedError


catalog = Catalog()
