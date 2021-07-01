from os.path import join, exists

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from h5py import File
from inflection import underscore


class Catalog(list):
    dir = 'figures'

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                idx = self.filenames.index(idx)
            except ValueError:
                raise ValueError(f"Figure `{idx}` is not registered.")
        return super().__getitem__(idx)

    @property
    def filenames(self):
        return [figure.filename.split('.')[0] for figure in self]

    def register(self, figure):
        if type(figure) is type:
            figure = figure()
        figure.reldir = join(self.dir, figure.reldir)
        self.append(figure)

    def draw_all(self, show=True):
        for figure in self:
            figure.generate()
            figure.save(show=show)

    def regenerate(self, idx):
        figure = self[idx]
        metadata = figure.Metadata(figure.filepath)
        metadata.generate()

    def regenerate_all(self):
        for i in range(self):
            self.regenerate(i)


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

    def save(self, show=True):
        plt.savefig(self.filepath)
        if show:
            plt.show()
        plt.close()

    def plot(self, data):
        raise NotImplementedError


catalog = Catalog()
