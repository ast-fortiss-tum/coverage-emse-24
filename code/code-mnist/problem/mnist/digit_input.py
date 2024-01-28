from problem.mnist import rasterization_tools
from problem.mnist.timer import Timer
from problem.mnist.properties import RUN, FEATURES,IMG_SIZE
from problem.mnist.folder import Folder
import json
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


class Digit:
    COUNT = 0

    def __init__(self, desc, label, seed):
        self.timestamp, self.elapsed_time = Timer.get_timestamps()
        self.id = Digit.COUNT
        self.run = RUN
        self.seed = seed
        # TODO
        self.features = FEATURES
        self.tool = "DeepHyperion"
        self.xml_desc = desc
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.expected_label = label
        self.predicted_label = None
        self.confidence = None
        self.rank = np.inf
        self.selected_counter = 0
        self.placed_mutant = 0
        Digit.COUNT += 1

    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': self.is_misbehavior(),
                'performance': str(self.confidence),
                'timestamp': str(self.timestamp),
                'elapsed': str(self.elapsed_time),
                'tool' : str(self.tool),
                'run' : str(self.run),
                'features': self.features,
                'rank': str(self.rank),
                'selected': str(self.selected_counter),
                'placed_mutant': str(self.placed_mutant)
        }

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_png(self, filename):
        plt.imsave(filename+'.png', self.purified.reshape(28, 28), cmap='gray', format='png')

    def save_npy(self, filename):
        np.save(filename, self.purified)
        test_img = np.load(filename+'.npy')
        diff = self.purified - test_img
        assert(np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def is_misbehavior(self):
        if self.expected_label == self.predicted_label:
            return False
        else:
            return True

    def export_relative(self, parent_folder, all=False):
        DST_ALL, DST_ARC = Folder.create_folder(parent_folder)
        if all:
            dst = join(DST_ALL, "mbr"+str(self.id))
        else:
            dst = join(DST_ARC, "mbr"+str(self.id))
        self.dump(dst)
        self.save_npy(dst)
        # self.save_png(dst)
        self.save_svg(dst)

    def export(self, all=False):
        if all:
            dst = join(Folder.DST_ALL, "mbr"+str(self.id))
        else:
            dst = join(Folder.DST_ARC, "mbr"+str(self.id))
        self.dump(dst)
        self.save_npy(dst)
        # self.save_png(dst)
        self.save_svg(dst)

    def clone(self):
        clone_digit = Digit(self.xml_desc, self.expected_label, self.seed)
        return clone_digit

    def distance(self, other):
        return np.linalg.norm(self.purified - other.purified)

    ''' Measure how much space is covered by digit (in image) in percent'''
    def coverage(self, min_saturation=0.2, relative=False):
        counter = 0
        counter_digit = 0
        for c in self.purified.flatten():
            if c > min_saturation:
                counter +=1
            if c!=0:
                counter_digit += 1
        if relative:
            total = counter_digit
        else:
            total = IMG_SIZE*IMG_SIZE
        return counter/total

    ''' Measure how bright the digits are in average'''
    def brightness(self, min_saturation=0):
        # TH_BRIGHT = 0.3
        counter = 0
        sum_bright = 0
        for c in self.purified.flatten():
            if c > min_saturation:
                counter +=1
                sum_bright += c
        return sum_bright/counter