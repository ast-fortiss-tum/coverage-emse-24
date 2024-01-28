import numpy as np
import sys

ARCHIVE_DIST_THRESHOLD = 2

class Archive:

    def __init__(self):
        self.archive = list()
        # self.archived_seeds = set()

    def get_archive(self):
        return self.archive

    def update_archive(self, ind):
        if ind not in self.archive and self.get_min_distance_from_archive(ind) > ARCHIVE_DIST_THRESHOLD:
            self.archive.append(ind)
            # self.archived_seeds.add(ind.seed)
        else:
            print("Individual not added becaue is in archive or is to close to some digit")
      

    def get_min_distance_from_archive(self, seed):
        distances = list()
        distances.append(1000)

        for archived_digit in self.archive:
            # print("Digit or some close digit is in archive.")
            if archived_digit.purified is not seed.purified:
                dist = np.linalg.norm(archived_digit.purified - seed.purified)
                # TODO fix, distance is somehow 0, even when digit not in archive
                # if dist == 0:
                #     print("Distance is 0, skip.")
                #     continue
                distances.append(dist)
        min_dist = min(distances)
        return min_dist