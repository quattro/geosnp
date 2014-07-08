__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
__all__ = ["parse_locations",
           "parse_coefficients"]

import geosnp
import numpy


def parse_locations(stream, n, k=2):
    X = numpy.rvs(size=[n, k])

    for idx, row in enumerate(stream):
        values = row.split()
        # skip these for now
        #snp = geosnp.SNPInfo(values[0], values[1], values[2], int(values[3]))
        locations = map(float, values[3:])
        if len(locations) != k:
            raise ValueError("Invalid number of locations in location file!")

        X[idx] = locations

    return X


def parse_coefficients(stream, n, l):
    pass
