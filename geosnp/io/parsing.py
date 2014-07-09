__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
__all__ = ["parse_locations",
           "parse_coefficients"]

from scipy import stats


def parse_locations(stream, n, k=2):
    X = stats.norm.rvs(size=[n, k])

    for idx, row in enumerate(stream):
        values = row.split()
        # skip these for now
        #snp = geosnp.Individual(*values[:6])
        locations = map(float, values[6:])
        if len(locations) != k:
            raise ValueError("Invalid number of locations in location file!")

        X[idx] = locations

    return X


def parse_coefficients(stream, n, l):
    pass
