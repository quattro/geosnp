__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"


class Individual(object):

    def __init__(self):
        self.family_id
        self.individual_id
        self.paternal_id
        self.maternal_id
        self.sex
        self.phenotype


class SNPInfo(object):

    def __init__(self):
        self.chromosome
        self.snp_id
        self.morgan
        self.position
        self.snp_minor
        self.snp_major


class Population(object):

    def __init__(self):
        self.population = list()
        self.snp_info = list()

    def __getitem__(self, item):
        return self.population[item]

    def __len__(self):
        return len(self.population)

    def num_snps(self):
        return len(self.snp_info)

    def __iter__(self):
        return iter(self.population)

    def next(self):
        return next(self.population)

    def iter_snps(self):
        return iter(self.snp_info)

    def next_snp(self):
        return next(self.snp_info)

    @classmethod
    def from_genotype_file(cls, file):
        pop = cls()
        return pop
