__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
import logging
import numpy as np

# Some useful constants
# Number of columns for file format
MAP_LEN = 4
BIM_LEN = 6
FAM_LEN = 6

# File name consts
MAP = 'map'
BIM = 'bim'

# Encoding const
GENOTYPE_PER_BYTE = 4

# Magic numbers for bed encoding
HEADER1 = 0x6C
HEADER2 = 0x1B

# Bed data ordering constants
IND_MAJOR = 0
SNP_MAJOR = 1

# Plink SNP value constants
PLINK_MAJOR = 0x00
PLINK_MINOR = 0x03
# these next two are technically flipped, but only because the bytes
# are encoded in reverse order
PLINK_HETERO = 0x02
PLINK_MISSING = 0x01

MISSING = -1
HOMO_MAJOR = 0
HOMO_MINOR = 2
HETERO = 1


class Individual(object):

    def __init__(self, fid, iid, pid, mid, sex, phen):
        self.family_id = fid
        self.individual_id = iid
        self.paternal_id = pid
        self.maternal_id = mid
        self.sex = sex
        self.phenotype = phen

    def __str__(self):
        return "\t".join([self.family_id, self.individual_id, self.paternal_id,
                          self.maternal_id, self.sex, self.phenotype])


class SNPInfo(object):

    def __init__(self, chrsme, snpid, morgan, pos):
        self.chromosome = chrsme
        self.id = snpid
        self.morgan = morgan
        self.position = pos
        self.minor = None
        self.major = None

    def __str__(self):
        return "\t".join([self.chromosome, self.id, self.morgan, str(self.position),
                          self.minor, self.major])


class Population(object):

    def __init__(self):
        self.individuals = list()
        self.snp_info = list()
        self.genotype_matrix = None

    def __getitem__(self, item):
        return self.individuals[item]

    def __len__(self):
        return len(self.individuals)

    def num_snps(self):
        return len(self.snp_info)

    def __iter__(self):
        return iter(self.individuals)

    def next(self):
        return next(self.individuals)

    def iter_snps(self):
        return iter(self.snp_info)

    def next_snp(self):
        return next(self.snp_info)

    def _read_bim_file(self, filename, mtype=BIM):
        with open(filename, "r") as bim_file:
            for row in bim_file:
                values = row.split()
                if len(values) != MAP_LEN and len(values) != BIM_LEN:
                    raise ValueError("Invalid map/bim file!")

                snp = SNPInfo(values[0], values[1], values[2], int(values[3]))
                if mtype == BIM:
                    snp.minor = values[4]
                    snp.major = values[5]

                self.snp_info.append(snp)
        return

    def _read_fam_file(self, filename):
        with open(filename, "r") as fam_file:
            for row in fam_file:
                values = row.split()
                if len(values) != FAM_LEN:
                    raise ValueError("Invalid fam file!")
                ind = Individual(*values)
                self.individuals.append(ind)
        return

    def _flip_snp(self):
        counts = dict([(HOMO_MAJOR, 0), (HOMO_MINOR, 0), (HETERO, 0), (MISSING, 0)])
        for sidx in range(self.num_snps()):
            for iidx, indv in enumerate(self.individuals):
                snp = self.genotype_matrix[iidx, sidx]
                counts[snp] += 1

            # we need to flip
            if counts[HOMO_MINOR] > counts[HOMO_MAJOR]:
                for iidx, indv in enumerate(self.individuals):
                    snp = self.genotype_matrix[iidx, sidx]
                    if snp == HOMO_MAJOR:
                        self.genotype_matrix[iidx, sidx] = HOMO_MINOR
                    elif snp == HOMO_MINOR:
                        self.genotype_matrix[iidx, sidx] = HOMO_MAJOR
                # swap the SNP info
                minor = self.snp_info[sidx].major
                self.snp_info[sidx].major = self.snp_info[sidx].minor
                self.snp_info[sidx].minor = minor

            # reset counts
            counts[HOMO_MAJOR] = 0
            counts[HOMO_MINOR] = 0
            counts[HETERO] = 0
            counts[MISSING] = 0

        return

    @classmethod
    def from_bed_files(cls, filename_prefix, map_type=BIM):
        pop = cls()

        # read the bim file
        logging.info('Reading bim file.')
        pop._read_bim_file(filename_prefix + ".bim", map_type)

        # read the fam file
        logging.info('Reading fam file.')
        pop._read_fam_file(filename_prefix + ".fam")

        n, m = len(pop), pop.num_snps()
        pop.genotype_matrix = np.zeros((n, m), dtype=np.int8)

        # read the bed file
        logging.info("{0} people and {1} SNPs.".format(n, m))
        logging.info('Reading bed file.')
        with open(filename_prefix + ".bed", "rb") as bed_file:
            # check the header for magic number
            header1 = ord(bed_file.read(1))
            header2 = ord(bed_file.read(1))
            if header1 != HEADER1 and header2 != HEADER2:
                raise ValueError("Illegal bed file!")

            # grab the data ordering info
            mode = ord(bed_file.read(1))

            # bed is in snp major mode, ie, 1st snp for all indv, 2nd snp for all indv...
            if mode == SNP_MAJOR:
                snp_block_size = n / GENOTYPE_PER_BYTE
                snp_residual_chunk_size = n % GENOTYPE_PER_BYTE

                for sidx in range(m):
                    # read in all the SNPs for the main block
                    block = bed_file.read(snp_block_size)
                    pidx = 0
                    for byte in block:
                        # just unroll the loop...
                        byte = ord(byte)
                        pop.genotype_matrix[pidx, sidx] = read_genotype(byte, 0)
                        pidx += 1
                        pop.genotype_matrix[pidx, sidx] = read_genotype(byte, 1)
                        pidx += 1
                        pop.genotype_matrix[pidx, sidx] = read_genotype(byte, 2)
                        pidx += 1
                        pop.genotype_matrix[pidx, sidx] = read_genotype(byte, 3)
                        pidx += 1

                    # if necessary read in the extra byte for the residual chunk
                    if snp_residual_chunk_size > 0:
                        byte = ord(bed_file.read(1))
                        for pos in range(snp_residual_chunk_size):
                            pop.genotype_matrix[pidx, sidx] = read_genotype(byte, pos)
                            pidx += 1
            # bed is in individual major mode, all snps for 1st indv, all snps for 2nd indv...
            elif mode == IND_MAJOR:
                # do this later. POPRES data is in SNP-Major mode
                pass
            else:
                raise ValueError("Bad bed mode!")

        # if the major and minor allels are reversed, flip them
        logging.info('Checking SNP encodings.')
        pop._flip_snp()
        return pop


def read_genotype(item, pos):
    mask = [0x03, 0x0C, 0x30, 0xC0]
    # mask off the region we want, and shift
    value = (item & mask[pos]) >> (pos * 2)
    translation = dict([(PLINK_MAJOR, HOMO_MAJOR), (PLINK_MINOR, HOMO_MINOR),
                        (PLINK_HETERO, HETERO), (PLINK_MISSING, MISSING)])
    return translation[value]
