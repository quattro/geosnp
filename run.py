__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
import sys
import geosnp


def main(args):
    import argparse as ap
    # build args parser
    parser = ap.ArgumentParser(description="Geographic Inference from SNP data.")
    parser.add_argument("bed_file_prefix", help="The prefix for the bed-file group")
    parser.add_argument("-m", "--bed_map_mode", required=False, default=geosnp.BIM,
                        help="Bed map file mode ['{}' or '{}']".format(geosnp.MAP, geosnp.BIM))
    parser.add_argument("-o", "--output", required=False, default=sys.stdout)

    # grab our arguments and construct the population
    args = parser.parse_args(args)
    population = geosnp.Population.from_bed_files(args.bed_file_prefix, args.bed_map_mode)

    # estimate!
    geosnp.est_loc(population.genotype_matrix)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
