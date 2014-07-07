#! /usr/bin/env python2.7
__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
import sys
import geosnp

from os import linesep


def main(args):
    import argparse as ap
    # build args parser
    parser = ap.ArgumentParser(description="Geographic Inference from SNP data.")
    parser.add_argument("bed_file_prefix", help="The prefix for the bed-file group")
    parser.add_argument("-m", "--bed-map-mode", required=False, default=geosnp.BIM,
                        help="Bed map file mode ['{}' or '{}']".format(geosnp.MAP, geosnp.BIM))
    parser.add_argument("-k", "--dim", required=False, type=int, default=2,
                        help="Dimensionality of geographic inference (2 or 3)")
    parser.add_argument("-l", "--loc-output", required=False, default=sys.stdout,
                        help="Output for the locations.")
    parser.add_argument("-c", "--cof-output", required=False, default=sys.stdout,
                        help="Output for the model coefficients.")

    # grab our arguments and construct the population
    args = parser.parse_args(args)

    if args.dim != 2 and args.dim != 3:
        parser.print_usage()
        return 1

    population = geosnp.Population.from_bed_files(args.bed_file_prefix, args.bed_map_mode)

    # estimate!
    print 'estimating'
    import pdb; pdb.set_trace()
    try:
        X, Y = geosnp.est_loc(population, k=args.dim)
        for idx, row in enumerate(X):
            person = population[idx]
            args.loc_output.write(str(person))
            line = "\t".join(row) + linesep
            args.loc_output.write(line)

        for idx, row in enumerate(Y):
            snp = population.snp_info[idx]
            args.cof_output.write(str(snp) + linesep)
            line = "\t".join(row) + linesep
            args.cof_output.write(line)

    except Exception as e:
        import traceback as tb
        tb.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
