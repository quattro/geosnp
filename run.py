#! /usr/bin/env python2.7
__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
import logging
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
    parser.add_argument("--loc-input", required=False, type=ap.FileType("r"),
                        help="Input location file. GeoSNP will only estimate coefficients when this is supplied.")
    parser.add_argument("--cof-input", required=False, type=ap.FileType("r"),
                        help="Input coefficient file. GeoSNP will only estimate locations when this is supplied.")
    parser.add_argument("-l", "--loc-output", required=False, default=sys.stdout,
                        help="Output for the locations.")
    parser.add_argument("-c", "--cof-output", required=False, default=sys.stdout,
                        help="Output for the model coefficients.")

    # Set up the log.
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()])

    # grab our arguments and construct the population
    args = parser.parse_args(args)

    if args.dim != 2 and args.dim != 3:
        parser.print_usage()
        return 1

    population = geosnp.Population.from_bed_files(args.bed_file_prefix, args.bed_map_mode)
    X = Y = None
    if args.loc_input is not None:
        X = geosnp.parse_locations(args.loc_input)
    if args.coff_input is not None:
        Y = geosnp.parse_coefficients(args.loc_input)

    # estimate!
    logging.info("Estimating...")
    try:
        Z, Y = geosnp.est_loc(population, X, Y, k=args.dim)
        k = args.dim
        for idx, row in enumerate(Z):
            row = row[k**2:k**2 + k]
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
        logging.error(str(e))
        tb.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
