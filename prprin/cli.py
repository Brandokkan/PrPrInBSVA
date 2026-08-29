"""Command line interface of the prprin package

This module turns the protein-protein interaction analysis into a command line
program, so that every parameter is given by the user when the program is run
and nothing has to be written inside the source code.

It is meant to be called through the 'run_prprin.py' script of the project, or
with 'python -m prprin', and it never asks anything interactively, so that it
can be used inside automated pipelines.
"""

import argparse
import sys
from pathlib import Path

from . import PrPrInObject
from .enrich import POSSIBLE_ENRICHMENT_BACKGROUND_STR
from .hubs import HUB_METHODS
from .metrics import SCORE_NAMES
from .retrive import read_identifiers_file, resolve_input_path

DEFAULT_OUTDIR = "prprin_results"
# the analysis writes these tables beside the plots when --save-tables is given
SAVED_TABLES = ("network", "node_data", "enrichment")


def build_parser():
    """Build the command line parser of the program

    Returns
    -------
    argparse.ArgumentParser
        The parser describing every accepted command line parameter
    """
    parser = argparse.ArgumentParser(
        prog="prprin",
        description="Build and analyse a protein-protein interaction network from the STRING database: "
                    "retrieve the interactions, compute the network metrics, find the hub proteins, "
                    "run the functional enrichment of the hubs and write the plots of the results.",
        epilog="example: %(prog)s data/test_big --hub-method consensus topk --top-k 5 "
               "--background genome --outdir results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "proteins",
        help="the proteins of interest: either their identifiers (gene symbols or stringIds) "
             "separated by spaces and quoted, or the path of a file containing one identifier per line",
    )

    # interaction retrieval
    retrieval = parser.add_argument_group("interaction retrieval")
    retrieval.add_argument("--species", type=int, default=9606,
                           help="NCBI taxonomy identifier of the species of interest (9606 is human)")
    retrieval.add_argument("--required-score", type=int, default=None,
                           help="minimum STRING score (0-1000) an interaction needs to be retrieved")
    retrieval.add_argument("--add-nodes", type=int, default=None,
                           help="number of extra interactors STRING adds to the given proteins")

    # network metrics
    metrics = parser.add_argument_group("network metrics")
    metrics.add_argument("--weight", choices=SCORE_NAMES, default="score",
                         help="the STRING score used as the weight of the edges")
    metrics.add_argument("--zero-distance", type=float, default=1000000,
                         help="distance given to the edges of weight zero in the weighted betweenness centrality")

    # hub proteins
    hubs = parser.add_argument_group("hub proteins")
    hubs.add_argument("--hub-method", nargs="+", choices=HUB_METHODS, default=["consensus"],
                      help="the method (or methods) used to select the hub proteins. "
                           "Every method adds its own 'is hub (method)' column and its own enrichment run")
    hubs.add_argument("--hub-metric", default="degree",
                      help="the metric used to rank the proteins. With the consensus method it also "
                           "accepts 'all' and 'unweighted'")
    hubs.add_argument("--top-k", type=int, default=10,
                      help="number of top proteins selected as hubs by the topk and consensus methods")
    hubs.add_argument("--n-sd", type=float, default=2,
                      help="number of standard deviations above the mean used as threshold by the sd method")
    hubs.add_argument("--ascending", action="store_true",
                      help="with the topk method, select the lowest values of the metric instead of the highest")
    hubs.add_argument("--how", choices=("intersection", "sum"), default="intersection",
                      help="how the consensus method combines the rankings of the different metrics")

    # functional enrichment
    enrichment = parser.add_argument_group("functional enrichment")
    enrichment.add_argument("--background", default="network",
                            help="background of the enrichment: 'network', 'genome', or the path of a file "
                                 "containing one protein identifier per line to use as a custom background")
    enrichment.add_argument("--no-enrich", action="store_true",
                            help="skip the functional enrichment (and its plot)")

    # output
    output = parser.add_argument_group("output")
    output.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="folder where every plot and table is written. It is created if missing")
    output.add_argument("--prefix", default="",
                        help="string put in front of the name of every written file")
    output.add_argument("--save-tables", action="store_true",
                        help="also write the interaction, node data and enrichment tables as CSV files")
    output.add_argument("--no-network", action="store_true", help="skip the interactive network plot")
    output.add_argument("--no-clustermap", action="store_true", help="skip the clustermap of the metrics")
    output.add_argument("--no-enrichment-plot", action="store_true", help="skip the dot plot of the enrichment")

    # network plot
    network_plot = parser.add_argument_group("network plot")
    network_plot.add_argument("--color-by", default=None,
                              help="column of the node data mapped on the color of the nodes. "
                                   "By default the hub mask is used when there is only one")
    network_plot.add_argument("--size-by", default="degree",
                              help="numeric column of the node data mapped on the size of the nodes")
    network_plot.add_argument("--highlight", default=None,
                              help="proteins ringed in the network plot: a hub mask column, an enriched term "
                                   "or its description, or protein identifiers separated by spaces")
    network_plot.add_argument("--min-score", type=float, default=0.0,
                              help="interactions with a score lower than this value are not drawn")
    network_plot.add_argument("--node-pos-scale", type=float, default=600,
                              help="scale of the pre computed layout of the network plot")
    network_plot.add_argument("--physics", action="store_true",
                              help="leave the physics simulation running instead of freezing the layout")

    # metrics and enrichment plots
    other_plots = parser.add_argument_group("metrics and enrichment plots")
    other_plots.add_argument("--metrics-to-show", default="absolute",
                             help="metrics drawn in the clustermap: 'all', 'absolute', 'relative', "
                                  "'unweighted', or a string contained in the wanted metric names")
    other_plots.add_argument("--normalize", action="store_true",
                             help="rescale every metric inside [0, 1] before clustering them")
    other_plots.add_argument("--category", default="Process",
                             help="category of enriched terms plotted (ex Process, KEGG, Component). "
                                  "'all' keeps every category")
    other_plots.add_argument("--top-n", type=int, default=20,
                             help="number of most significant enriched terms plotted for every run")

    parser.add_argument("-q", "--quiet", action="store_true",
                        help="do not print the progress of the analysis")

    return parser


def resolve_background(background):
    """Turn the background given on the command line into the value enrich() wants

    The two special strings are passed through, while anything else is read as
    the path of a file containing a custom background of protein identifiers.

    Parameters
    -------
    background: str
        The value given to the --background parameter.

    Returns
    -------
    str or list
        The background as enrich() accepts it
    """
    if background in POSSIBLE_ENRICHMENT_BACKGROUND_STR:
        return background

    background_path = resolve_input_path(background)
    if background_path is None:
        raise FileNotFoundError(f"the custom background '{background}' is not a file. The background must be one of "
                                f"{POSSIBLE_ENRICHMENT_BACKGROUND_STR} or the path of a file of protein identifiers")
    return read_identifiers_file(background_path)


def resolve_highlight_argument(highlight, ppi_object):
    """Turn the highlight given on the command line into the value plot_network() wants

    A string naming a column of the node data or an enriched term is passed
    through untouched, while anything else is split on the spaces into the
    list of protein identifiers to ring.

    Parameters
    -------
    highlight: str or None
        The value given to the --highlight parameter.

    ppi_object: PrPrInObject
        The analysed object, used to tell the column and term names apart from
        the protein identifiers.

    Returns
    -------
    str, list or None
        The selection as plot_network() accepts it
    """
    if highlight is None:
        return None

    # a hub mask column or a single enriched term is understood by plot_network() as it is
    if highlight in ppi_object.node_data.columns:
        return highlight
    if ppi_object.enrichment is not None:
        known_terms = set(ppi_object.enrichment["term"]) | set(ppi_object.enrichment["description"])
        if highlight in known_terms:
            return highlight

    return highlight.split()


def run_analysis(args, ppi_object=None):
    """Run the whole analysis described by the parsed command line parameters

    Parameters
    -------
    args: argparse.Namespace
        The parameters given on the command line, as build_parser() returns them.

    ppi_object: PrPrInObject or None
        The object the analysis is run on. A new empty one is made when None.

    Returns
    -------
    PrPrInObject
        The object holding every result of the analysis
    """
    def log(message):
        """Print the progress of the analysis unless the user asked for silence"""
        if not args.quiet:
            print(message)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    def out_path(name):
        """Build the absolute path of a written file inside the output folder"""
        return outdir / f"{args.prefix}{name}"

    if ppi_object is None:
        ppi_object = PrPrInObject()

    # interaction retrieval. Only the parameters actually given are passed to STRING, so
    # that its own defaults are kept for the missing ones
    retrieval_kwargs = {"species": args.species}
    if args.required_score is not None:
        retrieval_kwargs["required_score"] = args.required_score
    if args.add_nodes is not None:
        retrieval_kwargs["add_nodes"] = args.add_nodes

    log("retrieving the interactions from STRING...")
    ppi_object.get_raw_data(args.proteins, **retrieval_kwargs)

    log("building the graph...")
    ppi_object.create_graph()

    log("calculating the metrics of the network...")
    ppi_object.calculate_metrics(used_weight=args.weight, zero_distance=args.zero_distance)

    # hub selection. Every requested method adds its own boolean mask to the node data
    for method in args.hub_method:
        log(f"finding the hub proteins with the {method} method...")
        ppi_object.find_hubs(method=method, metric=args.hub_metric, k=args.top_k,
                             n_sd=args.n_sd, ascending=args.ascending, how=args.how)

    # functional enrichment, run once per hub selection method so that the results of the
    # different methods can be compared in the same plot
    if not args.no_enrich:
        background = resolve_background(args.background)
        for method in args.hub_method:
            log(f"running the functional enrichment of the hubs found with the {method} method...")
            ppi_object.enrich(f"is hub ({method})", background=background)

    # plots
    if not args.no_network:
        log("writing the interactive network plot...")
        ppi_object.plot_network(color_by=args.color_by, size_by=args.size_by,
                                highlight=resolve_highlight_argument(args.highlight, ppi_object),
                                min_score=args.min_score, physics=args.physics,
                                node_pos_scale=args.node_pos_scale,
                                heading=f"protein-protein interaction network of {args.proteins}",
                                file=out_path("network.html"))

    if not args.no_clustermap:
        log("writing the clustermap of the metrics...")
        ppi_object.plot_metrics(metrics_to_show=args.metrics_to_show, normalize=args.normalize,
                                hub_method=f"is hub ({args.hub_method[0]})",
                                file=out_path("clustermap.png"))

    if not args.no_enrich and not args.no_enrichment_plot and ppi_object.enrichment is not None:
        log("writing the dot plot of the enrichment...")
        ppi_object.plot_enrichment(category=args.category, top_n=args.top_n,
                                   file=out_path("enrichment.png"))

    # tables
    if args.save_tables:
        for table in SAVED_TABLES:
            content = getattr(ppi_object, table)
            if content is not None:
                content.to_csv(out_path(f"{table}.csv"))
                log(f"written {out_path(f'{table}.csv')}")

    log(f"\nthe analysis is done, every result is in {outdir}")
    return ppi_object


def main(argv=None):
    """Entry point of the command line program

    Parameters
    -------
    argv: list or None
        The command line parameters. When None, the ones the program was called
        with are used.

    Returns
    -------
    int
        The exit code of the program: 0 when the analysis went well, 1 when it
        stopped on an error
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_analysis(args)
    except (ValueError, TypeError, FileNotFoundError, OSError) as error:
        # the message is written on the standard error and the exit code tells an
        # automated pipeline that the analysis did not go through
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0
