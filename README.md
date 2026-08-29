# prprin - protein-protein interaction network analysis

`prprin` builds a protein-protein interaction network from the
[STRING](https://string-db.org/) database and analyses it: it computes the
classical network metrics of every protein, selects the hub proteins with three
different strategies, runs the functional enrichment of those hubs and writes an
interactive view of the network together with two summary plots.

The whole analysis is driven from the command line, so no parameter and no file
name has to be written inside the source code, and the program never asks
anything interactively: it can be put inside an automated pipeline as it is.

## Requirements

Python 3.9 or newer, and the packages listed in `requirements.txt`:

    pip install -r requirements.txt

An internet connection is needed at run time: both the interactions and the
enrichment are downloaded from the STRING web API.

## Quick start

From the folder where the archive was unpacked:

    python run_prprin.py test_big --hub-method topk --top-k 8 --outdir results --save-tables

This retrieves the interactions of the 100 proteins listed in `data/test_big`,
computes the metrics, takes the 8 proteins of highest degree as the hubs,
enriches them against the rest of the network and writes every plot and table in
`results`.

The complete list of the parameters is printed by:

    python run_prprin.py --help

The package can also be run as a module, which does exactly the same thing:

    python -m prprin test_big --top-k 8

## Giving the proteins

The only mandatory parameter is the set of proteins of interest, given either as
the identifiers themselves, quoted and separated by spaces:

    python run_prprin.py "TP53 BRCA1 BRCA2 ATM"

or as the path of a file holding one identifier per line:

    python run_prprin.py my_proteins.txt

Both gene symbols and STRING ids are accepted, and the two can be mixed. A path
is looked for as it was written (so absolute paths and paths relative to the
working directory both work) and only afterwards inside the `data` folder of the
project, which is why the two bundled example files can be called by their bare
name. A string that looks like a path but matches no file stops the program
instead of being sent to STRING as a list of identifiers.

## What the analysis does

1. **Retrieval.** The identifiers are mapped to STRING ids and the interactions
   between them are downloaded. `--species` chooses the organism (9606, human,
   by default), `--required-score` the minimum confidence an interaction needs,
   and `--add-nodes` asks STRING for extra interactors beyond the given proteins.
2. **Graph.** An undirected graph is built, keeping all eight STRING scores of
   every interaction as edge attributes, so any of them can be used as the
   weight.
3. **Metrics.** Ten metrics are computed for every protein and stored, beside its
   symbol, in the `node_data` table: `degree`, `degree centrality`, `betweenness centrality` and
   `clustering coefficient` for the un-weighted graph, and, for the score chosen
   with `--weight`, `weighted degree`, `weighted degree normalized`, `weighted
   clustering coefficient` and three weighted betweenness centralities. The three
   of them exist because betweenness treats the weights as distances while the
   STRING scores are similarities, so the conversion has to be chosen:
   `1 - score`, `1 / score` and `-log(score)` are all computed and can be
   compared.
4. **Hub selection.** `--hub-method` picks one or more of three strategies, and
   each of them adds its own `is hub (method)` column to `node_data`:
   - `topk` takes the `--top-k` best ranked proteins of a single metric;
   - `sd` takes the proteins lying more than `--n-sd` standard deviations above
     the mean of a single metric;
   - `consensus` combines the rankings of several metrics at once, either by
     summing the ranks or by intersecting the top k of each of them (`--how`).
5. **Enrichment.** The hubs of every method are sent to the STRING functional
   enrichment service. `--background` decides what they are compared against:
   `network` (the proteins of the network, the default), `genome` (the whole
   genome of the species), or the path of a file of identifiers to use as a
   custom background.

## What it writes

Everything goes in `--outdir` (`prprin_results` by default, created if missing),
with the name prefixed by `--prefix`:

| file | content |
| --- | --- |
| `network.html` | interactive network: the nodes are colored by one column of the node data (`--color-by`), sized by another one (`--size-by`), carry all of their metrics in the tooltip, and can be ringed with `--highlight`. Self contained, so it opens in any browser |
| `clustermap.png` | clustered heatmap of the metrics, with the hubs marked beside the rows |
| `enrichment.png` | dot plot of the enriched terms: gene ratio on the x axis, number of proteins as the size of the dot, significance as its color. Several hub methods are drawn side by side as separate panels |
| `network.csv`, `node_data.csv`, `enrichment.csv` | the three tables, written only with `--save-tables` |

Any plot can be skipped with `--no-network`, `--no-clustermap` or
`--no-enrichment-plot`, and the enrichment itself with `--no-enrich`.

## Using it as a library

The command line is a thin layer over one object, which can be used directly.
Every method returns the object itself, so the analysis can be chained:

```python
import prprin

obj = (prprin.PrPrInObject()
       .get_raw_data("TP53 BRCA1 BRCA2 EGFR MYC")
       .create_graph()
       .calculate_metrics()
       .find_hubs(method="topk", metric="degree", k=5))

obj.enrich(background="genome")
obj.plot_network(color_by="betweenness centrality", highlight="is hub (topk)")
print(obj)
```

The results stay in the containers of the object: `proteins`, `network`, `graph`,
`node_data`, `hubs` and `enrichment`. `find_hubs()` and `enrich()` can be called
several times on the same object, with different methods or different
backgrounds, and the results accumulate so that they can be compared in the same
plot.

Highlighting also understands the results of the enrichment, so a term found in
the previous step can be shown back on the network:

```python
obj.plot_network(highlight="DNA repair")
```

## Project layout

    run_prprin.py       stand alone script, the entry point of the project
    prprin/
        cli.py          command line interface
        core.py         the object holding the data of an analysis
        retrive.py      retrieval of the interactions from STRING
        graph.py        construction of the graph
        metrics.py      un-weighted and weighted network metrics
        hubs.py         the three hub selection strategies
        enrich.py       functional enrichment of the hubs
        viz.py          the three plots
    tests/              unit tests, one module per module of the package
    data/               example identifier files and example tables
    plots/              default destination of the plots of the library interface

## Example data

`data/test` holds four identifiers (three symbols and one STRING id, to show that
both are accepted) and gives a small network, useful to look at the interactive
plot node by node. `data/test_big` holds a hundred cancer related genes and gives
a network of a few thousand interactions, which is the size the hub selection and
the enrichment are meant for.

## Tests

The tests need no internet connection: every call to STRING is mocked, so they
check the analysis itself and not the availability of the service. From the
project folder:

    python -m unittest discover -s tests -t .

## Notes

- The two score parameters do not use the same scale, because they are passed to
  two different places. `--required-score` is the one of the STRING API and goes
  from 0 to 1000, while `--min-score`, which drops the weak interactions from the
  drawing only, is compared against the scores STRING returns, which go from 0 to
  1. `--required-score 700` and `--min-score 0.7` therefore mean the same thing.
- `--required-score` is left to the STRING default (400) when it is not given.
- Which categories of terms come out of the enrichment depends on the network:
  `--category Process` is a reasonable default but it can select nothing at all,
  and `--category all` shows what was actually found.
