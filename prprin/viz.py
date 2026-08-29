import re
from pathlib import Path
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability
PLOTS_DIR = BASE_DIR / "plots" # plots path safe for inter-machine operability

SYMBOL_COL_NAME = "symbol"
UNWEIGHTED_COLUMNS = ["degree", "degree centrality", "betweenness centrality", "clustering coefficient"]
# columns of the enrichment table (the one outputed by stringdb.get_enrichment()) needed to plot it
ENRICHMENT_COLUMNS = ["category", "term", "description", "number_of_genes",
                      "number_of_genes_in_background", "fdr", "run"]
# colors used when a boolean column (like the "is hub (method)" masks) is mapped on the nodes
BOOL_COLORS = {True: "#d62728", False: "#c6d5e3"}
DEFAULT_COLOR = "#97c2fc"   # color of the nodes when no coloring column is given
MISSING_COLOR = "#dddddd"   # color of the nodes not described by the coloring column
HIGHLIGHT_COLOR = "#111111" # color of the ring drawn around the highlighted nodes

# the vis-network bundled by pyvis writes the tooltips with innerText (so no HTML is
# rendered) inside a box styled with 'white-space: nowrap' (so no line break is shown).
# This style is injected in the generated page to make plain text multi line tooltips work
TOOLTIP_CSS = """div.vis-tooltip {
    white-space: pre;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
    text-align: left;
}"""

LEGEND_CSS = """div.prprin-legend {
    position: fixed;
    left: 14px;
    bottom: 14px;
    z-index: 999;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #808074;
    border-radius: 4px;
    padding: 8px 10px;
    font-family: verdana, sans-serif;
    font-size: 12px;
    color: #000;
}
div.prprin-legend b { display: block; margin-bottom: 5px; }
div.prprin-legend span.swatch {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: -2px;
}"""


def _scale(values, low, high):
    """Linearly rescale an array of values inside the [low, high] range

    Constant (or empty) inputs are mapped on the middle of the range, so that
    they do not produce zero sized nodes or invisible edges.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    v_min, v_max = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max == v_min:
        return np.full(values.shape, (low + high) / 2)
    return low + (values - v_min) * (high - low) / (v_max - v_min)


def node_colors(node_data, color_by=None, cmap="viridis"):
    """Map a column of the node data on a color for every node

    This function translates one column of node_data into a color per node.
    Boolean columns (like the "is hub (method)" masks) get two contrasting
    colors, numeric columns get a continuous color map, and any other column is
    treated as categorical.

    Parameters
    -------
    node_data: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container. It must be indexed by stringId.

    color_by: str or None
        The name of the column of node_data mapped on the color of the nodes.
        If None, every node gets the same default color.

    cmap: str
        The name of the matplotlib color map used for the numeric columns.

    Returns
    -------
    tuple
        A pandas.Series of hex colors indexed like node_data, and a dictionary
        describing the legend of the used encoding
    """
    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")
    if color_by is None:
        return pd.Series(DEFAULT_COLOR, index=node_data.index), {"kind": "none"}
    if not isinstance(color_by, str):
        raise TypeError("color_by must be a string telling the column to map on the node colors")
    if color_by not in node_data.columns:
        raise ValueError(f"color_by must indicate one of the columns present in node_data. The current columns in node_data are:\n{list(node_data.columns)}")

    column = node_data[color_by]

    if pd.api.types.is_bool_dtype(column):  # boolean masks (ex the "is hub (method)" columns)
        colors = column.map(BOOL_COLORS)
        legend = {"kind": "categorical", "title": color_by,
                  "items": [("True", BOOL_COLORS[True]), ("False", BOOL_COLORS[False])]}
    elif pd.api.types.is_numeric_dtype(column):  # metrics (ex degree, betweenness centrality)
        color_map = colormaps[cmap]
        v_min, v_max = float(np.nanmin(column)), float(np.nanmax(column))
        norm = Normalize(vmin=v_min, vmax=v_max)
        colors = pd.Series([to_hex(color_map(norm(value))) for value in column], index=column.index)
        legend = {"kind": "gradient", "title": color_by, "vmin": v_min, "vmax": v_max,
                  "stops": [to_hex(color_map(step)) for step in np.linspace(0, 1, 9)]}
    else:   # any other column is treated as categorical
        categories = sorted(column.dropna().unique().tolist(), key=str)
        color_map = colormaps["tab20" if len(categories) > 10 else "tab10"]
        lookup = {category: to_hex(color_map(i % color_map.N)) for i, category in enumerate(categories)}
        colors = column.map(lookup)
        legend = {"kind": "categorical", "title": color_by,
                  "items": [(str(category), lookup[category]) for category in categories]}

    return colors.fillna(MISSING_COLOR), legend


def node_tooltips(node_data, digits=4):
    """Build the plain text tooltip of every node from its metadata

    This function turns every row of node_data into an aligned key/value block
    of plain text, shown when the mouse hovers the node in the interactive plot.

    HTML is deliberately not used: the vis-network version bundled by pyvis
    writes the tooltip with innerText, so any tag would be shown literally.

    Parameters
    -------
    node_data: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container. It must be indexed by stringId.

    digits: int
        The number of significant digits used to format the float values.

    Returns
    -------
    dict
        Dictionary associating every stringId to its tooltip text
    """
    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")

    shown_columns = [col for col in node_data.columns if col != SYMBOL_COL_NAME]
    label_width = max((len(col) for col in shown_columns), default=0)

    tooltips = {}
    for string_id, row in node_data.iterrows():
        header = f"{row.get(SYMBOL_COL_NAME, string_id)}  ({string_id})"
        lines = [header, "-" * len(header)]
        for col in shown_columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.{digits}g}"
            lines.append(f"{col.ljust(label_width)} : {value}")
        tooltips[string_id] = "\n".join(lines)

    return tooltips


def labels_to_ids(labels, node_data, verbose=True):
    """Translate a collection of protein identifiers into stringIds

    This function accepts both stringIds and gene symbols and returns the
    stringIds of the proteins that are present in node_data. The identifiers
    that are not found are dropped and reported with a warning.

    Parameters
    -------
    labels: iterable
        The identifiers (stringId or symbol) to translate.

    node_data: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container. It must be indexed by stringId.

    Returns
    -------
    pandas.Index
        The stringIds of the found proteins
    """
    symbol_to_id = {}
    if SYMBOL_COL_NAME in node_data.columns:
        symbol_to_id = {str(symbol): string_id for string_id, symbol in node_data[SYMBOL_COL_NAME].items()}

    found, unknown = [], []
    for label in labels:
        label = str(label).strip()
        if label in node_data.index:
            found.append(label)
        elif label in symbol_to_id:
            found.append(symbol_to_id[label])
        else:
            unknown.append(label)

    if unknown and verbose:
        print(f"\nWarning: {len(unknown)} identifiers to highlight are not nodes of the network and were ignored: {unknown[:10]}\n")

    return pd.Index(list(dict.fromkeys(found)), name=node_data.index.name)


def resolve_highlight(highlight, node_data, enrichment=None):
    """Find the nodes to highlight from several kinds of selection

    This function accepts the results that emerged from the previous analyses
    and returns the stringIds of the proteins they point to. It understands:

    - None: nothing is highlighted
    - the name of a boolean column of node_data (ex "is hub (consensus)")
    - the term or the description of an enriched set (ex "GO:0006281" or
      "DNA repair"): the proteins of that set are taken from the 'inputGenes'
      column of the enrichment table
    - an iterable of stringIds or gene symbols
    - a single stringId or gene symbol

    Parameters
    -------
    highlight: None, str or iterable
        The selection of proteins to highlight (see above).

    node_data: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container. It must be indexed by stringId.

    enrichment: pandas.DataFrame or None
        The enrichment table. The one stored in the 'enrichment' container.
        Only needed to highlight an enriched term.

    Returns
    -------
    pandas.Index
        The stringIds of the nodes to highlight
    """
    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")

    if highlight is None:
        return pd.Index([], name=node_data.index.name)

    if isinstance(highlight, str):
        # a boolean column of node_data (ex an "is hub (method)" mask)
        if highlight in node_data.columns:
            if not pd.api.types.is_bool_dtype(node_data[highlight]):
                raise TypeError(f"the column '{highlight}' is not a boolean mask, so it cannot be used to select the nodes to highlight")
            return node_data.index[node_data[highlight]]

        # an enriched term or its description
        if isinstance(enrichment, pd.DataFrame) and len(enrichment) > 0:
            selected_terms = ((enrichment["term"] == highlight)
                              | (enrichment["description"].str.casefold() == highlight.casefold()))
            if selected_terms.any():
                genes = []
                for cell in enrichment.loc[selected_terms, "inputGenes"]:
                    genes.extend(str(cell).split(","))
                return labels_to_ids(genes, node_data)

        # a single protein
        highlight = [highlight]

    elif not isinstance(highlight, (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index)):
        raise TypeError("highlight must be None, a string or an iterable of protein identifiers")

    return labels_to_ids(highlight, node_data)


def build_legend_html(legend, highlight_label=None):
    """Build the HTML of the legend of the interactive network

    pyvis draws no legend, so a small floating box describing the used color
    encoding is injected in the generated page.
    """
    rows = []
    if legend.get("kind") == "categorical":
        rows.append(f"<b>{legend['title']}</b>")
        for label, color in legend["items"]:
            rows.append(f"<span class='swatch' style='background:{color}'></span>{label}<br>")
    elif legend.get("kind") == "gradient":
        gradient = ", ".join(legend["stops"])
        rows.append(f"<b>{legend['title']}</b>")
        rows.append(f"<div style='width:150px;height:12px;background:linear-gradient(to right, {gradient});border:1px solid #808074'></div>")
        rows.append(f"<span style='float:left'>{legend['vmin']:.3g}</span>"
                    f"<span style='float:right'>{legend['vmax']:.3g}</span><br style='clear:both'>")

    if highlight_label:
        rows.append(f"<span class='swatch' style='background:#fff;border:3px solid {HIGHLIGHT_COLOR}'></span>{highlight_label}<br>")

    if not rows:
        return ""
    return "<div class='prprin-legend'>" + "".join(rows) + "</div>"


def relabel_select_menu(html, id_to_label):
    """Make the node select menu of a pyvis page search and display by label

    pyvis's bundled template.html hard codes both the value and the displayed
    text of every <option> of the "select-node" dropdown to the raw vis.js node
    id (see the select_menu block of pyvis/templates/template.html). Since
    TomSelect (the JS widget backing that dropdown) searches the option text by
    default, this leaves the value (needed by the onchange="selectNode([value])"
    handler, so it must stay the node id) untouched and only swaps the visible
    text for the given label, so that typing a gene symbol finds the node.

    Parameters
    -------
    html: str
        The HTML page generated by pyvis (Network.generate_html()).

    id_to_label: dict
        Mapping of every node id to the label shown and searched in its place.

    Returns
    -------
    str
        The HTML page with the node select menu relabeled
    """
    def _relabel_options(select_block):
        """Swap the displayed text of every <option> of the matched <select> block"""
        def _relabel_option(option_match):
            """Keep the value of the matched <option> and relabel its text"""
            node_id = option_match.group(1)
            label = id_to_label.get(node_id, node_id)
            return f'<option value="{node_id}">{label}</option>'
        return re.sub(r'<option value="([^"]*)">[^<]*</option>', _relabel_option, select_block.group(0))

    html = re.sub(r'<select[^>]*id="select-node".*?</select>', _relabel_options, html, count=1, flags=re.DOTALL)
    return html.replace("Select a Node by ID", "Select a Node by symbol", 1)


def plot_a_graph(graph, node_data, color_by=None, size_by="degree", highlight=None, enrichment=None,
                 used_weight="score", min_score=0.0, cmap="viridis", physics=False,
                 file="network.html", node_pos_scale=600, node_size_range=(10, 45),
                 edge_width_range=(0.3, 4.0), heading="", seed=0, verbose=False):
    """Plot an interactive view of a protein-protein interaction graph

    This function writes a self contained HTML page showing the given graph,
    where the nodes are colored by one column of node_data, sized by another
    one, carry all of their metadata in the tooltip, and where a selection of
    proteins can be ringed to highlight it.

    Parameters
    -------
    graph: networkx.Graph
        The graph to plot. The one stored in the 'graph' container.

    node_data: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container. It must be indexed by stringId.

    color_by: str or None
        The name of the column of node_data mapped on the color of the nodes
        (see node_colors()).

    size_by: str or None
        The name of the numeric column of node_data mapped on the size of the
        nodes. If None, every node gets the same size.

    highlight: None, str or iterable
        The selection of proteins ringed in the plot (see resolve_highlight()).

    enrichment: pandas.DataFrame or None
        The enrichment table. The one stored in the 'enrichment' container.
        Only needed to highlight an enriched term.

    used_weight: str
        The name of the edge attribute used as the weight of the layout, as the
        width of the edges and as the filtering score.

    min_score: numeric
        Interactions with a used_weight lower than this value are not drawn.

    cmap: str
        The name of the matplotlib color map used for the numeric columns.

    physics: bool
        If False (default), the nodes are frozen on a pre computed spring
        layout, which stays readable on big networks. If True, the vis.js
        physics simulation is left running.

    file: str or pathlib.Path
        Name of the written HTML page. Relative names are saved in the 'plots'
        folder of the project.

    Returns
    -------
    pathlib.Path
        The path of the written HTML page
    """
    # input check
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")
    if not isinstance(used_weight, str):
        raise TypeError("used_weight must be a string telling the edge attribute to use as the weight")
    if size_by is not None:
        if not isinstance(size_by, str):
            raise TypeError("size_by must be a string telling the column to map on the node sizes")
        if size_by not in node_data.columns:
            raise ValueError(f"size_by must indicate one of the columns present in node_data. The current columns in node_data are:\n{list(node_data.columns)}")
        if not pd.api.types.is_numeric_dtype(node_data[size_by]):
            raise TypeError(f"the column '{size_by}' is not numeric, so it cannot be mapped on the node sizes")
    if not isinstance(physics, bool):
        raise TypeError("physics must be a boolean value")

    # view graph creation. The interactions are filtered and only the attributes needed by
    # vis.js are kept, so that the graph in the 'graph' container is never touched
    # (pyvis from_nx() writes the plotting attributes inside the graph it is given)
    view = nx.Graph()
    view.add_nodes_from(graph.nodes)
    for node_a, node_b, edge_attributes in graph.edges(data=True):
        score = float(edge_attributes.get(used_weight, 0.0))
        if score >= min_score:
            view.add_edge(node_a, node_b, **{used_weight: score})
    if view.number_of_edges() == 0:
        print(f"\nWarning: no interaction has a {used_weight} of at least {min_score}, the plotted network has no edge\n")

    # node encodings
    colors, legend = node_colors(node_data, color_by, cmap)
    tooltips = node_tooltips(node_data)
    highlighted = resolve_highlight(highlight, node_data, enrichment)
    if size_by is None:
        sizes = pd.Series(np.mean(node_size_range), index=node_data.index)
    else:
        sizes = pd.Series(_scale(node_data[size_by], *node_size_range), index=node_data.index)

    # layout pre computation. The positions are given to vis.js as node attributes,
    # so no physics simulation has to run in the browser to place the nodes
    pos = nx.spring_layout(view, weight=used_weight, seed=seed, scale=node_pos_scale)

    # node attributes assignment. pyvis from_nx() forwards every node attribute to vis.js,
    # so the whole encoding can be written on the graph before the conversion
    for node in view.nodes:
        x, y = pos[node]
        node_color = colors.get(node, MISSING_COLOR)
        is_highlighted = node in highlighted
        view.nodes[node].update(
            label=str(node_data.at[node, SYMBOL_COL_NAME]) if node in node_data.index else str(node),
            title=tooltips.get(node, str(node)),
            color={"background": node_color,
                   "border": HIGHLIGHT_COLOR if is_highlighted else node_color,
                   "highlight": {"background": node_color, "border": HIGHLIGHT_COLOR}},
            borderWidth=4 if is_highlighted else 1,
            borderWidthSelected=6 if is_highlighted else 2,
            size=float(sizes.get(node, np.mean(node_size_range))),
            x=float(x),
            y=float(y),
            physics=physics,
        )

    # edge attributes assignment. from_nx() moves the 'weight' attribute into the vis.js
    # 'width' one, so the scaled width is written there
    symbols = node_data[SYMBOL_COL_NAME].to_dict() if SYMBOL_COL_NAME in node_data.columns else {}
    edges = list(view.edges(data=True))
    widths = _scale([edge_attributes[used_weight] for _, _, edge_attributes in edges], *edge_width_range)
    for (node_a, node_b, edge_attributes), width in zip(edges, widths):
        edge_attributes["weight"] = float(width)
        edge_attributes["title"] = (f"{symbols.get(node_a, node_a)} - {symbols.get(node_b, node_b)}\n"
                                    f"{used_weight}: {edge_attributes[used_weight]:.3f}")

    # pyvis plot from the networkx one
    nt = Network(notebook=False, cdn_resources="in_line", height="800px", width="100%",
                 heading=heading, neighborhood_highlight=True, select_menu=True, filter_menu=True)
    nt.from_nx(view)
    nt.options.interaction.hover = True
    nt.options.interaction.tooltipDelay = 100
    nt.toggle_physics(physics)

    # the tooltip style and the legend are injected at the end of the body, so that they
    # override the style sheet that pyvis inlines in the head
    highlight_label = f"highlighted ({len(highlighted)} proteins)" if len(highlighted) else None
    injected = (f"<style>{TOOLTIP_CSS}\n{LEGEND_CSS}</style>\n"
                f"{build_legend_html(legend, highlight_label)}")
    html = nt.generate_html(notebook=False)
    # pyvis's bundled template.html renders <h1>{{heading}}</h1> twice unconditionally
    # (both instances end up before </head>, which browsers silently relocate into the
    # body), so the first occurrence is dropped here to keep the heading singular
    html = re.sub(r"<h1>.*?</h1>", "", html, count=1)
    html = relabel_select_menu(html, symbols)
    html = html.replace("</body>", injected + "\n</body>", 1) if "</body>" in html else html + injected

    # graph saving
    path_file = Path(file)
    if not path_file.is_absolute():
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path_file = PLOTS_DIR / path_file
    path_file.write_text(html, encoding="utf-8")

    if verbose:
        print(f"interactive network of {view.number_of_nodes()} proteins and "
              f"{view.number_of_edges()} interactions written in {path_file}")
    return path_file


def metrics_clustermap(metrics, metrics_to_show="absolute", normalize=False, hub_method=None, file="clustermap.png",
                       weight_used="score"):
    """Plot a clustered heatmap of the metrics of the nodes

    This function draws a seaborn clustermap of the chosen metrics, where every
    row is a protein (labelled with its symbol) and every column a metric, so
    that the proteins behaving in a similar way in the network end up close to
    each other in the row dendrogram.

    Parameters
    -------
    metrics: pandas.DataFrame
        The DataFrame containing the metrics associated to each node. The one
        stored in the 'node_data' container.

    metrics_to_show: str or list
        It tells which metrics are drawn as the columns of the clustermap.

        "all" takes every metric, "absolute" (default) the non normalized ones
        (degree and weighted degree), "relative" all the remaining ones, and
        "unweighted" the metrics calculated without the edge weights. Any other
        string keeps the metrics whose name contains it (ex "betweenness"),
        and a list is used as the column names themselves.

    normalize: bool
        If True, every metric is rescaled inside the [0, 1] range before the
        clustering, so that metrics with different orders of magnitude (ex the
        degree and the clustering coefficient) stay comparable.

    hub_method: str or None
        The name of an "is hub (method)" boolean column of metrics, drawn as a
        color strip beside the rows to show where the hubs ended up in the
        clustering. If None, no strip is drawn.

    file: str or pathlib.Path
        Name of the written image. Relative names are saved in the 'plots'
        folder of the project.

    weight_used: str
        The name of the score used as the weight of the edges of the graph. It
        is the one indicated between parentheses in the metric column names.

    Returns
    -------
    pathlib.Path
        The path of the written image
    """
    # input check
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    if not isinstance(metrics_to_show, (str, list)):
        raise TypeError("metrics_to_show must be a string or a list")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    if not isinstance(hub_method, str) and hub_method is not None:
        raise TypeError("hub_method must be a string or None")
    if hub_method is not None and hub_method not in [col for col in metrics.columns if "is hub" in col]:
        raise ValueError("hub_method must be the name of a column telling if a protein is a hub or not")
    if not isinstance(file, (str, Path)):
        raise TypeError("file must be a string containig the name of the clustermap file")
    if not isinstance(weight_used, str):
        raise TypeError("weight_used must be a string telling the attribute used as the weight in the graph")
    if not any([weight_used in col for col in metrics.columns]):
        raise ValueError("weight_used value is not the current one being used as the weights")

    # preparing metrics for plotting
    symbol_idx_metrics = metrics.set_index("symbol") # to show symbols instead of stringId in the plot
    absolute_cols = ["degree", f"weighted degree ({weight_used})"]
    # not_numeric_cols_name = [col for col in symbol_idx_metrics.columns if "is hub" in col]

    # convert metrics_to_show into the actual plotted metrics
    cols_without_hubs = [col for col in symbol_idx_metrics.columns if "is hub" not in col]
    if metrics_to_show == "all":
        used_cols = cols_without_hubs
    elif metrics_to_show == "absolute":
        used_cols = absolute_cols
    elif metrics_to_show == "relative":
        used_cols = [col for col in cols_without_hubs if col not in absolute_cols]
    elif metrics_to_show == "unweighted":
        used_cols = [col for col in cols_without_hubs if col in UNWEIGHTED_COLUMNS]
    elif not isinstance(metrics_to_show, list):
        used_cols = [col for col in cols_without_hubs if metrics_to_show in col]
    else:
        used_cols = metrics_to_show
    print(f"\nThe selected metrics for the clustermap are: {used_cols}\n")

    # what hub method to show if at all
    if hub_method is not None: 
        lut = {True:"green", False:"red"}
        row_colors = symbol_idx_metrics[hub_method].map(lut)
    else:
        row_colors = None

    # clustermap creation
    clustermap = sns.clustermap(symbol_idx_metrics[used_cols], row_colors=row_colors, standard_scale=1 if normalize else None)

    # graph saving
    path_file = Path(file)
    if not path_file.is_absolute():
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path_file = PLOTS_DIR / path_file
    clustermap.savefig(path_file)
    plt.close(clustermap.figure)

    return path_file


def enrichment_dotplot(enrichment, run=None, category="Process", top_n=20, file="enrichment.png",
                       cmap="viridis_r", label_width=60, verbose=False):
    """Plot the enriched terms of a functional enrichment as a dot plot

    This function draws a seaborn dot plot of the enrichment, where every row is
    an enriched term and every dot carries three values at once: the gene ratio
    (how much of the term is covered by the analysed proteins) on the x axis,
    the number of proteins hitting the term as the size of the dot, and the
    significance (-log10 of the fdr) as its color.

    When more than one enrichment run is plotted, the runs are drawn side by
    side as different panels, so that the results of different hub selection
    methods (or of different backgrounds) can be compared directly.

    Parameters
    -------
    enrichment: pandas.DataFrame
        The enrichment table. The one stored in the 'enrichment' container.

    run: str, iterable or None
        The name (or names) of the enrichment runs to plot, as they are written
        in the 'run' column added by enrich(). If None (default), every run
        present in the table is plotted.

    category: str, iterable or None
        The category (or categories) of terms to plot, as they are written in
        the 'category' column (ex "Process", "KEGG", "Component", "RCTM").
        "all" (or None) keeps every category.

    top_n: int
        The number of most significant terms kept for every run. Must be a
        positive integer.

    file: str or pathlib.Path
        Name of the written image. Relative names are saved in the 'plots'
        folder of the project.

    cmap: str
        The name of the matplotlib color map used for the significance. The
        reversed maps (ex the default "viridis_r") give the darkest color to
        the most significant terms.

    label_width: int
        The descriptions of the terms are cut to this number of characters, so
        that the long ones do not squeeze the plot.

    Returns
    -------
    pathlib.Path or None
        The path of the written image, or None if the selection left no term
    """
    # input check
    if not isinstance(enrichment, pd.DataFrame):
        raise TypeError("enrichment must be a pandas DataFrame. Structured like the one contained in 'enrichment'")
    missing_columns = [col for col in ENRICHMENT_COLUMNS if col not in enrichment.columns]
    if missing_columns:
        raise ValueError(f"the enrichment table misses the columns {missing_columns}. It must be the one outputed by enrich()")
    if not (isinstance(top_n, int) and not isinstance(top_n, bool)) or top_n <= 0:
        raise TypeError("top_n must be a positive integer value")
    if not isinstance(file, (str, Path)):
        raise TypeError("file must be a string containig the name of the dot plot file")
    if not isinstance(cmap, str):
        raise TypeError("cmap must be a string telling the matplotlib color map to use")

    plot_data = enrichment.copy()

    # run selection
    if run is not None:
        wanted_runs = [run] if isinstance(run, str) else list(run)
        unknown_runs = [name for name in wanted_runs if name not in set(plot_data["run"])]
        if unknown_runs:
            raise ValueError(f"the runs {unknown_runs} are not present in the enrichment table. The runs it contains are:\n{list(dict.fromkeys(plot_data['run']))}")
        plot_data = plot_data.loc[plot_data["run"].isin(wanted_runs)]

    # category selection
    if category is not None and category != "all":
        wanted_categories = [category] if isinstance(category, str) else list(category)
        plot_data = plot_data.loc[plot_data["category"].isin(wanted_categories)]

    # empty warning
    if len(plot_data) == 0:
        print(f"\nWarning: no enriched term is left after selecting the run {run} and the category {category}, nothing was plotted\n")
        return None

    # plotted values. The fdr can be exactly zero, which would give an infinite -log10, so
    # the zeros are floored one order of magnitude below the smallest non zero fdr
    plot_data["gene ratio"] = plot_data["number_of_genes"] / plot_data["number_of_genes_in_background"]
    non_zero_fdr = plot_data.loc[plot_data["fdr"] > 0, "fdr"]
    fdr_floor = non_zero_fdr.min() / 10 if len(non_zero_fdr) > 0 else 1e-300
    plot_data["-log10(fdr)"] = -np.log10(plot_data["fdr"].clip(lower=fdr_floor))

    # only the most significant terms of every run are kept
    plot_data = plot_data.sort_values(["fdr", "gene ratio"], ascending=[True, False])
    plot_data = plot_data.groupby("run", sort=False, group_keys=False).head(top_n)

    # y labels. The descriptions are cut, and the ones that collide after the cut are
    # told apart by their term identifier. A description repeated over several runs is
    # not a collision: it has to stay one single row shared by the panels
    plot_data["label"] = plot_data["description"].astype(str).str.slice(0, label_width)
    terms_per_label = plot_data.groupby("label")["term"].nunique()
    collisions = plot_data["label"].map(terms_per_label) > 1
    plot_data.loc[collisions, "label"] = (plot_data.loc[collisions, "label"]
                                          + " (" + plot_data.loc[collisions, "term"].astype(str) + ")")
    # the terms are drawn from the most to the least covered one
    label_order = plot_data.groupby("label", observed=True)["gene ratio"].max().sort_values(ascending=False).index
    plot_data["label"] = pd.Categorical(plot_data["label"], categories=label_order, ordered=True)

    # dot plot creation. The runs are drawn as different panels only when there are several
    plotted_runs = list(dict.fromkeys(plot_data["run"]))
    grid = sns.relplot(
        data=plot_data,
        x="gene ratio",
        y="label",
        hue="-log10(fdr)",
        size="number_of_genes",
        col="run" if len(plotted_runs) > 1 else None,
        palette=cmap,
        sizes=(40, 300),
        height=max(4.0, 0.32 * len(label_order)),
        aspect=1.1 if len(plotted_runs) > 1 else 1.6,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("gene ratio (genes of the term found / genes of the term in the background)", "")
    if len(plotted_runs) == 1:
        grid.figure.suptitle(plotted_runs[0])
    for ax in grid.axes.flat:
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)
    grid.figure.tight_layout()

    # graph saving
    path_file = Path(file)
    if not path_file.is_absolute():
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path_file = PLOTS_DIR / path_file
    grid.savefig(path_file, dpi=150, bbox_inches="tight")
    plt.close(grid.figure)

    if verbose:
        print(f"dot plot of {len(plot_data)} enriched terms over {len(plotted_runs)} runs written in {path_file}")
    return path_file


class VizPPI:
    def plot_network(self, color_by=None, size_by="degree", highlight=None, min_score=0.0,
                     cmap="viridis", physics=False, file="network.html", node_pos_scale=600,
                     heading=""):
        """Plot an interactive view of the protein-protein interaction network

        This method writes a self contained HTML page showing the graph stored
        in the 'graph' container, annotated with the metrics stored in the
        'node_data' container.

        Every node is colored by one column of 'node_data', sized by another
        one and carries all of its metadata in the tooltip shown on hover. A
        selection of proteins, possibly coming from the enrichment stored in
        the 'enrichment' container, can be ringed to highlight it.

        Parameters
        -------
        color_by: str or None
            The name of the column of 'node_data' mapped on the color of the
            nodes. Boolean columns (like the "is hub (method)" masks) get two
            contrasting colors, numeric ones a continuous color map.

            If None, the only "is hub (method)" mask present in 'node_data' is
            used. If there is none, or more than one, the nodes are not colored.

        size_by: str or None
            The name of the numeric column of 'node_data' mapped on the size of
            the nodes. If None, every node gets the same size.

        highlight: None, str or iterable
            The selection of proteins ringed in the plot. It can be the name of
            a boolean column of 'node_data', the term or the description of an
            enriched set found by enrich() (ex "GO:0006281" or "DNA repair"),
            an iterable of stringIds or gene symbols, or a single one of them.

        min_score: numeric
            Interactions with a score lower than this value are not drawn.

        cmap: str
            The name of the matplotlib color map used for the numeric columns.

        physics: bool
            If False (default), the nodes are frozen on a pre computed spring
            layout, which stays readable on big networks. If True, the vis.js
            physics simulation is left running.

        file: str or pathlib.Path
            Name of the written HTML page. Relative names are saved in the
            'plots' folder of the project.
        """
        # input check
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("there is no graph saved in the 'graph' container. Run create_graph() first")
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")

        # default coloring: the only hub mask present in 'node_data', if there is one
        if color_by is None:
            hub_masks = [col for col in self.node_data.columns if "is hub" in col]
            color_by = hub_masks[0] if len(hub_masks) == 1 else None

        plot_a_graph(self.graph, self.node_data, color_by=color_by, size_by=size_by,
                     highlight=highlight, enrichment=self.enrichment,
                     used_weight=self.used_weight_name or "score", min_score=min_score,
                     cmap=cmap, physics=physics, file=file, node_pos_scale=node_pos_scale,
                     heading=heading)

        return self

    def plot_metrics(self, metrics_to_show="absolute", normalize=False, hub_method=None, file="clustermap.png"):
        """Plot a clustered heatmap of the metrics of the network

        This method writes an image of a clustermap of the metrics stored in
        the 'node_data' container, where every row is a protein (labelled with
        its symbol) and every column a metric, so that the proteins behaving in
        a similar way in the network end up close to each other in the row
        dendrogram.

        Parameters
        -------
        metrics_to_show: str or list
            It tells which metrics are drawn as the columns of the clustermap.

            "all" takes every metric, "absolute" (default) the non normalized
            ones (degree and weighted degree), "relative" all the remaining
            ones, and "unweighted" the metrics calculated without the edge
            weights. Any other string keeps the metrics whose name contains it
            (ex "betweenness"), and a list is used as the column names.

        normalize: bool
            If True, every metric is rescaled inside the [0, 1] range before
            the clustering, so that metrics with different orders of magnitude
            (ex the degree and the clustering coefficient) stay comparable.

        hub_method: str or None
            The name of an "is hub (method)" boolean column of 'node_data',
            drawn as a color strip beside the rows to show where the hubs ended
            up in the clustering. If None, no strip is drawn.

        file: str or pathlib.Path
            Name of the written image. Relative names are saved in the 'plots'
            folder of the project.
        """
        # input check
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")

        # metrics creation adn saving
        metrics_clustermap(self.node_data, metrics_to_show, normalize, hub_method, file, weight_used=self.used_weight_name)

        return self

    def plot_enrichment(self, run=None, category="Process", top_n=20, file="enrichment.png",
                        cmap="viridis_r"):
        """Plot the enriched terms of the network as a dot plot

        This method writes an image of a dot plot of the enrichment stored in
        the 'enrichment' container, where every row is an enriched term and
        every dot carries three values at once: the gene ratio (how much of the
        term is covered by the hub proteins) on the x axis, the number of
        proteins hitting the term as the size of the dot, and the significance
        (-log10 of the fdr) as its color.

        When enrich() was run more than once, the runs are drawn side by side
        as different panels, so that the results of different hub selection
        methods (or of different backgrounds) can be compared directly.

        Parameters
        -------
        run: str, iterable or None
            The name (or names) of the enrichment runs to plot, as they are
            written in the 'run' column of 'enrichment' (ex "hubs (consensus)
            vs network"). If None (default), every run is plotted.

        category: str, iterable or None
            The category (or categories) of terms to plot, as they are written
            in the 'category' column (ex "Process", "KEGG", "Component",
            "RCTM"). "all" (or None) keeps every category.

        top_n: int
            The number of most significant terms kept for every run. Must be a
            positive integer.

        file: str or pathlib.Path
            Name of the written image. Relative names are saved in the 'plots'
            folder of the project.

        cmap: str
            The name of the matplotlib color map used for the significance. The
            reversed maps (ex the default "viridis_r") give the darkest color
            to the most significant terms.
        """
        # input check
        if not isinstance(self.enrichment, pd.DataFrame):
            raise ValueError("there is no enrichment saved in the 'enrichment' container. Run enrich() first")

        # dot plot creation and saving
        enrichment_dotplot(self.enrichment, run=run, category=category, top_n=top_n,
                           file=file, cmap=cmap)

        return self
