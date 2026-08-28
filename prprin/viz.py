from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability
PLOTS_DIR = BASE_DIR / "plots" # plots path safe for inter-machine operability

SYMBOL_COL_NAME = "symbol"
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


class VizPPI:
    def plot_network(self, color_by=None, size_by="degree", highlight=None, min_score=0.0,
                     cmap="viridis", physics=False, file="network.html", node_pos_scale=600,
                     heading="", verbose=True):
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

    def plot_metrics(self, metrics="unweighted", file="metrics.html"): ...
    def plot_enrichment(self, run=None, category="Process", top_n=20,
                        file="enrichment.html"): ...
    def report(self, out="prprin_report.html"): ...   # stitches the three together


if __name__ == "__main__":
    pass
