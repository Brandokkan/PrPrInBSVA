import prprin as pr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability

if __name__ == "__main__":
    first_obj = pr.PrPrInObject()
    # first_obj.get_raw_data("test_big").create_graph().calculate_metrics().find_hubs(method="consensus", k=5, metric="degree").find_hubs("topk", "betweenness centrality", k=5)
    # first_obj.enrich("is hub (topk)").enrich("is hub (consensus)")
    first_obj.get_raw_data("test_big").create_graph().calculate_metrics().find_hubs(method="consensus", k=5, metric="degree")
    first_obj.enrich()
    # first_obj.get_raw_data("test").create_graph().calculate_metrics().find_hubs(method="consensus", k=1, metric="degree")
    # first_obj.enrich()
    # first_obj.plot_network(color_by="weighted degree (score)", node_pos_scale=3000, highlight="is hub (consensus)", heading="interactive graph of proteins inside test_big." \
    # " Highlighted the hubs found by consensus", file="test_big network.html")
    # first_obj.plot_network(color_by="weighted degree (score)", node_pos_scale=300, highlight="is hub (consensus)", heading="interactive graph of proteins inside test." \
    # " Highlighted the hubs found by consensus", file="test network.html")
    # print(first_obj.graph.degree(weight="score"))
    # print(first_obj, first_obj.proteins, first_obj.network, sep="\n\n")
    # print(first_obj.node_data.T)
    # first_obj.node_data.to_csv(DATA_DIR / "ex_large_node_data.csv", index=True)
    print(first_obj.enrichment)
    # print(first_obj)
