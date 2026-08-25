import prprin as pr

if __name__ == "__main__":
    first_obj = pr.PrPrInObject()
    first_obj.get_raw_data("r").create_graph()
    print(first_obj, first_obj.proteins, first_obj.network, sep="\n\n")