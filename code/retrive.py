import stringdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability


def read_input():
    """Reads and proces the user input to obtain the proteins identifiers

    This function accepts, reads and checks the user input 
    to obtain the proteins identifiers. Identifiers can be
    the protein symbol or stringId.

    It accepts two kind of inputs:
        - A simple string containing the identifiers of the
        proteins of interest separated by spaces.
        - A path (global or relative) to a file were each line
        contains one identifier of a protein of interest (the 
        relative path is in the data folder).

    Returns
    -------
    list
        list containing the indentifiers of the proteins of
        interest.
    """
    in_input_loop = True
    while in_input_loop:
        inp_string = input("Insert the identifiers of the protein of interest separated\n" \
                        "by spaces or a path (global or relative) to a file containing\n" \
                        "the identifiers of the proteins on separate lines\n")
        while True:
            confirm = input("do you confirm your input (y/n): ")
            if confirm.lower() == "y":
                print("input confirmed")
                in_input_loop = False
                break
            elif confirm.lower() == "n":
                print("input denied. Try again")
                break
            else:
                print("invalid confermation input. write y or n")

    user_path = Path(inp_string.strip()).expanduser().resolve()
    user_path_data = DATA_DIR / inp_string.strip()
    user_path_data = user_path_data.expanduser().resolve()
    if user_path.is_file():
        with open(user_path, "r", encoding="utf-8") as ppi_file:
            protein_ids = ppi_file.readlines()
            protein_ids = [pid.strip() for pid in protein_ids]
    elif user_path_data.is_file():
        with open(user_path_data, "r", encoding="utf-8") as ppi_file:
            protein_ids = ppi_file.readlines()
            protein_ids = [pid.strip() for pid in protein_ids]
    else:
        protein_ids = inp_string.split()

    return protein_ids


def interaction_retrival():
    """Complete pipeline from input to protein interaction data

    Retrives the protein interaction data of interest from
    the STRING database.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the interaction data between the proteins
    """



def main(): # only used for testing. Delete later
    # genes = ['TP53', 'BRCA1', 'FANCD1', 'FANCL', "9606.ENSP00000497910"]
    # string_ids = stringdb.get_string_ids(genes)
    # print(string_ids)
    # enrichment_df = stringdb.get_enrichment(string_ids.stringId)
    # # print(enrichment_df.loc[enrichment_df["fdr"] == min(enrichment_df["fdr"])]["description"].values[0])
    # # print(enrichment_df)
    # ppi_df = stringdb.get_ppi_enrichment(string_ids["stringId"])
    # network_df = stringdb.get_network(string_ids["stringId"])
    # # print(ppi_df)
    # print(network_df)

    print(read_input())

if __name__ == "__main__": # only used for testing. Delete later
    main()
    pass