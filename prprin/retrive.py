import stringdb
from pathlib import Path
from pandas import DataFrame, concat, unique, read_csv

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability


def read_input(verbose = False):
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
            confirm = input("\ndo you confirm your input (y/n): ")
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

    if verbose:
        print() # spacing
        print(protein_ids)

    return protein_ids


def interaction_retrival(input_proteins=None, verbose = False):
    """Complete pipeline from input to protein interaction data

    Retrives the protein interaction data of interest from
    the STRING database.

    Parameters
    -------
    input_proteins: str or None
        The identifiers of the protein separated by spaces or the name
        of a file in the 'data' folder containing the identifiers (one
        identifier per line)

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the interaction data between the proteins
    """

    # input check
    if not (isinstance(input_proteins, str) or input_proteins is None):
        raise TypeError("input_proteins must be a string or be left empty")

    if isinstance(input_proteins, str):   # case were proteins were supplied directlly to the function
        data_file_path = DATA_DIR / input_proteins
        if data_file_path.is_file():
            with open(data_file_path, "r", encoding="utf-8") as ppi_file:
                    protein_ids = ppi_file.readlines()
                    protein_ids = [pid.strip() for pid in protein_ids]
        else:
            protein_ids = input_proteins.split()
        string_ids = stringdb.get_string_ids(protein_ids)
        print("proteins found:")
        print(string_ids)
    else:   # case for the manual input of proteins from the console
        # user input check
        checking_string_id = True
        while checking_string_id:
            protein_ids = read_input()
            string_ids = stringdb.get_string_ids(protein_ids)
            print() # spacing
            print(string_ids)
            while True:
                are_string_ok = input("\nare the above string ids the intended ones (y/n)?: ")
                if are_string_ok.lower() == "y":
                    print("string ids confirmed")
                    checking_string_id = False
                    break
                elif are_string_ok.lower() == "n":
                    print("string ids rejected. re-insert the identifiers")
                    break
                else:
                    print("invalid character. please insert y or n")

    # transform into interaction data
    network_df = stringdb.get_network(string_ids["stringId"])
    if verbose:
        print() # spacing
        print(network_df)
    return network_df


def get_unique_proteins(ppi_df, verbose=False):
    """Get the unique symbols and string ids from protein interaction data

    This function obtains the unique proteins (symbol and string id) from
    the protein-protein interaction DataFrame outputed by interaction_retrival().

    Parameters
    -------
    ppi_df: pandas.DataFrame
        DataFrame that contains protein-protein interaction data. like the one
        outputed by interaction_retrival()

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the unique proteins inside ppi_df.
        Made of two columns (stringId and symbol).
    """
    #input check
    if not isinstance(ppi_df, DataFrame):
        raise TypeError("ppi_df must be a pandas DataFrame")

    # get unique proteins
    a_proteins = ppi_df[["stringId_A", "preferredName_A"]]
    a_proteins = a_proteins.rename(columns={"stringId_A": "stringId", "preferredName_A":"symbol"})
    b_proteins = ppi_df[["stringId_B", "preferredName_B"]]
    b_proteins = b_proteins.rename(columns={"stringId_B": "stringId", "preferredName_B":"symbol"})
    proteins = concat([a_proteins, b_proteins], ignore_index=True)
    proteins = proteins.drop_duplicates(ignore_index=True)

    if verbose:
        print(proteins)
    return proteins

# TODO: consider adding deletion mode
class RetrivePPI:
    def get_raw_data(self, input_proteins=None, mode="replace"):
        """ Asks the user to input the proteins of interest and assigns the
        protein-protein interactiond ata to the relative container

        This method first asks the user to input the protein of interest
        manually or through a file and then assigns the protein-protein
        interaction data to the 'network' container and the unique proteins
        in the network to the 'proteins' container.

        Parameters
        -------
        mode: str
            must be one of 'add', 'a', 'replace' or 'r'.

            replace mode: replaces the current protein-protein interaction
            data (if any) with the inputed one.

        input_proteins: str or None
                The identifiers of the protein separated by spaces or the name
                of a file in the 'data' folder containing the identifiers (one
                identifier per line)
        """
        #input check
        add_str = ("add", "a")
        replace_str = ("replace", "r")
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in add_str + replace_str:
            raise ValueError(f"mode must be one of 'add', 'a', 'replace' or 'r'. Got {mode} instead.")

        # protein data retrival and asignment
        network_df = interaction_retrival(input_proteins)
        unique_proteins = get_unique_proteins(network_df)
        if mode in replace_str: # replace mode
            self.proteins = unique_proteins
            self.network = network_df
        elif mode in add_str:   # addition mode
            pass    # TODO: consider implement addition mode


        return self # enables chaning multiple methods togheter in the final object


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

    # ppi_df_path = DATA_DIR / "ex_ppi_df.csv"
    # ppi_df_path.resolve()
    # interaction_retrival(True).to_csv(ppi_df_path)

    get_unique_proteins(read_csv(DATA_DIR / "ex_ppi_df.csv"), True)

if __name__ == "__main__": # only used for testing. Delete later
    main()
    pass