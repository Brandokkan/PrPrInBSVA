"""Stand alone script running the protein-protein interaction analysis

This script is the entry point of the project. Every parameter of the analysis
is given on the command line, so nothing has to be changed inside the source
code to run it on different data.

Since Python puts the folder of the running script at the front of the import
path, this script works from any working directory, wherever the project is
unpacked.

Examples
-------
    python run_prprin.py data/test_big --hub-method consensus topk --top-k 5

    python run_prprin.py "TP53 BRCA1 BRCA2 ATM" --required-score 700 --outdir results

    python run_prprin.py --help
"""

import sys

from prprin.cli import main

if __name__ == "__main__":
    sys.exit(main())
