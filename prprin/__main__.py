"""Entry point that makes the package runnable with 'python -m prprin'"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
