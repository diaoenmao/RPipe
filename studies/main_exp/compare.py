"""Manual full/partial curve readout; normal launch uses the Study process hook."""

import sys

from process.curves import main


if __name__ == '__main__':
    sys.exit(main())
