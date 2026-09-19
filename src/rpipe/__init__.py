"""RPipe — research execution substrate.

- ``rpipe.structure`` — control, four layers, artifact, make
- ``rpipe.flow`` — study execution; ``python -m rpipe`` is the cli
"""

import sys

# On Windows, Conda NumPy and pip Torch may bundle different libiomp5md.dll files.
# Loading NumPy first lets both use one runtime instead of enabling the unsafe
# KMP_DUPLICATE_LIB_OK workaround.
if sys.platform == 'win32':
    import numpy  # noqa: F401

__version__ = '0.3.0'
