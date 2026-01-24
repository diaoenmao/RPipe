import os, sys

# Resolve project root (the folder containing `src/`)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src = os.path.join(_root, "src")

# Add `src` to sys.path if not already present
if _src not in sys.path:
    sys.path.insert(0, _src)
