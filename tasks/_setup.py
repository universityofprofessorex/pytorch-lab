"""
Decides if vendor bundles are used or not.
Setup python path accordingly.
"""

import os.path
import sys

# -----------------------------------------------------------------------------
# DEFINES:
# -----------------------------------------------------------------------------
HERE = os.path.dirname(__file__)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS:
# -----------------------------------------------------------------------------
def setup_path_for_bundle(bundle_path, pos=0):
    if os.path.exists(bundle_path):
        syspath_insert(pos, os.path.abspath(bundle_path))
        return True
    return False


def syspath_insert(pos, path):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(pos, path)


def syspath_append(path):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.append(path)
