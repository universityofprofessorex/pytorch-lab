# pylint: disable=wrong-import-position, wrong-import-order, invalid-name
"""
Invoke build script.
Show all tasks with::
    invoke -l
.. seealso::
    * http://pyinvoke.org
    * https://github.com/pyinvoke/invoke
"""
###############################################################################
# Catch exceptions and go into ipython/ipdb
# import sys

# from IPython.core.debugger import Tracer  # noqa
# from IPython.core import ultratb

# sys.excepthook = ultratb.FormattedTB(
#     mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__
# )
###############################################################################


import logging
from invoke import Collection, Context, Config
from invoke import task
from .constants import ROOT_DIR, PROJECT_BIN_DIR, DATA_DIR, SCRIPT_DIR

from . import local
from . import ci

from .ml_logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__, provider="Invoke", level=logging.INFO)

LOGGER.disable("invoke")

ns = Collection()
ns.add_collection(local)
ns.add_collection(ci)
