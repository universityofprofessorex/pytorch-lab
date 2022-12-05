#!/usr/bin/env python3

# SOURCE: https://blog.bartab.fr/fastapi-logging-on-the-fly/
from __future__ import annotations

from typing import Any, List, Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel

import collections

try:  # python 3
    from collections import abc
except ImportError:  # python 2
    import collections as abc

import concurrent.futures
from datetime import datetime
import functools
import gc
import inspect
import logging
from logging import Logger, LogRecord
import os
from pathlib import Path
from pprint import pformat
# import slack
import sys
from time import process_time
from types import FrameType

from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Union, cast

from loguru import logger
from loguru._defaults import LOGURU_FORMAT

class LoggerPatch(BaseModel):
    name: str
    level: str



class LoggerModel(BaseModel):
    name: str
    level: Optional[int]
    # children: Optional[List["LoggerModel"]] = None
    # fixes: https://github.com/samuelcolvin/pydantic/issues/545
    children: Optional[List[Any]] = None
    # children: ListLoggerModel = None


LoggerModel.update_forward_refs()



# SOURCE: https://github.com/joint-online-judge/fastapi-rest-framework/blob/b0e93f0c0085597fcea4bb79606b653422f16700/fastapi_rest_framework/logging.py#L43
def format_record(record: Dict[str, Any]) -> str:
    """
    Custom format for loguru loggers.
    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.
    Example:
    >>> payload = [{"users":[{"name": "Nick", "age": 87, "is_active": True},
    >>>     {"name": "Alex", "age": 27, "is_active": True}], "count": 2}]
    >>> logger.bind(payload=).debug("users payload")
    >>> [   {   'count': 2,
    >>>         'users': [   {'age': 87, 'is_active': True, 'name': 'Nick'},
    >>>                      {'age': 27, 'is_active': True, 'name': 'Alex'}]}]
    """

    format_string = LOGURU_FORMAT
    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string


if TYPE_CHECKING:
    from better_exceptions.log import BetExcLogger
    from loguru._logger import Logger as _Logger

LOGLEVEL_MAPPING = {
    50: "CRITICAL",
    40: "ERROR",
    30: "WARNING",
    20: "INFO",
    10: "DEBUG",
    0: "NOTSET",
}


class InterceptHandler(logging.Handler):
    """
    Intercept all logging calls (with standard logging) into our Loguru Sink
    See: https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """

    loglevel_mapping = {
        logging.CRITICAL: "CRITICAL",
        logging.ERROR: "ERROR",
        logging.FATAL: "FATAL",
        logging.WARNING: "WARNING",
        logging.INFO: "INFO",
        logging.DEBUG: "DEBUG",
        1: "DUMMY",
        0: "NOTSET",
    }

    # from logging import DEBUG
    # from logging import ERROR
    # from logging import FATAL
    # from logging import INFO
    # from logging import WARN
    # https://issueexplorer.com/issue/tiangolo/fastapi/4026
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            # DISABLED 12/10/2021 # level = str(record.levelno)
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = frame.f_back
            # DISABLED 12/10/2021 # frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )

# """ Logging handler intercepting existing handlers to redirect them to loguru """
class LoopDetector(logging.Filter):
    """
    Log filter which looks for repeating WARNING and ERROR log lines, which can
    often indicate that a module is spinning on a error or stuck waiting for a
    condition.

    When a repeating line is found, a summary message is printed and a message
    optionally sent to Slack.
    """

    LINE_HISTORY_SIZE = 50
    LINE_REPETITION_THRESHOLD = 5

    def __init__(self) -> None:
        self._recent_lines: Deque[str] = collections.deque(
            maxlen=self.LINE_HISTORY_SIZE
        )
        self._supressed_lines: collections.Counter = collections.Counter()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True

        self._recent_lines.append(record.getMessage())

        counter = collections.Counter(list(self._recent_lines))
        repeated_lines = [
            line
            for line, count in counter.most_common()
            if count > self.LINE_REPETITION_THRESHOLD
            and line not in self._supressed_lines
        ]

        if repeated_lines:
            for line in repeated_lines:
                self._supressed_lines[line] = self.LINE_HISTORY_SIZE

        for line, count in self._supressed_lines.items():
            self._supressed_lines[line] = count - 1
            # mypy doesn't understand how to deal with collection.Counter's
            # unary addition operator
            self._supressed_lines = +self._supressed_lines  # type: ignore

        # https://docs.python.org/3/library/logging.html#logging.Filter.filter
        # The docs lie when they say that this returns an int, it's really a bool.
        # https://bugs.python.org/issue42011
        # H6yQOs93Cgg
        return True


def get_logger(
    name: str,
    provider: Optional[str] = None,
    level: int = logging.INFO,
    logger: logging.Logger = logger,
) -> logging.Logger:

    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": format_record,
                "level": logging.DEBUG,
                "enqueue": True,
                "diagnose": True,
            },
        
        ],
        
    }

    logger.remove()
    logger.configure(**config)
    
    logger.add(
        sys.stdout,
        format=format_record,
        filter="requests.packages.urllib3.connectionpool",
        level="ERROR",
        enqueue=True,
        diagnose=True,
    )

    logger.add(
        sys.stdout,
        format=format_record,
        filter="handler",
        level="ERROR",
        enqueue=True,
        diagnose=True,
    )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="asyncio",
    #     level="ERROR",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="selenium",
    #     level="ERROR",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="webdriver_manager",
    #     level="ERROR",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="arsenic",
    #     level="DEBUG",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="aiohttp",
    #     level="DEBUG",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="tensorflow",
    #     level="DEBUG",
    #     enqueue=True,
    #     diagnose=True,
    # )
    # logger.add(
    #     sys.stdout,
    #     format=format_record,
    #     filter="keras",
    #     level="ERROR",
    #     enqueue=True,
    #     diagnose=True,
    # )

    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    return logger


# SOURCE: https://github.com/joint-online-judge/fastapi-rest-framework/blob/b0e93f0c0085597fcea4bb79606b653422f16700/fastapi_rest_framework/logging.py#L43
def intercept_all_loggers(level: int = logging.DEBUG) -> None:
    logging.basicConfig(handlers=[InterceptHandler()], level=level)
    logging.getLogger("uvicorn").handlers = []


# SOURCE: https://github.com/jupiterbjy/CUIAudioPlayer/blob/dev_master/CUIAudioPlayer/LoggingConfigurator.py
def get_caller_stack_name(depth=1):
    """
    Gets the name of caller.
    :param depth: determine which scope to inspect, for nested usage.
    """
    return inspect.stack()[depth][3]


# SOURCE: https://github.com/jupiterbjy/CUIAudioPlayer/blob/dev_master/CUIAudioPlayer/LoggingConfigurator.py
def get_caller_stack_and_association(depth=1):
    stack_frame = inspect.stack()[depth][0]
    f_code_ref = stack_frame.f_code

    def get_reference_filter():
        for obj in gc.get_referrers(f_code_ref):
            try:
                if obj.__code__ is f_code_ref:  # checking identity
                    return obj
            except AttributeError:
                continue

    actual_function_ref = get_reference_filter()
    try:
        return actual_function_ref.__qualname__
    except AttributeError:
        return "<Module>"


# https://stackoverflow.com/questions/52715425


def log_caller():
    return f"<{get_caller_stack_name()}>"


def get_lm_from_tree(loggertree: LoggerModel, find_me: str) -> LoggerModel:
    if find_me == loggertree.name:
        LOGGER.debug("Found")
        return loggertree
    else:
        for ch in loggertree.children:
            LOGGER.debug(f"Looking in: {ch.name}")
            i = get_lm_from_tree(ch, find_me)
            if i:
                return i


def generate_tree() -> LoggerModel:
    # pylint: disable=no-member
    # adapted from logging_tree package https://github.com/brandon-rhodes/logging_tree
    rootm = LoggerModel(
        name="root", level=logging.getLogger().getEffectiveLevel(), children=[]
    )
    nodesm = {}
    items = list(logging.root.manager.loggerDict.items())  # type: ignore
    items.sort()
    for name, loggeritem in items:
        if isinstance(loggeritem, logging.PlaceHolder):
            nodesm[name] = nodem = LoggerModel(name=name, children=[])
        else:
            nodesm[name] = nodem = LoggerModel(
                name=name, level=loggeritem.getEffectiveLevel(), children=[]
            )
        i = name.rfind(".", 0, len(name) - 1)  # same formula used in `logging`
        if i == -1:
            parentm = rootm
        else:
            parentm = nodesm[name[:i]]
        parentm.children.append(nodem)
    return rootm


# SMOKE-TESTS
if __name__ == "__main__":
    from logging_tree import printout

    LOGGER = get_logger("Logger Smoke Tests", provider="Logger")
    intercept_all_loggers()

    def dump_logger_tree():
        rootm = generate_tree()
        LOGGER.debug(rootm)

    def dump_logger(logger_name: str):
        LOGGER.debug(f"getting logger {logger_name}")
        rootm = generate_tree()
        lm = get_lm_from_tree(rootm, logger_name)
        return lm

    LOGGER.info("TESTING TESTING 1-2-3")
    printout()

    # <--""
    #    Level NOTSET so inherits level NOTSET
    #    Handler <InterceptHandler (NOTSET)>
    #      Formatter fmt='%(levelname)s:%(name)s:%(message)s' datefmt=None
    #    |
    #    o<--"asyncio"
    #    |   Level NOTSET so inherits level NOTSET
    #    |
    #    o<--[concurrent]
    #        |
    #        o<--"concurrent.futures"
    #            Level NOTSET so inherits level NOTSET
    # [INFO] Logger: TESTING TESTING 1-2-3
