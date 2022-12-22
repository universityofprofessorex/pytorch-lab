"""
supporting task functions
"""

# Passing an environment variable containing unicode literals to a subprocess
# on Windows and Python2 raises a TypeError. Since there is no unicode
# string in this script, we don't import unicode_literals to avoid the issue.
from __future__ import absolute_import, division, print_function

import contextlib
import copy
import errno
import logging
import os
import select
import shutil
from shutil import which
import stat
import subprocess
import sys

from invoke import Exit

COLOR_WARNING = "red"
COLOR_DANGER = "red"
COLOR_SUCCESS = "green"
COLOR_CAUTION = "yellow"
COLOR_STABLE = "blue"


# from keyrings.cryptfile.cryptfile import CryptFileKeyring
ENV_WHITELIST = [
    "POSTGRES_DB",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "DB_CONNECTION",
    "SECRET_KEY",
    "BETTER_EXCEPTIONS",
    "USE_LOCAL_DB_FOR_TEST",
    "APIURL",
]

# from tasks.core import clean, execute_sql
from .ml_logger import get_logger  # noqa: E402

# from pytorch_lab.utils.parser import get_domain_from_fqdn

LOGGER = get_logger(__name__, provider="Invoke utils", level=logging.INFO)


def _check_exe(exe):
    """Look for executable"""
    exe_path = which(exe)
    if not exe_path:
        msg = "Couldn't find `{}`.\n".format(exe)
        raise Exit(msg)


# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
def is_venv():
    """Check to see if we are currently in a virtual environment

    Returns:
        [type] -- [description]
    """
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def get_compose_env(c, loc="docker", name=None):
    """
    The variables are looked up in this priority: invoke.yaml dev.env variables, environment variables,
    If `name` is provided, it will look up for `name` in os.environ
        then it will try to load it if VAULT_{name} is defined
    Vault variables
    """
    env = copy.copy(c[loc]["env"])
    env["VERSION"] = "0.0.1"
    env["NAME"] = c["name"]

    # environment variables have priority over what's inside invoke.yaml
    for key in env:
        if key in os.environ:
            env[key] = "{}".format(os.getenv(key))

    for key in list(env):
        # print("ENV: {}".format(key))
        if key not in ENV_WHITELIST:
            del env[key]

    if name:
        if name in env:
            return env[name]

    # print("type(ENV) = {}".format(type(env)))

    return env


def confirm():
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Are you sure you want to execute this command [Y/N]? ").lower()
    return answer == "y"


# SOURCE: https://git.corp.adobe.com/Evergreen/pops/blob/master/src/pops/tasks/libs/utils.py
# def confirm(ctx, msg="Are you sure you want to execute this command?"):
#     """
#     Ask user to enter Y or N (case-insensitive).
#     :return: True if the answer is Y.
#     :rtype: bool
#     """
#     if not ctx.ask_confirmation:
#         return True
#     answer = None
#     acceptable_answers = ["y", "n", ""] if ctx.default_confirmation else ["y", "n"]
#     display_options = "[Y/n]" if ctx.default_confirmation else "[y/n]"
#     while answer not in acceptable_answers:
#         answer = input(f"{msg}\n{display_options} ").lower()
#         if ctx.default_confirmation and answer == "":
#             answer = "y"
#     return answer == "y"

# def get_keyring():
#     """
#     Build a CryptFileKeyring object
#     """
#     if os.environ.get('KEYRING_PASS', None) is None:
#         logger.error('KEYRING_PASS env variable not found')
#         raise RuntimeError('KEYRING_PASS environment variable was not defined')
#     cfk = CryptFileKeyring()
#     cfk.keyring_key = os.environ.get('KEYRING_PASS')
#     # this is needed for the keyring command
#     cfk.set_password('keyring-setting', 'password reference', 'password reference value')
#     return cfk

# def get_secret(env_var_name, message):
#     """
#     Ask the user to provide a secret.
#     If the environment variable with the name of `env_var_name` exists, the secret is set to that value
#     """
#     if os.environ.get(env_var_name, None) is None:
#         secret = getpass(message)
#         if not secret:
#             logger.error('Empty input.')
#             sys.exit(1)
#     else:
#         secret = os.environ.get(env_var_name)
#     return secret

# SOURCE: https://github.com/bossjones/pocketsphinx-build/blob/master/pocketsphinx_build/build.py

# SOURCE: https://github.com/ARMmbed/mbed-cli/blob/f168237fabd0e32edcb48e214fc6ce2250046ab3/test/util.py
# Process execution
class ProcessException(Exception):
    pass


class Console:  # pylint: disable=too-few-public-methods

    quiet = False

    @classmethod
    def message(cls, str_format, *args):
        if cls.quiet:
            return

        if args:
            print(str_format % args)
        else:
            print(str_format)

        # Flush so that messages are printed at the right time
        # as we use many subprocesses.
        sys.stdout.flush()


def pquery(command, stdin=None, **kwargs):
    # SOURCE: https://github.com/ARMmbed/mbed-cli/blob/f168237fabd0e32edcb48e214fc6ce2250046ab3/test/util.py
    # Example:
    print(" ".join(command))
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    )
    stdout, _ = proc.communicate(stdin)

    if proc.returncode != 0:
        raise ProcessException(proc.returncode)

    return stdout.decode("utf-8")


# Directory navigation
@contextlib.contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


def scm(dir=None):
    if not dir:
        dir = os.getcwd()

    if os.path.isdir(os.path.join(dir, ".git")):
        return "git"
    elif os.path.isdir(os.path.join(dir, ".hg")):
        return "hg"


def _popen(cmd_arg):
    devnull = open("/dev/null")
    cmd = subprocess.Popen(cmd_arg, stdout=subprocess.PIPE, stderr=devnull, shell=True)
    retval = cmd.stdout.read().strip()
    err = cmd.wait()
    cmd.stdout.close()
    devnull.close()
    if err:
        raise RuntimeError("Failed to close %s stream" % cmd_arg)
    return retval


def _popen_stdout(cmd_arg, cwd=None):
    # if passing a single string, either shell mut be True or else the string must simply name the program to be executed without specifying any arguments
    cmd = subprocess.Popen(
        cmd_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        bufsize=4096,
        shell=True,
    )
    Console.message("BEGIN: {}".format(cmd_arg))
    # output, err = cmd.communicate()

    for line in iter(cmd.stdout.readline, b""):
        # Print line
        _line = line.rstrip()
        Console.message(">>> {}".format(_line.decode("utf-8")))

    Console.message("END: {}".format(cmd_arg))


# Higher level functions
def remove(path):
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(path, onerror=remove_readonly)


def move_f(src, dst):
    shutil.move(src, dst)


def copy_f(src, dst):
    shutil.copytree(src, dst)


def git_clone(repo_url, dest, sha="master"):
    # First check if folder exists
    if not os.path.exists(dest):
        # check if folder is a git repo
        if scm(dest) != "git":
            clone_cmd = "git clone {repo} {dest}".format(repo=repo_url, dest=dest)
            _popen_stdout(clone_cmd)

            # CD to directory
            with cd(dest):
                checkout_cmd = "git checkout {sha}".format(sha=sha)
                _popen_stdout(checkout_cmd)


def whoami():
    whoami = _popen("who")
    return whoami


def environ_append(key, value, separator=" ", force=False):
    old_value = os.environ.get(key)
    if old_value is not None:
        value = old_value + separator + value
    os.environ[key] = value


def environ_prepend(key, value, separator=" ", force=False):
    old_value = os.environ.get(key)
    if old_value is not None:
        value = value + separator + old_value
    os.environ[key] = value


def environ_remove(key, value, separator=":", force=False):
    old_value = os.environ.get(key)
    if old_value is not None:
        old_value_split = old_value.split(separator)
        value_split = [x for x in old_value_split if x != value]
        value = separator.join(value_split)
    os.environ[key] = value


def environ_set(key, value):
    os.environ[key] = value


def environ_get(key):
    return os.environ.get(key)


def path_append(value):
    if os.path.exists(value):
        environ_append("PATH", value, ":")


def path_prepend(value, force=False):
    if os.path.exists(value):
        environ_prepend("PATH", value, ":", force)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def dump_env_var(var):
    Console.message("Env Var:{}={}".format(var, os.environ.get(var, "<EMPTY>")))
