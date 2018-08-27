import argparse
import logging
import os
import gzip
import unicodedata
import sys

from traitlets.config.loader import KeyValueConfigLoader
from traitlets.config.loader import PyFileConfigLoader


class OptionPerLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []
        return arg_line.split()


def ensure_dir(p):
    parent = os.path.dirname(p)
    if not os.path.exists(parent):
        os.makedirs(parent)


def tokens_to_sent(tokens, sent_start):
    sent = ""

    for token, span in tokens:
        if span[0] > len(sent) + sent_start:
            padding = ' ' * (span[0] - len(sent) - sent_start)
            sent += padding
        sent += token
    return sent


def find_by_id(folder, docid):
    for filename in os.listdir(folder):
        if filename.startswith(docid):
            return os.path.join(folder, filename)


def rm_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def set_basic_log(log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)


def set_file_log(log_file, log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, filename=log_file)


def basic_console_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)


def load_command_line_config(args):
    cl_loader = KeyValueConfigLoader()
    return cl_loader.load_config(args)


def load_all_config(args):
    cl_conf = load_command_line_config(args[2:])
    conf = PyFileConfigLoader(args[1]).load_config()
    conf.merge(cl_conf)
    return conf


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if
    unicodedata.category(chr(i)).startswith('P')
)


def remove_punctuation(text):
    return text.translate(tbl)


def get_env(var_name):
    if var_name not in os.environ:
        raise KeyError("Please supply the directory as environment "
                       "variable: {}".format(var_name))
    else:
        return os.environ[var_name]
