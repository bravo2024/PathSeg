"""Structured logging for PathSeg."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logger(
    name: str = "pathseg",
    level: str = "INFO",
    log_file: str | None = "logs/training.log",
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with file and optional console handler.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR).
    log_file : str or None
        Path to log file.  Parent directories are created automatically.
        If None, no file handler is added.
    console : bool
        If True, also log to stderr.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(_FORMAT)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def get_logger(name: str = "pathseg") -> logging.Logger:
    """Get an existing logger or create a default one."""
    return logging.getLogger(name)
