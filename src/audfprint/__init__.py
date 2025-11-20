# coding=utf-8
"""Landmark-based audio fingerprinting library."""

from audfprint.core.analyzer import Analyzer
from audfprint.core.hash_table import DatabaseType, HashTable
from audfprint.core.matcher import Matcher

__version__ = "20251119"

__all__ = [
    "Analyzer",
    "Matcher",
    "HashTable",
    "DatabaseType",
    "__version__",
]
