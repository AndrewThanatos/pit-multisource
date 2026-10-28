"""Point-in-time multi-source dataset builder and article reproduction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pit_multisource")
except PackageNotFoundError:
    __version__ = "0+unknown"
