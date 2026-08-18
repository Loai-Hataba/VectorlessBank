"""
loaders/base_loader.py

WHAT THIS FILE DOES
--------------------
Defines the abstract "shape" every loader must follow: a `load()`
method that takes no arguments and returns a `list[Record]`.

WHY IT EXISTS
-------------
This is a contract, not working code. Python won't let you create an
instance of BaseLoader directly (it will raise an error), only its
subclasses (CardsLoader, OffersLoader, CampaignsLoader). This
guarantees that anyone writing a new loader in the future -- for
example, if you add a "General Information" document later -- knows
exactly what method to implement, and the rest of the pipeline can
call `.load()` on ANY loader without knowing which one it is.

This is the classic "program to an interface, not an implementation"
principle, and it's what makes it safe to add/remove/replace a data
source without touching any other file.

INPUTS  : none (abstract class)
OUTPUTS : none (abstract class) -- subclasses return list[Record]
"""

from abc import ABC, abstractmethod
from loaders.record import Record


class BaseLoader(ABC):
    """Every concrete loader (CardsLoader, OffersLoader, ...) inherits this."""

    @abstractmethod
    def load(self) -> list[Record]:
        """
        Read the raw source file and return a list of normalized Record objects.
        Must be implemented by every subclass.
        """
        raise NotImplementedError
