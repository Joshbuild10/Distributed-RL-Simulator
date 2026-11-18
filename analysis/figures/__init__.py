"""Figure and table definitions. Importing this package registers all of them (spec.REGISTRY), in
dashboard order: validation, feasibility, centralisation, lengths, appendix tables."""
from . import validation, feasibility, centralisation, lengths, tables  # noqa: F401
