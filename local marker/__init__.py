"""Stage-4 local-marker computations (OBC Haldane hexagon defects)."""
from .model import HaldaneOBC
from .geometry import FigureRegions
from .marker import exact_marker, marker_from_occupied
__all__ = ['HaldaneOBC', 'FigureRegions', 'exact_marker', 'marker_from_occupied']
