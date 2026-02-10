from dataclasses import dataclass

@dataclass
class PeakLocation:
    """
    Contains the location of an XPS at a specific point in time
    and the uncertainty in the peak position
    """
    value: float
    uncertainty: float