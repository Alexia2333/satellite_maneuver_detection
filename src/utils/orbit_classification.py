
"""
Orbit classification helpers.

Provides a simple rule-based classifier to separate GEO and LEO using
orbital period (in minutes) or semi-major axis (in km), with optional
fallbacks based on satellite name patterns. This is intentionally
transparent and adjustable.
"""
from typing import Optional, Literal, Dict

def classify_orbit(
    period_minutes: Optional[float] = None,
    sma_km: Optional[float] = None,
    satellite_name: Optional[str] = None,
    geo_period_tol: float = 45.0
) -> Literal["GEO", "LEO", "UNKNOWN"]:
    """
    Classify orbit as GEO or LEO (or UNKNOWN).

    Parameters
    ----------
    period_minutes : Optional[float]
        Orbital period in minutes. GEO is ~1436 minutes. A tolerance is applied.
    sma_km : Optional[float]
        Semi-major axis in kilometers. GEO is ~42164 km; LEO typically < 10000 km.
    satellite_name : Optional[str]
        Optional name hint; may help on tie-breaking.
    geo_period_tol : float
        Tolerance around 1436 minutes to be considered GEO.

    Returns
    -------
    Literal["GEO", "LEO", "UNKNOWN"]
    """
    GEO_PERIOD = 1436.0
    GEO_SMA = 42164.0

    if period_minutes is not None:
        if abs(period_minutes - GEO_PERIOD) <= geo_period_tol:
            return "GEO"
        if period_minutes < 200.0 or period_minutes < 700.0:
            return "LEO"

    if sma_km is not None:
        if abs(sma_km - GEO_SMA) < 1500.0:
            return "GEO"
        if sma_km < 10000.0:
            return "LEO"

    if satellite_name:
        name = satellite_name.upper()
        # Heuristics, extend as needed
        if any(k in name for k in ["FY", "FENGYUN", "GOES", "HIMAWARI", "METEOSAT"]):
            return "GEO"

    return "UNKNOWN"
