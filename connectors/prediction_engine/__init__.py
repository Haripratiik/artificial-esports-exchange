"""This venue, in the shape a prediction-market engine already speaks.

A Kalshi-shaped engine keeps its venue behind three files and talks to
everything else through one data model. This package is the same three files
for this exchange, so such an engine can trade here by naming a different
venue rather than by growing a branch.

Nothing here imports the exchange. It speaks to the published REST API over
HTTP exactly as an outside client would, which is the property that makes any
audit built on it worth running: a test that shares the venue's own model of
what a coherent price is cannot falsify that model.
"""

from connectors.prediction_engine.models import Event, Market, Series, Side, Venue

__all__ = ["Event", "Market", "Series", "Side", "Venue"]
