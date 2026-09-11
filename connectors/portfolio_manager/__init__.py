"""This venue behind the broker interface a portfolio manager already calls.

`ArenaBroker` implements the same nine methods and one property that an
Alpaca-shaped `BrokerClient` declares, so an allocator, a sizing model and a
bot loop written against that base class trade here without a branch.
"""

from connectors.portfolio_manager.broker import (
    AccountInfo,
    ArenaBroker,
    BrokerOrder,
    OrderResult,
    Position,
)

__all__ = ["ArenaBroker", "AccountInfo", "BrokerOrder", "OrderResult", "Position"]
