"""
Connectors — optional data-ingestion layer for BusinessAgent.

Each connector fetches live data from an external API and normalises it
into the parameter dictionary expected by BusinessAgent domains.

Connectors are **optional**: if the required API key is absent or the
API call fails, ``.fetch()`` returns ``None`` and no exception is raised.
"""

from .stripe_connector import StripeConnector
from .hubspot_connector import HubSpotConnector

__all__ = [
    "StripeConnector",
    "HubSpotConnector",
]
