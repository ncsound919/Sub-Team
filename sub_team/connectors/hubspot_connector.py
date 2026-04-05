"""
HubSpot Connector — fetches sales metrics from the HubSpot CRM API.

Returns a normalised dictionary of sales-domain parameters suitable
for use in ``BusinessProblem.parameters``.

Environment variables
---------------------
    HUBSPOT_API_KEY  — HubSpot private app access token.  If absent,
                       ``.fetch()`` returns ``None``.

Graceful degradation
--------------------
- Missing API key -> returns None
- Network / API error -> logs warning, returns None
- Partial data -> returns what is available, missing fields omitted
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

_log = logging.getLogger(__name__)


class HubSpotConnector:
    """
    Fetches sales metrics from the HubSpot CRM API.

    Usage::

        connector = HubSpotConnector()
        data = connector.fetch()
        if data is not None:
            problem.parameters.update(data)
    """

    BASE_URL = "https://api.hubapi.com/crm/v3"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        api_key : str, optional
            HubSpot access token.  Falls back to ``HUBSPOT_API_KEY`` env var.
        """
        self._api_key = api_key or os.environ.get("HUBSPOT_API_KEY")

    def fetch(self) -> Optional[Dict[str, float]]:
        """
        Fetch sales metrics from HubSpot.

        Returns
        -------
        dict or None
            Normalised sales parameter dict with keys like
            ``pipeline_value_usd``, ``win_rate_pct``, ``avg_deal_size_usd``,
            ``avg_sales_cycle_days``.  Returns ``None`` if the API key is
            absent or the API call fails.
        """
        if not self._api_key:
            _log.debug("HUBSPOT_API_KEY not set; skipping HubSpot fetch.")
            return None

        try:
            import requests
        except ImportError:
            _log.warning("requests library not installed; cannot fetch from HubSpot.")
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        result: Dict[str, float] = {}

        # ── Fetch deals ─────────────────────────────────────────────────
        try:
            resp = requests.get(
                f"{self.BASE_URL}/objects/deals",
                headers=headers,
                params={
                    "limit": 100,
                    "properties": "amount,dealstage,closedate,createdate",
                },
                timeout=15,
            )
            resp.raise_for_status()
            deals_data = resp.json()
            deals = deals_data.get("results", [])

            open_pipeline = 0.0
            closed_won_count = 0
            closed_won_total = 0.0
            total_deals = len(deals)
            cycle_days_sum = 0.0
            cycle_days_count = 0

            for deal in deals:
                props = deal.get("properties", {})
                amount = float(props.get("amount", 0) or 0)
                stage = (props.get("dealstage", "") or "").lower()

                if "closedwon" in stage or "closed won" in stage:
                    closed_won_count += 1
                    closed_won_total += amount

                    # Calculate cycle days
                    create_date = props.get("createdate")
                    close_date = props.get("closedate")
                    if create_date and close_date:
                        try:
                            from datetime import datetime
                            fmt = "%Y-%m-%dT%H:%M:%S"
                            # Handle various date formats
                            cd = create_date[:19]
                            cld = close_date[:19]
                            d_create = datetime.strptime(cd, fmt)
                            d_close = datetime.strptime(cld, fmt)
                            days = (d_close - d_create).days
                            if days >= 0:
                                cycle_days_sum += days
                                cycle_days_count += 1
                        except (ValueError, TypeError):
                            pass
                elif "closedlost" not in stage and "closed lost" not in stage:
                    # Open deal
                    open_pipeline += amount

            result["pipeline_value_usd"] = open_pipeline

            if total_deals > 0:
                result["win_rate_pct"] = (closed_won_count / total_deals) * 100

            if closed_won_count > 0:
                result["avg_deal_size_usd"] = closed_won_total / closed_won_count

            if cycle_days_count > 0:
                result["avg_sales_cycle_days"] = cycle_days_sum / cycle_days_count

        except Exception:
            _log.debug("Failed to fetch HubSpot deals.", exc_info=True)

        return result if result else None
