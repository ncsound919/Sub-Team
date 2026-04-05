"""
Stripe Connector — fetches financial metrics from the Stripe API.

Returns a normalised dictionary of finance-domain parameters suitable
for use in ``BusinessProblem.parameters``.

Environment variables
---------------------
    STRIPE_API_KEY  — Stripe secret key (sk_...).  If absent, ``.fetch()``
                      returns ``None``.

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


class StripeConnector:
    """
    Fetches financial metrics from the Stripe API.

    Usage::

        connector = StripeConnector()
        data = connector.fetch()
        if data is not None:
            problem.parameters.update(data)
    """

    BASE_URL = "https://api.stripe.com/v1"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        api_key : str, optional
            Stripe API key.  Falls back to ``STRIPE_API_KEY`` env var.
        """
        self._api_key = api_key or os.environ.get("STRIPE_API_KEY")

    def fetch(self) -> Optional[Dict[str, float]]:
        """
        Fetch financial metrics from Stripe.

        Returns
        -------
        dict or None
            Normalised finance parameter dict with keys like ``mrr_usd``,
            ``arr_usd``, ``cash_balance_usd``.  Returns ``None`` if the
            API key is absent or the API call fails.
        """
        if not self._api_key:
            _log.debug("STRIPE_API_KEY not set; skipping Stripe fetch.")
            return None

        try:
            import requests
        except ImportError:
            _log.warning("requests library not installed; cannot fetch from Stripe.")
            return None

        headers = {"Authorization": f"Bearer {self._api_key}"}
        result: Dict[str, float] = {}

        # ── Fetch subscriptions to compute MRR ──────────────────────────
        try:
            resp = requests.get(
                f"{self.BASE_URL}/subscriptions",
                headers=headers,
                params={"status": "active", "limit": 100},
                timeout=15,
            )
            resp.raise_for_status()
            subs_data = resp.json()

            total_mrr_cents = 0
            for sub in subs_data.get("data", []):
                for item in sub.get("items", {}).get("data", []):
                    plan = item.get("plan", {}) or item.get("price", {})
                    amount = plan.get("amount", 0)
                    interval = plan.get("interval", "month")
                    # Normalise to monthly
                    if interval == "year":
                        amount = amount / 12
                    elif interval == "week":
                        amount = amount * 4.33
                    elif interval == "day":
                        amount = amount * 30
                    total_mrr_cents += amount

            mrr_usd = total_mrr_cents / 100.0
            result["mrr_usd"] = mrr_usd
            result["arr_usd"] = mrr_usd * 12
        except Exception:
            _log.debug("Failed to fetch Stripe subscriptions.", exc_info=True)

        # ── Fetch balance for cash_balance ──────────────────────────────
        try:
            resp = requests.get(
                f"{self.BASE_URL}/balance",
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            balance_data = resp.json()

            total_available = 0
            for entry in balance_data.get("available", []):
                total_available += entry.get("amount", 0)

            result["cash_balance_usd"] = total_available / 100.0
        except Exception:
            _log.debug("Failed to fetch Stripe balance.", exc_info=True)

        return result if result else None
