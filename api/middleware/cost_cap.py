"""Daily cost-cap ASGI middleware.

Applies to /ingest and /query endpoints. Uses fixed-rate per-request cost
estimates to approximate daily OpenAI spend. Counter resets at midnight UTC.
In a single-worker process (HF Spaces) an in-memory counter is sufficient.
"""

from datetime import date

from fastapi.responses import JSONResponse

# Rough USD cost per request (conservative estimates)
_COST_USD: dict[str, float] = {
    "/ingest": 0.05,   # ~500 chunks × text-embedding-3-small
    "/query": 0.003,   # gpt-4o-mini ~500 in + 200 out tokens
}
_USD_TO_INR: float = 84.0

_daily_spend_inr: float = 0.0
_reset_date: date = date.today()


def _reset_if_new_day() -> None:
    global _daily_spend_inr, _reset_date
    today = date.today()
    if today != _reset_date:
        _daily_spend_inr = 0.0
        _reset_date = today


class DailyCostCapMiddleware:
    """Rejects /ingest and /query with 429 when the daily INR cap is exceeded."""

    def __init__(self, app, daily_cap_inr: int = 50) -> None:
        self._app = app
        self._cap_inr = daily_cap_inr

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        cost_key = "/ingest" if path == "/ingest" else "/query" if path == "/query" else None

        if cost_key is not None:
            global _daily_spend_inr
            _reset_if_new_day()
            if _daily_spend_inr >= self._cap_inr:
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Daily cost cap of ₹{self._cap_inr} reached. "
                            "Service resets at midnight UTC."
                        )
                    },
                )
                await response(scope, receive, send)
                return
            _daily_spend_inr += _COST_USD[cost_key] * _USD_TO_INR

        await self._app(scope, receive, send)
