"""Fetch fund-performance data for every AMFI subcategory from CRISIL."""

from __future__ import annotations

import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUBCATEGORY_ENDPOINT = "https://polling.crisil.com/gateway/pollingsebi/api/amfi/getsubcategory"
FUND_PERFORMANCE_ENDPOINT = (
    "https://polling.crisil.com/gateway/pollingsebi/api/amfi/fundperformance"
)
REPORT_DATE = "24-Aug-2026"
MAX_RETRY_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 2


def post_json(endpoint: str, payload: dict[str, Any], timeout: float = 30) -> Any:
    """POST a JSON payload to CRISIL and return its decoded JSON response.

    Raises:
        RuntimeError: If CRISIL returns an HTTP error, unreachable endpoint,
            or a non-JSON response.
    """
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://polling.crisil.com",
            "Referer": "https://polling.crisil.com",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            )
        },
    )

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode(
                    response.headers.get_content_charset() or "utf-8"
                )
            break
        except HTTPError as error:
            if attempt == MAX_RETRY_ATTEMPTS:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"CRISIL returned HTTP {error.code}: {detail}") from error

            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 0
            except ValueError:
                delay = 0
            delay = max(delay, INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
            print(
                f"CRISIL returned HTTP {error.code}; retrying in {delay:g} seconds "
                f"(attempt {attempt}/{MAX_RETRY_ATTEMPTS})...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        except URLError as error:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise RuntimeError(f"Could not reach the CRISIL endpoint: {error.reason}") from error

            delay = INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
            print(
                f"Could not reach CRISIL; retrying in {delay:g} seconds "
                f"(attempt {attempt}/{MAX_RETRY_ATTEMPTS})...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("CRISIL returned a non-JSON response") from error


def get_subcategories(category: int = 1, timeout: float = 30) -> list[dict[str, Any]]:
    """Fetch the subcategories belonging to an AMFI category."""
    response = post_json(SUBCATEGORY_ENDPOINT, {"category": category}, timeout)
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise RuntimeError("CRISIL returned an unexpected subcategory response")
    return response["data"]


def get_all_fund_performance(
    category: int = 1,
    report_date: str = REPORT_DATE,
    maturity_type: int = 1,
    mfid: int = 0,
    timeout: float = 30,
) -> list[dict[str, Any]]:
    """Fetch fund-performance responses for every subcategory in a category."""
    return get_fund_performance_for_subcategories(
        get_subcategories(category, timeout),
        category=category,
        report_date=report_date,
        maturity_type=maturity_type,
        mfid=mfid,
        timeout=timeout,
    )


def get_fund_performance_for_subcategories(
    subcategories: list[dict[str, Any]],
    category: int = 1,
    report_date: str = REPORT_DATE,
    maturity_type: int = 1,
    mfid: int = 0,
    timeout: float = 30,
) -> list[dict[str, Any]]:
    """Fetch fund-performance responses for a previously fetched subcategory list."""
    results: list[dict[str, Any]] = []
    for subcategory in subcategories:
        subcategory_id = subcategory.get("id")
        if not isinstance(subcategory_id, int):
            raise RuntimeError(f"Subcategory is missing a numeric ID: {subcategory!r}")

        response = post_json(
            FUND_PERFORMANCE_ENDPOINT,
            {
                "maturityType": maturity_type,
                "category": category,
                "subCategory": subcategory_id,
                "mfid": mfid,
                "reportDate": report_date,
            },
            timeout,
        )
        results.append(
            {
                "subcategory": subcategory,
                "response": response,
            }
        )
    return results


def main() -> int:
    try:
        print(json.dumps(get_all_fund_performance(), indent=2, ensure_ascii=False))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
