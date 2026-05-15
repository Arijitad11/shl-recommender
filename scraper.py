"""
scraper.py – Attempts to scrape the live SHL Individual Test Solutions catalog.
Falls back gracefully to the static catalog_data.py if the site blocks requests.
Run standalone:  python scraper.py   (produces catalog.json)
"""

from __future__ import annotations
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from catalog_data import CATALOG

log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_PAGE = f"{BASE_URL}/solutions/products/product-catalog/"
CATALOG_JSON = Path(__file__).parent / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.shl.com/",
}

# type=1  →  Individual Test Solutions only (pre-packaged job solutions excluded)
FILTER_PARAMS = {"type": "1", "start": "0"}


def _parse_row(row: BeautifulSoup, base: str) -> dict | None:
    """Extract one product from a catalog table row."""
    try:
        link = row.find("a")
        if not link:
            return None
        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = base + href if href.startswith("/") else href

        # test_type letters sit in single-letter <td> cells
        tds = row.find_all("td")
        test_types = [td.get_text(strip=True) for td in tds
                      if re.fullmatch(r"[A-Z]", td.get_text(strip=True))]

        return {
            "name": name,
            "url": url,
            "test_type": test_types or ["K"],
            "description": "",
            "tags": [],
            "job_levels": [],
            "remote_testing": True,
            "adaptive": False,
        }
    except Exception as exc:
        log.debug("Row parse error: %s", exc)
        return None


def scrape_live() -> list[dict]:
    """Attempt live scrape; return empty list if blocked or error."""
    session = requests.Session()
    session.headers.update(HEADERS)

    products: list[dict] = []
    page = 0

    while True:
        params = {**FILTER_PARAMS, "start": str(page * 12)}
        try:
            resp = session.get(CATALOG_PAGE, params=params, timeout=15)
        except requests.RequestException as exc:
            log.warning("Scrape request failed: %s", exc)
            break

        if resp.status_code != 200:
            log.warning("Non-200 from SHL catalog: %s", resp.status_code)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Locate catalog rows (SHL uses a custom component; try several selectors)
        rows = (
            soup.select("tr.product-catalogue__row")
            or soup.select(".catalogue-table tbody tr")
            or soup.select("table tr")[1:]
        )

        if not rows:
            log.debug("No rows found on page %d – stopping", page)
            break

        for row in rows:
            item = _parse_row(row, BASE_URL)
            if item and item["name"]:
                products.append(item)

        # Pagination
        if not soup.select_one("a[rel='next'], .pagination__next"):
            break
        page += 1
        if page > 30:
            break
        time.sleep(0.4)

    log.info("Live scrape returned %d items", len(products))
    return products


def _merge(live: list[dict], static: list[dict]) -> list[dict]:
    """
    Merge live items with static catalog.
    - If live has an item with the same name as static, update its url & test_type
      but keep the richer static description/tags.
    - Live-only items are appended as-is.
    """
    by_name = {item["name"].lower(): item for item in static}
    seen = set(by_name.keys())

    for item in live:
        key = item["name"].lower()
        if key in by_name:
            # Update URL and test_type from live source but keep rich metadata
            by_name[key]["url"] = item["url"] or by_name[key]["url"]
            if item["test_type"]:
                by_name[key]["test_type"] = item["test_type"]
        else:
            by_name[key] = item
            seen.add(key)

    return list(by_name.values())


def build_catalog(force_refresh: bool = False) -> list[dict]:
    """
    Return catalog list.  Tries live scrape first; if it yields nothing,
    falls back to static data.  Caches to catalog.json for subsequent calls.
    """
    if not force_refresh and CATALOG_JSON.exists():
        with CATALOG_JSON.open() as f:
            data = json.load(f)
        log.info("Loaded %d items from cached catalog.json", len(data))
        return data

    live = scrape_live()
    merged = _merge(live, CATALOG) if live else CATALOG

    CATALOG_JSON.write_text(json.dumps(merged, indent=2))
    log.info("Catalog built with %d items (live=%d, static=%d)", len(merged), len(live), len(CATALOG))
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cat = build_catalog(force_refresh=True)
    print(f"Total catalog items: {len(cat)}")
    for item in cat[:5]:
        print(f"  {item['name']}  [{','.join(item['test_type'])}]  {item['url']}")
