"""Company filings from SEC EDGAR.

No key, but the SEC requires a User-Agent with a name and email
(SEC_USER_AGENT in .env) and allows at most 10 requests per second.
"""

import re
from html.parser import HTMLParser

from finagent import config
from finagent.cache import cached
from finagent.tools._http import get_json, get_text

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def _headers() -> dict:
    return {"User-Agent": config.require("SEC_USER_AGENT")}


def ticker_to_cik(ticker: str) -> tuple[int, str]:
    """Return (CIK, company title) for a ticker."""
    def fetch():
        raw = get_json(TICKERS_URL, headers=_headers())
        return {
            row["ticker"].upper(): [row["cik_str"], row["title"]]
            for row in raw.values()
        }

    mapping = cached("sec_tickers", {}, fetch, ttl_hours=24 * 30)
    ticker = ticker.upper().replace(".", "-")
    if ticker not in mapping:
        raise ValueError(f"{ticker} not found in SEC ticker list")
    cik, title = mapping[ticker]
    return int(cik), title


def parse_recent_filings(payload: dict, cik: int, forms: set[str] | None,
                         limit: int) -> list[dict]:
    recent = payload.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(recent.get("form", [])):
        if forms and form not in forms:
            continue
        acc = recent["accessionNumber"][i]
        doc = recent["primaryDocument"][i]
        acc_path = acc.replace("-", "")
        out.append({
            "form": form,
            "filing_date": recent["filingDate"][i],
            "report_date": recent.get("reportDate", [""] * (i + 1))[i],
            "accession": acc,
            "description": recent.get(
                "primaryDocDescription", [""] * (i + 1))[i],
            "url": ARCHIVE_URL.format(cik=cik, acc=acc_path, doc=doc),
        })
        if len(out) >= limit:
            break
    return out


def get_recent_filings(ticker: str, form: str = "10-K,10-Q,8-K",
                       limit: int = 5) -> list[dict]:
    """Recent SEC filings for a company, newest first.

    Returns form type, filing date, period (report_date), and a URL to
    the main document. 10-K = annual report, 10-Q = quarterly report,
    8-K = material event (earnings releases, leadership changes, deals).
    Use to find official disclosures, then get_filing_text to read one.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        form: Comma-separated form types, or "any".
        limit: Max filings to return.
    """
    cik, _ = ticker_to_cik(ticker)
    payload = cached(
        "sec_submissions", {"cik": cik},
        lambda: get_json(SUBMISSIONS_URL.format(cik=cik),
                         headers=_headers()),
        ttl_hours=12,
    )
    forms = None if form.lower() == "any" else {
        f.strip().upper() for f in form.split(",")
    }
    return parse_recent_filings(payload, cik, forms, limit)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "ix:header"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "ix:header") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = " ".join(parser.parts).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def find_section(text: str, section: str, max_chars: int) -> str:
    """Return text starting at a section heading.

    Headings usually appear first in the table of contents, so when the
    heading occurs more than once we take the second occurrence.
    """
    hits = [m.start() for m in re.finditer(re.escape(section), text,
                                           re.IGNORECASE)]
    if not hits:
        return ""
    start = hits[1] if len(hits) > 1 else hits[0]
    return text[start:start + max_chars]


def get_filing_text(url: str, section: str | None = None,
                    max_chars: int = 12000) -> dict:
    """Plain text of an SEC filing document (retrieval tool).

    Use after get_recent_filings to read what a filing says. Pass a
    section heading to jump to it, e.g. "Risk Factors", "Management's
    Discussion", "Results of Operations", or "Liquidity". Without a
    section you get the start of the document.

    Args:
        url: Document URL from get_recent_filings.
        section: Optional heading to jump to.
        max_chars: Max characters of text to return.
    """
    text = cached(
        "sec_docs", {"url": url},
        lambda: html_to_text(get_text(url, headers=_headers())),
        ttl_hours=24 * 30,
    )
    if section:
        excerpt = find_section(text, section, max_chars)
        found = bool(excerpt)
        excerpt = excerpt or text[:max_chars]
    else:
        excerpt, found = text[:max_chars], None
    return {
        "url": url,
        "section": section,
        "section_found": found,
        "total_chars": len(text),
        "text": excerpt,
    }
