#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Context Protocol (MCP) Server: openlibrary_mcp
====================================================

Retrieves book metadata from the OpenLibrary API using ISBN or title.
Now includes a safety wrapper to prevent recursive MCP re-triggers
(by moving isbn/title fields into a `metadata` sub-dict).

Author: Bob’s MCP system integration
"""

import requests
from typing import Optional
from mcp.server.fastmcp import FastMCP
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

server = FastMCP("openlibrary")

# ------------------------------------------------------------
# Create a shared session with retry + custom User-Agent
# ------------------------------------------------------------
_session = requests.Session()
_session.headers.update({
    "User-Agent": "mcp-openlibrary/1.0 (eta@rutgers.edu)",
    "Accept": "application/json",
})
_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))

# ------------------------------------------------------------
# Prevent recursive triggering: safe output wrapper
# ------------------------------------------------------------
def _sanitize_output(obj: dict) -> dict:
    """Prevent recursive triggering: move top-level fields like isbn/title into metadata"""
    if not isinstance(obj, dict):
        return obj

    safe = {}
    metadata = {}
    for k, v in obj.items():
        if k in {"isbn", "title", "authors", "publisher", "publish_date", "pages"}:
            metadata[k] = v
        else:
            safe[k] = v
    if metadata:
        safe["metadata"] = metadata
    return safe


# ------------------------------------------------------------
# Main tool: get_book_info
# ------------------------------------------------------------
@server.tool()
def get_book_info(
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> dict:
    """
      Look up book metadata from OpenLibrary by ISBN or title with optional author filter.
      Args:
        isbn (Optional[str]): Book ISBN-10 or ISBN-13, preferred when available.
        title (Optional[str]): Book title used for text search when ISBN is not provided.
        author (Optional[str]): Author name to narrow title-based search.
      Returns:
        result (dict): Sanitized dictionary with core book metadata, messages, or error details.
    """

    try:
        # --- Case 1: ISBN ---
        if isbn:
            if isbn.startswith("0") and len(isbn) == 13:
                isbn = "978" + isbn[1:]

            url = f"https://openlibrary.org/api/books"
            params = {"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"}
            r = _session.get(url, params=params, timeout=10)
            r.raise_for_status()

            data = r.json()
            key = f"ISBN:{isbn}"
            if key not in data:
                return _sanitize_output({
                    "message": f"Book not found for ISBN {isbn}."
                })

            book = data[key]
            return _sanitize_output({
                "query_type": "isbn",
                "isbn": isbn,
                "title": book.get("title", "Unknown"),
                "authors": [a["name"] for a in book.get("authors", [])],
                "publish_date": book.get("publish_date"),
                "publisher": [p["name"] for p in book.get("publishers", [])],
                "pages": book.get("number_of_pages"),
            })

        elif title:
            query = title
            if author:
                query += f" {author}"

            url = "https://openlibrary.org/search.json"
            params = {"q": query}
            r = _session.get(url, params=params, timeout=10)
            r.raise_for_status()

            results = r.json().get("docs", [])
            if not results:
                return _sanitize_output({
                    "message": f"No book found for query '{query}'."
                })

            top = results[0]
            return _sanitize_output({
                "query_type": "search",
                "query": query,
                "title": top.get("title"),
                "authors": top.get("author_name", []),
                "publish_date": top.get("first_publish_year"),
                "publisher": top.get("publisher", []),
                "isbn": top.get("isbn", [None])[0],
            })

        else:
            return _sanitize_output({
                "error": "Please provide either an ISBN or a title."
            })

    except requests.Timeout:
        return _sanitize_output({"error": "Request to OpenLibrary timed out."})
    except requests.RequestException as e:
        return _sanitize_output({"error": f"Network error: {str(e)}"})
    except Exception as e:
        return _sanitize_output({"error": f"Unexpected error: {str(e)}"})


# ------------------------------------------------------------
if __name__ == "__main__":
    print("[openlibrary_mcp] Server starting with book lookup tool...")
    server.run(transport="stdio")
