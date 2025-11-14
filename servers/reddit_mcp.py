# servers/reddit_mcp.py
# -*- coding: utf-8 -*-
"""
Reddit MCP Server (OAuth fixed version)
---------------------------------------
✔ Uses refresh_token for real OAuth authentication
✔ Connects to oauth.reddit.com (no HTML errors)
✔ Supports both 'hot' feed and full post fetch
✔ Returns clean text for direct PPT integration
"""

import os
import aiohttp
import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP

# ----------------------------------------------------------------------
# MCP Server initialization
# ----------------------------------------------------------------------
server = FastMCP("reddit")
logging.getLogger().setLevel(logging.WARNING)

# ----------------------------------------------------------------------
# Load Reddit OAuth credentials
# ----------------------------------------------------------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_REFRESH_TOKEN = os.getenv("REDDIT_REFRESH_TOKEN")

if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_REFRESH_TOKEN]):
    print("[reddit_mcp] ⚠️ Reddit API credentials not configured properly.")


# ----------------------------------------------------------------------
# Helper: Get OAuth Access Token
# ----------------------------------------------------------------------
async def _get_access_token() -> Optional[str]:
    async with aiohttp.ClientSession() as session:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": REDDIT_REFRESH_TOKEN,
        }
        auth = aiohttp.BasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        async with session.post("https://www.reddit.com/api/v1/access_token", data=data, auth=auth) as resp:
            if resp.status != 200:
                msg = f"OAuth failed: {resp.status}"
                print(f"[reddit_mcp] ❌ {msg}")
                raise Exception(msg)
            token_data = await resp.json()
            return token_data["access_token"]


# ----------------------------------------------------------------------
# TOOL 1: SEARCH HOT POSTS
# ----------------------------------------------------------------------
@server.tool()
async def search_hot_posts(subreddit: str = "MachineLearning", limit: int = 5) -> str:
    """
    Retrieve the hottest (trending) posts from a specific Reddit subreddit.

    This tool is particularly useful for:
    - Augmenting slides, essays, or research with trending public opinions.
    - Auto-answering detected questions (integration with presentation MCP).
    - Discovering real-time discussion topics.

    Args:
        subreddit (str):
            Name of the subreddit without prefix (e.g., "technology", "AskReddit").
        limit (int):
            Number of posts to retrieve (default: 5, max recommended: 10).

    Returns:
        str:
            A plain-text list of posts. Each post includes:
                - Title
                - Score (upvotes)
                - Comment count
                - Author username
                - Link URL to the Reddit post
            Example output:
                - What is AGI?
                  Score: 5421
                  Comments: 821
                  Author: user123
                  Link: https://reddit.com/...

    Error Handling:
        Returns an error string if Reddit API fails or credentials are missing.
    """
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_REFRESH_TOKEN]):
        return "Reddit credentials not configured."

    try:
        access_token = await _get_access_token()

        headers = {
            "Authorization": f"bearer {access_token}",
            "User-Agent": "MCP-Reddit-Server/1.0"
        }

        url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}&raw_json=1"

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"Error: Reddit API returned {resp.status}"
                data = await resp.json()

        results = []
        for post in data.get("data", {}).get("children", []):
            p = post["data"]
            results.append(
                f"- {p.get('title')}\n"
                f"  Score: {p.get('score')} | Comments: {p.get('num_comments')}\n"
                f"  Author: {p.get('author')}\n"
                f"  Link: https://reddit.com{p.get('permalink')}\n"
            )

        if not results:
            return "No results found."
        return "\n".join(results)

    except Exception as e:
        return f"Error fetching Reddit posts: {e}"


# ----------------------------------------------------------------------
# TOOL 2: GET FULL POST CONTENT
# ----------------------------------------------------------------------
@server.tool()
async def get_post_content(post_id: str, comment_limit: int = 10, comment_depth: int = 2) -> str:
    """
    Retrieve a detailed Reddit post including nested comments.

    Ideal for:
    - Deep-dive Q&A generation.
    - Extracting public argument structures or debates.
    - Narrative content enrichment for reports or slides.

    Args:
        post_id (str):
            Reddit post ID in base36 format (e.g., "abc123").
        comment_limit (int):
            Max number of top-level comments to fetch (default: 10).
        comment_depth (int):
            Maximum nesting level of comment threads (default: 2).

    Returns:
        str:
            A comprehensive text block including:
                - Title, Score, Author
                - Body content
                - Hierarchical comments (formatted with -- indentation)

    Note:
        Uses async/await and avoids `asyncio.run()` to support MCP concurrency.
    """
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_REFRESH_TOKEN]):
        return "Reddit credentials not configured."

    try:
        access_token = await _get_access_token()
        headers = {
            "Authorization": f"bearer {access_token}",
            "User-Agent": "MCP-Reddit-Server/1.0"
        }

        url = f"https://oauth.reddit.com/comments/{post_id}?limit={comment_limit}&depth={comment_depth}&raw_json=1"

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"Error: Reddit API returned {resp.status}"
                data = await resp.json()

        if not data or len(data) < 2:
            return "No post data found."

        post_data = data[0]["data"]["children"][0]["data"]
        comments = data[1]["data"]["children"]

        result = [
            f"**{post_data.get('title')}**",
            f"Author: {post_data.get('author')}",
            f"Score: {post_data.get('score')}",
            f"Content: {post_data.get('selftext') or '(no text)'}",
            "\nComments:\n"
        ]

        for c in comments[:comment_limit]:
            if c["kind"] == "t1":
                d = c["data"]
                result.append(f"- {d.get('author')}: {d.get('body')[:200]}")

        return "\n".join(result)

    except Exception as e:
        return f"Error fetching Reddit post content: {e}"


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    server.run(transport="stdio")
