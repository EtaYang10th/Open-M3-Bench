# paper_search_mcp/server.py
from typing import List, Dict, Optional
import httpx
from mcp.server.fastmcp import FastMCP
from .academic_platforms.arxiv import ArxivSearcher
from .academic_platforms.pubmed import PubMedSearcher
from .academic_platforms.biorxiv import BioRxivSearcher
from .academic_platforms.medrxiv import MedRxivSearcher
from .academic_platforms.google_scholar import GoogleScholarSearcher
from .academic_platforms.iacr import IACRSearcher
from .academic_platforms.semantic import SemanticSearcher
from .academic_platforms.crossref import CrossRefSearcher

# from .academic_platforms.hub import SciHubSearcher
from .paper import Paper

# Initialize MCP server
mcp = FastMCP("paper_search_server")

# Instances of searchers
arxiv_searcher = ArxivSearcher()
pubmed_searcher = PubMedSearcher()
biorxiv_searcher = BioRxivSearcher()
medrxiv_searcher = MedRxivSearcher()
google_scholar_searcher = GoogleScholarSearcher()
iacr_searcher = IACRSearcher()
semantic_searcher = SemanticSearcher()
crossref_searcher = CrossRefSearcher()
# scihub_searcher = SciHubSearcher()


# Asynchronous helper to adapt synchronous searchers
async def async_search(searcher, query: str, max_results: int, **kwargs) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        # Assuming searchers use requests internally; we'll call synchronously for now
        if 'year' in kwargs:
            papers = searcher.search(query, year=kwargs['year'], max_results=max_results)
        else:
            papers = searcher.search(query, max_results=max_results)
        return [paper.to_dict() for paper in papers]


@mcp.tool()
async def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from arXiv.
    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(arxiv_searcher, query, max_results)

    if papers is None:
        return []

    if not isinstance(papers, list):
        raise TypeError(f"Expected List[Dict] but got {type(papers).__name__}")

    for p in papers:
        if not isinstance(p, dict):
            raise TypeError(f"Invalid paper metadata type: {type(p).__name__}")

    return papers

@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from PubMed.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(pubmed_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_biorxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from bioRxiv.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(biorxiv_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_medrxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from medRxiv.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(medrxiv_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_google_scholar(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from Google Scholar.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(google_scholar_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_iacr(
    query: str, max_results: int = 10, fetch_details: bool = True
) -> List[Dict]:
    """
    Search academic papers from the IACR ePrint Archive.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.
      fetch_details (bool): Whether to fetch detailed metadata.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    async with httpx.AsyncClient() as client:
        papers = iacr_searcher.search(query, max_results, fetch_details)
        return [paper.to_dict() for paper in papers] if papers else []


@mcp.tool()
async def download_arxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Download the PDF of an arXiv paper.

    Args:
      paper_id (str): arXiv identifier.
      save_path (str): Directory to save the PDF.

    Returns:
      path (str): File path of the downloaded PDF.
    """
    async with httpx.AsyncClient() as client:
        return arxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_pubmed(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Attempt to download the PDF of a PubMed paper.

    Args:
      paper_id (str): PubMed ID (PMID).
      save_path (str): Directory to save the PDF.

    Returns:
      message (str): Status or explanation of download support.
    """
    try:
        return pubmed_searcher.download_pdf(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)


@mcp.tool()
async def download_biorxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Download the PDF of a bioRxiv paper.

    Args:
      paper_id (str): bioRxiv DOI.
      save_path (str): Directory to save the PDF.

    Returns:
      path (str): File path of the downloaded PDF.
    """
    return biorxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_medrxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Download the PDF of a medRxiv paper.

    Args:
      paper_id (str): medRxiv DOI.
      save_path (str): Directory to save the PDF.

    Returns:
      path (str): File path of the downloaded PDF.
    """
    return medrxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_iacr(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Download the PDF of an IACR ePrint paper.

    Args:
      paper_id (str): IACR ePrint identifier.
      save_path (str): Directory to save the PDF.

    Returns:
      path (str): File path of the downloaded PDF.
    """
    return iacr_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def read_arxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Read and extract text content from an arXiv paper PDF.

    Args:
      paper_id (str): arXiv identifier.
      save_path (str): Directory where the PDF is or will be stored.

    Returns:
      text (str): Extracted plain text content.
    """
    try:
        return arxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_pubmed_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Attempt to read and extract text from a PubMed paper.

    Args:
      paper_id (str): PubMed ID (PMID).
      save_path (str): Directory path (may be unused).

    Returns:
      message (str): Status or explanation of read support.
    """
    return pubmed_searcher.read_paper(paper_id, save_path)


@mcp.tool()
async def read_biorxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Read and extract text content from a bioRxiv paper PDF.

    Args:
      paper_id (str): bioRxiv DOI.
      save_path (str): Directory where the PDF is or will be stored.

    Returns:
      text (str): Extracted plain text content.
    """
    try:
        return biorxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_medrxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Read and extract text content from a medRxiv paper PDF.

    Args:
      paper_id (str): medRxiv DOI.
      save_path (str): Directory where the PDF is or will be stored.

    Returns:
      text (str): Extracted plain text content.
    """
    try:
        return medrxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_iacr_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Read and extract text content from an IACR ePrint paper PDF.

    Args:
      paper_id (str): IACR ePrint identifier.
      save_path (str): Directory where the PDF is or will be stored.

    Returns:
      text (str): Extracted plain text content.
    """
    try:
        return iacr_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def search_semantic(query: str, year: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    """
    Search academic papers from Semantic Scholar.

    Args:
      query (str): Search query text.
      year (Optional[str]): Optional year filter expression.
      max_results (int): Maximum number of papers to return.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    kwargs = {}
    if year is not None:
        kwargs['year'] = year
    papers = await async_search(semantic_searcher, query, max_results, **kwargs)
    return papers if papers else []


@mcp.tool()
async def download_semantic(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Download the PDF of a Semantic Scholar paper.

    Args:
      paper_id (str): Semantic Scholar paper identifier or alias (ID, DOI, ARXIV, etc.).
      save_path (str): Directory to save the PDF.

    Returns:
      path (str): File path of the downloaded PDF.
    """
    return semantic_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def read_semantic_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Read and extract text content from a Semantic Scholar paper.

    Args:
      paper_id (str): Semantic Scholar paper identifier or alias (ID, DOI, ARXIV, etc.).
      save_path (str): Directory where the PDF is or will be stored.

    Returns:
      text (str): Extracted plain text content.
    """
    try:
        return semantic_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def search_crossref(query: str, max_results: int = 10, **kwargs) -> List[Dict]:
    """
    Search academic papers from the CrossRef database.

    Args:
      query (str): Search query text.
      max_results (int): Maximum number of papers to return.
      **kwargs: Optional CrossRef filter or sort parameters.

    Returns:
      papers (List[Dict]): Paper metadata as dictionaries.
    """
    papers = await async_search(crossref_searcher, query, max_results, **kwargs)
    return papers if papers else []


@mcp.tool()
async def get_crossref_paper_by_doi(doi: str) -> Dict:
    """
    Get a specific CrossRef record by DOI.

    Args:
      doi (str): Digital Object Identifier string.

    Returns:
      paper (Dict): Paper metadata dictionary or empty dict if not found.
    """
    async with httpx.AsyncClient() as client:
        paper = crossref_searcher.get_paper_by_doi(doi)
        return paper.to_dict() if paper else {}


@mcp.tool()
async def download_crossref(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Attempt to download the PDF of a CrossRef-referenced paper.

    Args:
      paper_id (str): CrossRef DOI.
      save_path (str): Directory to save the PDF.

    Returns:
      message (str): Status or explanation of download support.
    """
    try:
        return crossref_searcher.download_pdf(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)


@mcp.tool()
async def read_crossref_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """
    Attempt to read and extract text content from a CrossRef-referenced paper.

    Args:
      paper_id (str): CrossRef DOI.
      save_path (str): Directory where the PDF is or would be stored.

    Returns:
      message (str): Status or explanation of read support.
    """
    return crossref_searcher.read_paper(paper_id, save_path)


if __name__ == "__main__":
    mcp.run(transport="stdio")
