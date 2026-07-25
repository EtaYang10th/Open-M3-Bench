# paper_search_mcp/sources/arxiv.py
from typing import List
from datetime import datetime
import requests
import feedparser
from ..paper import Paper
from PyPDF2 import PdfReader
import os
import time


def _get_with_backoff(url, params=None, *, max_retries=5, base_delay=3.0, timeout=30):
    """GET with exponential backoff. arXiv requires >=3s between calls and
    throttles with HTTP 429. Retries on 429/5xx and network errors instead of
    silently returning an empty feed."""
    delay = base_delay
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                print(f"[arxiv] HTTP {resp.status_code}, retry {attempt}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            print(f"[arxiv] request error ({e}); retry {attempt}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError(f"arXiv request failed after {max_retries} retries (last status not 200)")

class PaperSource:
    """Abstract base class for paper sources"""
    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

class ArxivSearcher(PaperSource):
    """Searcher for arXiv papers"""
    BASE_URL = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        params = {
            'search_query': query,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        response = _get_with_backoff(self.BASE_URL, params=params)
        feed = feedparser.parse(response.content)
        if not feed.entries:
            print(f"[arxiv] no entries for query={query!r} (status={response.status_code})")
        papers = []
        for entry in feed.entries:
            try:
                authors = [author.name for author in entry.authors]
                published = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
                updated = datetime.strptime(entry.updated, '%Y-%m-%dT%H:%M:%SZ')
                pdf_url = next((link.href for link in entry.links if link.type == 'application/pdf'), '')
                papers.append(Paper(
                    paper_id=entry.id.split('/')[-1],
                    title=entry.title,
                    authors=authors,
                    abstract=entry.summary,
                    url=entry.id,
                    pdf_url=pdf_url,
                    published_date=published,
                    updated_date=updated,
                    source='arxiv',
                    categories=[tag.term for tag in entry.tags],
                    keywords=[],
                    doi=entry.get('doi', '')
                ))
            except Exception as e:
                print(f"Error parsing arXiv entry: {e}")
        return papers

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        response = requests.get(pdf_url)
        output_file = f"{save_path}/{paper_id}.pdf"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        return output_file

    def read_paper(self, paper_id: str, save_path: str = "./downloads") -> str:
        """Read a paper and convert it to text format.
        
        Args:
            paper_id: arXiv paper ID
            save_path: Directory where the PDF is/will be saved
            
        Returns:
            str: The extracted text content of the paper
        """
        # First ensure we have the PDF
        pdf_path = f"{save_path}/{paper_id}.pdf"
        if not os.path.exists(pdf_path):
            pdf_path = self.download_pdf(paper_id, save_path)
        
        # Read the PDF
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF for paper {paper_id}: {e}")
            return ""

if __name__ == "__main__":
    # Test ArxivSearcher functionality
    searcher = ArxivSearcher()
    
    # Test search functionality
    print("Testing search functionality...")
    query = "machine learning"
    max_results = 5
    try:
        papers = searcher.search(query, max_results=max_results)
        print(f"Found {len(papers)} papers for query '{query}':")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title} (ID: {paper.paper_id})")
    except Exception as e:
        print(f"Error during search: {e}")
    
    # Test PDF download functionality
    if papers:
        print("\nTesting PDF download functionality...")
        paper_id = papers[0].paper_id
        save_path = "./downloads"  # ensure this directory exists
        try:
            os.makedirs(save_path, exist_ok=True)
            pdf_path = searcher.download_pdf(paper_id, save_path)
            print(f"PDF downloaded successfully: {pdf_path}")
        except Exception as e:
            print(f"Error during PDF download: {e}")

    # Test paper reading functionality
    if papers:
        print("\nTesting paper reading functionality...")
        paper_id = papers[0].paper_id
        try:
            text_content = searcher.read_paper(paper_id)
            print(f"\nFirst 500 characters of the paper content:")
            print(text_content[:500] + "...")
            print(f"\nTotal length of extracted text: {len(text_content)} characters")
        except Exception as e:
            print(f"Error during paper reading: {e}")