"""
citation_verifier.py

This script analyses a Grokipedia article and checks whether the citations attached to each claim rely on independent sources. It extracts the citations from the article, fetches the external references, and groups them by their underlying sources/domains.

Usage:
    python citation_verifier.py <Grokipedia article URL>

Example:
    python citation_verifier.py https://grokipedia.com/page/Elon_Musk

The script will output the citation sources for each citation and show how many unique sources underpin them.
"""

import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import collections

def extract_citation_links(article_url):
    """Return a list of lists of external links for each citation in the Grokipedia article."""
    resp = requests.get(article_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    citation_refs = []
    # Find all sup tags with class reference (the citation markers).
    for sup in soup.find_all('sup'):
        # citation markers have anchors pointing to footnotes (#cite_note-1 etc.)
        a = sup.find('a')
        if not a or not a.get('href'):
            continue
        href = a['href']
        # if it's an internal footnote
        if href.startswith('#'):
            footnote = soup.find(id=href[1:])
            if not footnote:
                continue
            # gather external links in the footnote
            links = []
            for link in footnote.find_all('a', href=True):
                url = link['href']
                # convert relative to absolute
                full_url = urljoin(article_url, url)
                # only keep http(s) links
                if full_url.startswith('http'):
                    links.append(full_url)
            citation_refs.append(links)
        else:
            citation_refs.append([urljoin(article_url, href)])
    return citation_refs

def get_domain(url):
    parsed = urlparse(url)
    return parsed.netloc.lower()

def analyze_dependencies(citation_refs):
    """Return mapping of citation index to its source domain(s) and overall grouping."""
    citation_sources = {}
    for idx, links in enumerate(citation_refs, start=1):
        domains = [get_domain(link) for link in links]
        citation_sources[idx] = domains
    # group by domain
    domain_to_citations = collections.defaultdict(list)
    for cit_idx, domains in citation_sources.items():
        for d in set(domains):
            domain_to_citations[d].append(cit_idx)
    return citation_sources, domain_to_citations

def main():
    if len(sys.argv) < 2:
        print("Usage: python citation_verifier.py <Grokipedia article URL>")
        sys.exit(1)
    article_url = sys.argv[1]
    citation_refs = extract_citation_links(article_url)
    citation_sources, domain_to_citations = analyze_dependencies(citation_refs)
    print("Citation sources by citation number:")
    for i in sorted(citation_sources):
        print(f"  Citation [{i}]: {citation_sources[i]}")
    print("\nUnique sources and which citations they appear in:")
    for domain, cites in domain_to_citations.items():
        print(f"  {domain}: cited in {cites}")
    print(f"\nActual source count: {len(domain_to_citations)}")

if __name__ == '__main__':
    main()
