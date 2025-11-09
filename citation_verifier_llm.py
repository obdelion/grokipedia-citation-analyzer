#!/usr/bin/env python3
"""
citation_verifier_llm.py (small model version)

This script analyses citations in a Grokipedia article and attempts to identify
whether multiple citations ultimately refer to the same original source.
It uses an open-source language model from Hugging Face (defaulting to the
lightweight `google/flan-t5-small`) to read each cited article and answer
which news outlet or wire service is credited.

Usage:
    python citation_verifier_llm.py https://grokipedia.com/page/SomePage

If no URL argument is provided, the script will prompt you to enter a Grokipedia URL.

Dependencies:
    pip install requests beautifulsoup4 transformers torch

Notes:
    - The default model (flan-t5-small) is much smaller (~250MB) than flan-t5-base,
      so it downloads faster and requires less memory. You can change the model
      name by editing the load_llm() function or passing a different value when
      calling it.
    - On first run, the model weights will be downloaded and cached locally.
      Subsequent runs reuse the cached weights.
"""
import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # Handle missing transformers gracefully

def extract_citation_links(article_url):
    """Extract citation links from a Grokipedia article."""
    resp = requests.get(article_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    citation_refs = []
    # Extract from <sup> markers
    for sup in soup.find_all('sup'):
        a = sup.find('a')
        if not a or not a.get('href'):
            continue
        href = a['href']
        if href.startswith('#'):
            footnote = soup.find(id=href[1:])
            if not footnote:
                continue
            links = []
            for link in footnote.find_all('a', href=True):
                full_url = urljoin(article_url, link['href'])
                if full_url.startswith('http'):
                    links.append(full_url)
            citation_refs.append(links)
        else:
            citation_refs.append([urljoin(article_url, href)])
    # Fallback: collect all external links if no citations found
    if not citation_refs:
        links = []
        for link in soup.find_all('a', href=True):
            full_url = urljoin(article_url, link['href'])
            if full_url.startswith('http'):
                links.append(full_url)
        citation_refs = [[url] for url in links]
    return citation_refs

def fetch_article_text(url):
    """Fetch article text by concatenating paragraph elements."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(resp.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs).strip()

def load_llm(model_name="google/flan-t5-small"):
    """Load a text-generation pipeline from transformers with a specified model."""
    if pipeline is None:
        raise ImportError("transformers library is not installed. Run: pip install transformers torch")
    return pipeline("text2text-generation", model=model_name)

def identify_source(text, llm):
    """Use the LLM to identify the credited news outlet or wire service."""
    if not text:
        return "none"
    prompt = (
        "Identify the news outlet or wire service credited as the original source in the following article. "
        "If there is no explicit attribution to another outlet or wire service, respond with 'none'. "
        "Article: " + text
    )
    try:
        result = llm(prompt, max_length=50)[0]['generated_text']
    except Exception:
        return "none"
    answer = result.strip().split("\n")[0]
    return answer.lower()

def main():
    article_url = sys.argv[1] if len(sys.argv) > 1 else input("Enter the Grokipedia article URL: ").strip()
    if not article_url:
        print("No URL provided.")
        return
    citations = extract_citation_links(article_url)
    try:
        llm = load_llm()
    except ImportError as e:
        print(str(e))
        return
    citation_sources = []
    for citation_links in citations:
        if not citation_links:
            citation_sources.append("none")
            continue
        url = citation_links[0]
        text = fetch_article_text(url)
        src = identify_source(text, llm)
        citation_sources.append(src)
    source_map = {}
    for idx, src in enumerate(citation_sources, start=1):
        source_map.setdefault(src, []).append(idx)
    print("Citation ultimate sources by citation number:")
    for i, src in enumerate(citation_sources, start=1):
        print(f"  Citation [{i}]: {src}")
    print("\nUnique ultimate sources and which citations they appear in:")
    for src, nums in source_map.items():
        print(f"  {src}: cited in {nums}")
    unique = [s for s in source_map if s != "none"]
    print(f"\nActual unique cited ultimate source count: {len(unique)}")

if __name__ == "__main__":
    main()
