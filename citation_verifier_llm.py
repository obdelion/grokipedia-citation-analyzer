#!/usr/bin/env python3
"""
citation_verifier_llm.py

This script analyses citations in a Grokipedia article and attempts to identify
whether multiple citations ultimately refer to the same original source.
It uses an open-source language model from Hugging Face (e.g., google/flan-t5-base)
to read each cited article and answer which news outlet or wire service is credited.

Usage:
    python citation_verifier_llm.py https://grokipedia.com/page/SomePage

If no URL argument is provided, the script will prompt you to enter a Grokipedia URL.

Dependencies:
    pip install requests beautifulsoup4 transformers torch

Note: Running an open-source LLM can require significant memory. You may choose
a smaller model by changing the model name in load_llm().
"""
import sys
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # Handle missing transformers gracefully

def extract_citation_links(article_url):
    """
    Return a list of lists of external links for each citation in the Grokipedia article.
    Falls back to collecting all external links if no citation markers are found.
    """
    resp = requests.get(article_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    citation_refs = []

    # parse <sup> markers
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

    # fallback: if no citations found, collect all external links
    if not citation_refs:
        links = []
        for link in soup.find_all('a', href=True):
            full_url = urljoin(article_url, link['href'])
            if full_url.startswith('http'):
                links.append(full_url)
        citation_refs = [[url] for url in links]
    return citation_refs

def fetch_article_text(url):
    """Fetch the text content of an article by concatenating its paragraphs."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(resp.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    return text.strip()

def load_llm(model_name="google/flan-t5-base"):
    """Load a text-generation pipeline from transformers."""
    if pipeline is None:
        raise ImportError("transformers library is not installed. Run: pip install transformers torch")
    # Use text2text-generation pipeline since we ask a question about the article.
    return pipeline("text2text-generation", model=model_name)

def identify_source(text, llm):
    """
    Use the LLM to identify which news outlet or wire service is credited.
    Returns a lowercase string naming the credited source or 'none'.
    """
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
    # Simplify the result to the first line
    answer = result.strip().split("\n")[0]
    return answer.lower()

def main():
    # Determine article URL from CLI or prompt
    if len(sys.argv) > 1:
        article_url = sys.argv[1]
    else:
        article_url = input("Enter the Grokipedia article URL: ").strip()
    if not article_url:
        print("No URL provided.")
        return
    # Extract citation references
    citations = extract_citation_links(article_url)
    # Load the LLM
    try:
        llm = load_llm()
    except ImportError as e:
        print(str(e))
        return
    # Identify ultimate sources for each citation
    citation_sources = []
    for citation_links in citations:
        if not citation_links:
            citation_sources.append("none")
            continue
        url = citation_links[0]
        text = fetch_article_text(url)
        source = identify_source(text, llm)
        citation_sources.append(source)
    # Group citations by source
    source_map = {}
    for idx, src in enumerate(citation_sources, start=1):
        source_map.setdefault(src, []).append(idx)
    # Print the report
    print("Citation ultimate sources by citation number:")
    for i, src in enumerate(citation_sources, start=1):
        print(f"  Citation [{i}]: {src}")
    print("\nUnique ultimate sources and which citations they appear in:")
    for src, citation_nums in source_map.items():
        print(f"  {src}: cited in {citation_nums}")
    # Report actual unique source count excluding 'none'
    unique = [s for s in source_map.keys() if s != "none"]
    print(f"\nActual unique cited ultimate source count: {len(unique)}")

if __name__ == "__main__":
    main()
