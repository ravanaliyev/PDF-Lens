#!/usr/bin/env python3
"""
Main entry point for PDF Analysis.
Run with: python3 main.py [pdf_file] [--config config_file]
"""
import argparse
import sys
import os

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_parser import PDFParser, PDFParserError

def main():
    parser = argparse.ArgumentParser(description="Analyze a PDF file to extract structure, TOC, and content.")
    parser.add_argument("pdf_path", nargs="?", default="test.pdf", help="Path to the PDF file (default: test.pdf)")
    parser.add_argument("--config", default="config.json", help="Path to configuration file (default: config.json)")
    args = parser.parse_args()

    print(f"Analyzing: {args.pdf_path} using config: {args.config}")

    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found.")
        sys.exit(1)

    try:
        # Initialize parser
        p = PDFParser(args.pdf_path, args.config)
    except Exception as e:
        print(f"Error initializing parser: {e}")
        sys.exit(1)

    try:
        print("\n=== PDF ANALYSIS REPORT ===\n")

        # 1. TOC (metadata or visual via get_toc())
        print("1) TOC (get_toc() result):")
        try:
            toc = p.get_toc()
        except Exception as e:
            toc = []
            print(f"  Error getting TOC: {e}")

        print(f"  Total TOC entries: {len(toc)}")
        for i, e in enumerate(toc, 1):
            print(f"   {i:2d}. Level {e.get('level', '?')}: {e.get('title')} -> Page {e.get('page')}")

        # 2. Visual TOC explicit
        print("\n2) Visual TOC (explicit _extract_visual_toc()):")
        try:
            visual = p._extract_visual_toc()
        except Exception as e:
            visual = []
            print(f"  Error extracting visual TOC: {e}")
        print(f"  Visual TOC entries: {len(visual)}")
        for i, e in enumerate(visual, 1):
            print(f"   {i:2d}. Level {e.get('level', '?')}: {e.get('title')} -> Page {e.get('page')}")

        # 3. Content-based headings
        print("\n3) Content-based headings (force content detection):")
        try:
            content_headings = p.detect_headings_from_content(force_content=True)
        except Exception as e:
            content_headings = []
            print(f"  Error in content detection: {e}")
        print(f"  Total content-detected headings: {len(content_headings)}")
        for i, h in enumerate(content_headings, 1):
            print(f"   {i:2d}. Level {h.get('level','?')}: {h.get('text')} (Page {h.get('page_number')})")

        # 4. Pages stats: raw vs clean
        print("\n4) Pages & character statistics:")
        pages = p.get_pages()
        raw_total = sum(len(pg.get('raw_text','') or '') for pg in pages)
        clean_total = sum(len(pg.get('clean_text','') or '') for pg in pages)
        reduction = raw_total - clean_total
        reduction_pct = (reduction / raw_total * 100) if raw_total else 0
        print(f"  Pages: {len(pages)}")
        print(f"  Raw total characters:   {raw_total}")
        print(f"  Clean total characters: {clean_total}")
        print(f"  Reduction: {reduction} chars ({reduction_pct:.2f}% )")

        # 5. Noise (headers/footers)
        print("\n5) Detected repeating headers/footers (noise set):")
        try:
            noise = p.detect_repeating_headers_footers()
        except Exception as e:
            noise = set()
            print(f"  Error detecting noise: {e}")
        print(f"  Noise entries count: {len(noise)}")
        for s in list(noise)[:50]:
            print(f"   - {s}")

        # 6. First 3 pages text
        print("\n6) First 3 pages (clean text preview):")
        for pg in pages[:3]:
            print('\n' + '-'*80)
            print(f"Page {pg['page_number']}")
            print('-'*80)
            txt = pg.get('clean_text','') or ''
            # print first 1500 chars
            print(txt[:1500])
            if len(txt) > 1500:
                print(f"\n... (truncated, total {len(txt)} chars)")

        # 7. Topics and full content of first 3 topics
        print("\n7) Topics (get_text_by_topics) and full content of first 3 topics:")
        try:
            topics = p.get_text_by_topics()
        except Exception as e:
            topics = []
            print(f"  Error extracting topics: {e}")
        print(f"  Total topics: {len(topics)}")
        for i, t in enumerate(topics[:3], 1):
            title = t.get('topic_title')
            level = t.get('level', '?')
            start = t.get('start_line')
            end = t.get('end_line')
            content = t.get('content','')
            page = t.get('page_number', 'N/A')
            print('\n' + '='*100)
            print(f"Topic {i}: {title} (Level {level}) Page: {page} Lines: {start}-{end} Length: {len(content)} chars")
            print('-'*100)
            print(content)

        print('\n=== END OF REPORT ===')

    finally:
        p.close()

if __name__ == "__main__":
    main()
