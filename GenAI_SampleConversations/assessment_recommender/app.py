import argparse
import json
import re
from urllib.request import urlopen
from urllib.error import URLError

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"


STOPWORDS = {
    'and','the','with','for','a','an','to','of','in','on','by','will','be','we',
    'their','they','as','is','are','or','from','that','this','which','at'
}


def fetch_catalog(url=CATALOG_URL):
    try:
        with urlopen(url) as r:
            raw = r.read()
            # decode with replacement to avoid control-char failures
            text = raw.decode('utf-8', errors='replace')
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # try to remove problematic control characters (aggressive)
                import re as _re
                cleaned = _re.sub(r'[\x00-\x1f]', ' ', text)
                return json.loads(cleaned)
    except URLError as e:
        raise RuntimeError(f"Failed to fetch catalog: {e}")


def tokenize(text):
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9+#\.\-]+", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def score_product(jd_tokens, product):
    text = ' '.join(filter(None, [product.get('name',''), product.get('description','')] )).lower()
    keys = ' '.join(product.get('keys') or []).lower()
    levels = ' '.join(product.get('job_levels') or []).lower()
    score = 0
    matches = set()
    for t in jd_tokens:
        if t in text:
            score += 3
            matches.add(t)
        if t in keys:
            score += 2
            matches.add(t)
        if t in levels:
            score += 2
            matches.add(t)
        # partial matches
        if len(t) > 4 and t in product.get('name','').lower():
            score += 1
            matches.add(t)
    return score, sorted(matches)


def map_seniority(jd_text):
    jd = jd_text.lower()
    if 'senior' in jd or 'lead' in jd or 'principal' in jd:
        return 'Professional Individual Contributor'
    if 'manager' in jd or 'lead' in jd or 'director' in jd:
        return 'Manager'
    if 'graduate' in jd or 'entry' in jd or 'junior' in jd:
        return 'Entry-Level'
    return None


def recommend(jd_text, catalog, top_n=7):
    tokens = tokenize(jd_text)
    senior_level = map_seniority(jd_text)
    scored = []
    for p in catalog:
        s, matches = score_product(tokens, p)
        # small boost if job level matches inferred seniority
        if senior_level and p.get('job_levels') and senior_level in p.get('job_levels'):
            s += 2
        if s > 0:
            scored.append((s, matches, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for s, matches, p in scored[:top_n]:
        results.append({
            'name': p.get('name'),
            'link': p.get('link'),
            'keys': p.get('keys'),
            'duration': p.get('duration'),
            'languages': p.get('languages'),
            'score': s,
            'matches': matches
        })
    return results


def main():
    ap = argparse.ArgumentParser(description='Recommend SHL assessments from a JD')
    ap.add_argument('--jd', help='Job description text', type=str)
    ap.add_argument('--jd-file', help='Path to file containing JD text', type=str)
    ap.add_argument('--top', help='Top N results', type=int, default=8)
    args = ap.parse_args()
    if not args.jd and not args.jd_file:
        ap.error('provide --jd or --jd-file')
    jd_text = args.jd
    if args.jd_file:
        with open(args.jd_file, 'r', encoding='utf-8') as f:
            jd_text = f.read()

    print('Fetching product catalog (this may take a few seconds)...')
    catalog = fetch_catalog()
    print(f'Products in catalog: {len(catalog)}')
    results = recommend(jd_text, catalog, top_n=args.top)
    print('\nTop recommendations:\n')
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']} ({', '.join(r.get('keys') or [])})")
        print(f"   Link: {r['link']}")
        print(f"   Duration: {r.get('duration') or 'n/a'} | Languages: {', '.join(r.get('languages') or [])}")
        print(f"   Score: {r['score']} | Matches: {', '.join(r['matches'])}\n")


if __name__ == '__main__':
    main()
