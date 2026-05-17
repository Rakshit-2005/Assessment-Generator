import json
import re
from urllib.request import urlopen
from urllib.error import URLError

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"


def fetch_catalog(url=CATALOG_URL):
    try:
        with urlopen(url) as r:
            raw = r.read()
            text = raw.decode('utf-8', errors='replace')
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import re as _re
                cleaned = _re.sub(r'[\x00-\x1f]', ' ', text)
                return json.loads(cleaned)
    except URLError as e:
        raise RuntimeError(f"Failed to fetch catalog: {e}")


def normalize_product(p: dict) -> dict:
    # ensure keys exist and are normalized strings/lists
    return {
        'entity_id': p.get('entity_id'),
        'name': (p.get('name') or '').strip(),
        'description': (p.get('description') or '').strip(),
        'keys': p.get('keys') or [],
        'job_levels': p.get('job_levels') or [],
        'languages': p.get('languages') or [],
        'link': p.get('link') or p.get('url') or '',
        'duration': p.get('duration') or ''
    }


def load_catalog():
    raw = fetch_catalog()
    return [normalize_product(p) for p in raw]
