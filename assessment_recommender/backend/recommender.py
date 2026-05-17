from typing import List, Dict, Optional, Tuple
import logging
import math

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

from catalog_loader import load_catalog
try:
    from llm_adapter import parse_jd
    PARSER_AVAILABLE = True
except Exception:
    PARSER_AVAILABLE = False

logger = logging.getLogger(__name__)


class Recommender:
    def __init__(self, catalog: Optional[List[Dict]] = None, use_embeddings: bool = True, prefer_hf: bool = False):
        self.catalog = catalog or load_catalog()
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        self.index_docs = [ (p['name'] + ' ' + p['description']).strip() for p in self.catalog ]
        import os
        HF_TOKEN = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if HF_TOKEN:
            try:
                from hf_adapter import HfEmbedder
                HF_AVAILABLE = True
            except Exception:
                HF_AVAILABLE = False
        else:
            HF_AVAILABLE = False
        if self.use_embeddings:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.doc_embeddings = self.model.encode(self.index_docs, convert_to_numpy=True)
            except Exception as e:
                logger.warning('Failed to load embedding model, falling back to tfidf: %s', e)
                self.use_embeddings = False
        self.use_hf_api = prefer_hf and HF_AVAILABLE
        # if local not available and HF token present, initialize HF embedder
        if not self.use_embeddings and self.use_hf_api:
            try:
                self.hf = HfEmbedder()
                # fetch embeddings from HF for catalog
                import numpy as _np
                self.doc_embeddings = self.hf.embed(self.index_docs)
            except Exception as e:
                logger.warning('Failed to fetch embeddings from HF API: %s', e)
                self.use_hf_api = False
        if not self.use_embeddings:
            # fallback: tfidf
            self.vectorizer = TfidfVectorizer(stop_words='english')
            if self.index_docs:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.index_docs)
            else:
                self.tfidf_matrix = None

    def recommend(self, jd: str, top_n: int = 8) -> List[Dict]:
        # attempt to parse JD for inferred skills/level to boost scoring (rule-based local parser)
        inferred = parse_jd(jd) if PARSER_AVAILABLE else {}

        if self.use_embeddings:
            q_emb = self.model.encode([jd], convert_to_numpy=True)
            sims = cosine_similarity(q_emb, self.doc_embeddings)[0]
        elif self.use_hf_api:
            # use HF inference API for embeddings
            try:
                import numpy as _np
                q_emb = self.hf.embed([jd])
                # cosine similarity between (1,dim) and (n,dim)
                from sklearn.metrics.pairwise import cosine_similarity as _cos
                sims = _cos(q_emb, self.doc_embeddings)[0]
            except Exception:
                sims = [0.0]*len(self.catalog)
        else:
            if not jd or not self.tfidf_matrix:
                sims = [0.0]*len(self.catalog)
            else:
                q_vec = self.vectorizer.transform([jd])
                sims = _cos_sim(q_vec, self.tfidf_matrix)[0]

        scored = []
        for i, score in enumerate(sims):
            if score > 0:
                # boost score when product keys match inferred skills or job level
                boost = 0.0
                try:
                    skills = set([s.lower() for s in inferred.get('skills', [])])
                    level = inferred.get('level', '')
                except Exception:
                    skills = set()
                    level = ''

                p = self.catalog[i]
                p_keys = set([k.lower() for k in (p.get('keys') or []) if isinstance(k, str)])
                matches = skills.intersection(p_keys)
                if matches:
                    boost += 0.12 * len(matches)
                try:
                    p_levels = [l.lower() for l in (p.get('job_levels') or []) if isinstance(l, str)]
                    if level and level.lower() in p_levels:
                        boost += 0.08
                except Exception:
                    pass

                final_score = float(score) + boost
                scored.append((final_score, self.catalog[i]))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, p in scored[:top_n]:
            results.append({
                'score': round(float(s), 4),
                'name': p['name'],
                'link': p['link'],
                'keys': p.get('keys'),
                'duration': p.get('duration'),
                'languages': p.get('languages')
            })
        # if nothing matched, fall back to keyword heuristic using simple token overlap
        if not results:
            tokens = set(t.strip().lower() for t in jd.split() if len(t) > 2)
            hits = []
            for p in self.catalog:
                text = (p['name'] + ' ' + p['description']).lower()
                common = tokens.intersection(set(text.split()))
                if common:
                    hits.append((len(common), p))
            hits.sort(key=lambda x: x[0], reverse=True)
            for c, p in hits[:top_n]:
                results.append({'score': float(c), 'name': p['name'], 'link': p['link'], 'keys': p.get('keys'), 'duration': p.get('duration'), 'languages': p.get('languages')})

        return results
