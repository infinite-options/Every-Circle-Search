"""
Drop-in search demo (does NOT modify QdrntTest.py).

Same routes/shape as QdrntTest so the app can point SEARCH_BASE_URL here:
  GET /search_business
  GET /search_expertise
  GET /search_wishes
  GET /search_global
  GET /search_suggest

Ranking: Qdrant dense + BM25 fused with native RRF (same as production), plus
global type weights. Location is a hard radius filter only (no proximity boost).

Each hit includes normalized channel scores in score_breakdown:
  dense_score / semantic_score  — MiniLM cosine, clipped to 0–1
  sparse_score                  — BM25 min-max normalized to 0–1 (raw kept as sparse_score_raw)
  has_sparse_score              — true when BM25 returned this hit
  lexical_score                 — fuzzy mix, already 0–1
  dense_sparse_score            — Qdrant RRF of dense + BM25, 0–1
  semantic_lexical_score        — Python RRF of semantic + fuzzy lexical, 0–1
  legacy_boosts                 — old token/phrase boosts
  legacy_additive_score         — old additive mode: semantic + those boosts

Search endpoints return { results, search_categories } with four buckets:
  sparse    — has_sparse_score (wins if the hit is also an exact match)
  exact     — exact/contains match, no sparse score (e.g. "games" → GameStop)
  semantic  — neither of those, semantic_score > 0.4
  other     — everything else
All hits get passes_relevance_cutoff: true so the app shows every result.

Exact-match boost (demo only): multiplicative ×EXACT_MATCH_BOOST_FACTOR on the fused
RRF score when a query token is an exact document token or a substring of one.

Default port 5002 (SEARCH_COMPARE_PORT).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from qdrant_client.models import Document
from thefuzz import fuzz
import pymysql

import QdrntTest as qt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DEMO_PORT = int(os.getenv("SEARCH_COMPARE_PORT", "5002"))
LEGACY_SEMANTIC_PASS_MIN = qt._env_float("LEGACY_SEMANTIC_PASS_MIN", 0.38)
LEGACY_LEXICAL_PASS_MIN = qt._env_float("LEGACY_LEXICAL_PASS_MIN", 0.35)
LEGACY_RRF_K = qt._env_float("LEGACY_RRF_K", 60.0)
SEMANTIC_CATEGORY_MIN = qt._env_float("SEMANTIC_CATEGORY_MIN", 0.4)
# Whole-token equality ("game" in "board game"). Short tokens like "tv" still count.
EXACT_MATCH_MIN_TOKEN_LEN = int(os.getenv("EXACT_MATCH_MIN_TOKEN_LEN", "3"))
# Substring containment ("games" in "gamestop"). Longer floor avoids "art"→"party".
CONTAINS_MATCH_MIN_TOKEN_LEN = int(os.getenv("CONTAINS_MATCH_MIN_TOKEN_LEN", "4"))
# Multiplicative lift on fused RRF when query tokens match document text.
# 1.20 is a moderate content signal without flattening BM25 order.
EXACT_MATCH_BOOST_FACTOR = qt._env_float("EXACT_MATCH_BOOST_FACTOR", 1.20)

SEARCH_CATEGORY_META = (
    ("sparse", "Sparse (BM25)"),
    ("exact", "Exact match"),
    ("semantic", f"Semantic only (>{SEMANTIC_CATEGORY_MIN:g})"),
    ("other", "Other"),
)


def clamp01(value):
    num = qt.safe_float(value)
    if num is None:
        return None
    return max(0.0, min(1.0, num))


def minmax_norm_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {uid: 1.0 for uid in score_map}
    return {uid: (val - lo) / (hi - lo) for uid, val in score_map.items()}


def normalize_tokens(text):
    if text is None:
        return []
    s = str(text).strip().lower()
    if not s:
        return []
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return [t for t in cleaned.split() if t]


def row_match_text(row: dict, uid_field: str) -> str:
    """Same fields that get indexed: name/bio/tags for businesses, title+desc otherwise."""
    if uid_field == "business_uid":
        parts: List[Any] = [
            row.get("business_name"),
            row.get("business_short_bio"),
            row.get("business_tag_line"),
        ]
        for key in ("tags", "bs_tags", "bs_service_names", "custom_tags"):
            vals = row.get(key)
            if isinstance(vals, list):
                parts.extend(vals)
            elif vals:
                parts.append(vals)
        return " ".join(str(p) for p in parts if p)
    if uid_field == "profile_expertise_uid":
        return " ".join(
            str(p)
            for p in (row.get("profile_expertise_title"), row.get("profile_expertise_description"))
            if p
        )
    if uid_field == "profile_wish_uid":
        return " ".join(
            str(p) for p in (row.get("profile_wish_title"), row.get("profile_wish_description")) if p
        )
    return ""


def detect_exact_match(query: str, row: dict, uid_field: str) -> dict:
    """
    Exact token match or containment: query 'games' matches 'GameStop'.
    Kind is 'token' when any query token equals a document token, else 'contains'.
    """
    empty = {
        "has_exact_match": False,
        "exact_match_kind": None,
        "exact_match_tokens": [],
        "exact_match_boost_factor": None,
    }
    q_tokens = normalize_tokens(query)
    doc_tokens = normalize_tokens(row_match_text(row, uid_field))
    if not q_tokens or not doc_tokens:
        return empty

    token_hits = []
    contains_hits = []
    for q in q_tokens:
        if len(q) >= EXACT_MATCH_MIN_TOKEN_LEN and q in doc_tokens:
            token_hits.append(q)
            continue
        if len(q) >= CONTAINS_MATCH_MIN_TOKEN_LEN and any(q in tok and q != tok for tok in doc_tokens):
            contains_hits.append(q)

    q_phrase = " ".join(q_tokens)
    haystack = " ".join(doc_tokens)
    if (
        not token_hits
        and not contains_hits
        and len(q_phrase) >= CONTAINS_MATCH_MIN_TOKEN_LEN
        and q_phrase in haystack
    ):
        contains_hits.append(q_phrase)

    matched = token_hits or contains_hits
    if not matched:
        return empty
    return {
        "has_exact_match": True,
        "exact_match_kind": "token" if token_hits else "contains",
        "exact_match_tokens": matched,
        "exact_match_boost_factor": EXACT_MATCH_BOOST_FACTOR,
    }


def apply_exact_match_boost(rows: List[dict], query: str, uid_field: str) -> None:
    """Lift fused score after RRF when query tokens match document text."""
    for row in rows:
        details = detect_exact_match(query, row, uid_field)
        breakdown = row.get("score_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}
            row["score_breakdown"] = breakdown
        breakdown.update(details)
        if not details["has_exact_match"]:
            continue
        base = qt.safe_float(row.get("score")) or 0.0
        boosted = base * EXACT_MATCH_BOOST_FACTOR
        row["score"] = boosted
        breakdown["score_before_exact_match"] = base
        breakdown["final_score"] = boosted


def fuzzy_norm(query: str, text: Any) -> float:
    if not query or not text:
        return 0.0
    return fuzz.token_set_ratio(str(query), str(text)) / 100.0


def business_lexical_details(query: str, payload: dict) -> dict:
    """Legacy token/phrase boosts that used to be added on top of semantic score."""
    empty = {
        "token_tag": 0.0,
        "token_name": 0.0,
        "token_tagline": 0.0,
        "token_bio": 0.0,
        "phrase_name": 0.0,
        "phrase_tag": 0.0,
        "total_lexical_boost": 0.0,
    }
    q_tokens = normalize_tokens(query)
    if not q_tokens:
        return empty

    name_tokens = normalize_tokens(payload.get("business_name"))
    bio_tokens = normalize_tokens(payload.get("business_short_bio"))
    tag_line_tokens = normalize_tokens(payload.get("business_tag_line"))

    tag_tokens = []
    for key in ("tags", "bs_tags", "bs_service_names", "custom_tags"):
        for t in payload.get(key, []) or []:
            tag_tokens.extend(normalize_tokens(t))

    name_set = set(name_tokens)
    bio_set = set(bio_tokens)
    tag_line_set = set(tag_line_tokens)
    tag_set = set(tag_tokens)

    token_tag = token_name = token_tagline = token_bio = 0.0
    for q in q_tokens:
        if q in tag_set:
            token_tag += 0.22
        if q in name_set:
            token_name += 0.16
        if q in tag_line_set:
            token_tagline += 0.12
        if q in bio_set:
            token_bio += 0.06

    phrase_name = 0.0
    phrase_tag = 0.0
    q_str = " ".join(q_tokens)
    if q_str:
        if q_str in " ".join(name_tokens):
            phrase_name += 0.08
        if q_str in " ".join(tag_tokens):
            phrase_tag += 0.12

    total = token_tag + token_name + token_tagline + token_bio + phrase_name + phrase_tag
    return {
        "token_tag": token_tag,
        "token_name": token_name,
        "token_tagline": token_tagline,
        "token_bio": token_bio,
        "phrase_name": phrase_name,
        "phrase_tag": phrase_tag,
        "total_lexical_boost": total,
    }


def business_lexical_score(query: str, row: dict) -> float:
    name = row.get("business_name") or ""
    tagline = row.get("business_tag_line") or ""
    bio = row.get("business_short_bio") or ""
    service_names = qt.csv_to_tokens(row.get("bs_service_names") or row.get("all_service_names"))
    service_tags = qt.csv_to_tokens(row.get("bs_tags") or row.get("all_service_tags"))
    custom_tags = qt.csv_to_tokens(row.get("custom_tags") or row.get("all_custom_tags"))

    score = 0.0
    score += 0.30 * fuzzy_norm(query, name)
    score += 0.10 * fuzzy_norm(query, tagline)
    score += 0.08 * fuzzy_norm(query, bio)
    score += 0.22 * fuzzy_norm(query, " ".join(service_names))
    score += 0.18 * fuzzy_norm(query, " ".join(service_tags))
    score += 0.22 * fuzzy_norm(query, " ".join(custom_tags))
    return min(1.0, score)


def expertise_lexical_score(query: str, row: dict) -> float:
    title = row.get("profile_expertise_title") or ""
    desc = row.get("profile_expertise_description") or ""
    details = row.get("profile_expertise_details") or ""
    return min(
        1.0,
        0.55 * fuzzy_norm(query, title)
        + 0.30 * fuzzy_norm(query, desc)
        + 0.15 * fuzzy_norm(query, details),
    )


def wish_lexical_score(query: str, row: dict) -> float:
    title = row.get("profile_wish_title") or ""
    desc = row.get("profile_wish_description") or ""
    return min(1.0, 0.60 * fuzzy_norm(query, title) + 0.40 * fuzzy_norm(query, desc))


def parse_search_args():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    return {
        "query": query,
        "final_limit": qt.get_limit(limit_param, 99999),
        "candidate_limit": qt.hybrid_candidate_limit(qt.get_limit(limit_param, 99999)),
        "user_lat": qt.safe_float(request.args.get("user_lat")),
        "user_lon": qt.safe_float(request.args.get("user_lon")),
        "max_distance": qt.safe_float(request.args.get("max_distance")),
        "min_rating": qt.safe_float(request.args.get("min_rating")),
        "max_rating": qt.safe_float(request.args.get("max_rating")),
    }


def qdrant_named_search(collection_name: str, query, using: str, limit: int):
    response = qt.qdrant.query_points(
        collection_name=collection_name,
        query=query,
        using=using,
        limit=limit,
        with_payload=True,
    )
    points = getattr(response, "points", None)
    if points is not None:
        return points
    if isinstance(response, dict):
        return response.get("points", [])
    return []


def scores_by_uid(hits, uid_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        uid = payload.get(uid_field)
        score = qt.safe_float(getattr(hit, "score", None))
        if uid and score is not None:
            out[uid] = score
    return out


def ensure_synced(kind: str):
    qt.prepare_search_indexes()
    if kind == "business":
        qt.biz_map = qt.sync_businesses(qt.biz_map)
    elif kind == "expertise":
        qt.exp_map = qt.sync_expertise(qt.exp_map)
    elif kind == "wish":
        qt.wish_map = qt.sync_wishes(qt.wish_map)


def fetch_business_rows(uids: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not uids:
        return out
    conn = qt.mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    placeholders = ",".join(["%s"] * len(uids))
    cur.execute(
        f"""
        SELECT *
        FROM business
        WHERE business_uid IN ({placeholders})
          AND business_is_active = 1
        """,
        uids,
    )
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        row["business_latitude"] = qt.safe_float(row.get("business_latitude"))
        row["business_longitude"] = qt.safe_float(row.get("business_longitude"))
        row["business_google_rating"] = qt.safe_float(row.get("business_google_rating"))
        row["business_reward_amount"] = qt.safe_float(row.get("business_reward_amount"))
        row["business_price_level"] = qt.safe_int(row.get("business_price_level"))
        out[row["business_uid"]] = row
    return out


def fetch_expertise_rows(uids: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not uids:
        return out
    conn = qt.mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    placeholders = ",".join(["%s"] * len(uids))
    cur.execute(
        f"""
        SELECT profile_expertise.*,
               user_email_id,
               profile_personal_first_name, profile_personal_last_name,
               profile_personal_email_is_public, profile_personal_phone_number,
               profile_personal_phone_number_is_public,
               profile_personal_city, profile_personal_state, profile_personal_country,
               profile_personal_location_is_public,
               profile_personal_latitude, profile_personal_longitude,
               profile_personal_image, profile_personal_image_is_public,
               profile_personal_tag_line, profile_personal_tag_line_is_public,
               profile_personal_moderated
        FROM profile_expertise
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_expertise_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_expertise_uid IN ({placeholders})
          AND COALESCE(profile_expertise_moderated, 0) = 0
          AND COALESCE(profile_personal_moderated, 0) = 0
          AND profile_expertise_is_public = 1
        """,
        uids,
    )
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        row["profile_personal_latitude"] = qt.safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = qt.safe_float(row.get("profile_personal_longitude"))
        row["profile_expertise_latitude"] = qt.safe_float(row.get("profile_expertise_latitude"))
        row["profile_expertise_longitude"] = qt.safe_float(row.get("profile_expertise_longitude"))
        out[row["profile_expertise_uid"]] = row
    return out


def fetch_wish_rows(uids: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not uids:
        return out
    conn = qt.mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    placeholders = ",".join(["%s"] * len(uids))
    cur.execute(
        f"""
        SELECT profile_wish.*,
               user_email_id,
               profile_personal_first_name, profile_personal_last_name,
               profile_personal_email_is_public, profile_personal_phone_number,
               profile_personal_phone_number_is_public,
               profile_personal_city, profile_personal_state, profile_personal_country,
               profile_personal_location_is_public,
               profile_personal_latitude, profile_personal_longitude,
               profile_personal_image, profile_personal_image_is_public,
               profile_personal_tag_line, profile_personal_tag_line_is_public,
               profile_personal_moderated
        FROM profile_wish
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_wish_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_wish_uid IN ({placeholders})
          AND COALESCE(profile_wish_moderated, 0) = 0
          AND COALESCE(profile_personal_moderated, 0) = 0
          AND profile_wish_is_public = 1
        """,
        uids,
    )
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        row["profile_personal_latitude"] = qt.safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = qt.safe_float(row.get("profile_personal_longitude"))
        row["profile_wish_latitude"] = qt.safe_float(row.get("profile_wish_latitude"))
        row["profile_wish_longitude"] = qt.safe_float(row.get("profile_wish_longitude"))
        out[row["profile_wish_uid"]] = row
    return out


def apply_business_filters(rows: List[dict], args: dict) -> List[dict]:
    filtered = []
    for merged in rows:
        include, dist = qt.distance_filter_passes(
            args["user_lat"],
            args["user_lon"],
            args["max_distance"],
            merged.get("business_latitude"),
            merged.get("business_longitude"),
        )
        if not include:
            continue
        if dist is not None:
            merged["distance_miles"] = dist

        rating = qt.safe_float(merged.get("business_google_rating"))
        if rating is not None:
            if args["min_rating"] is not None and rating < args["min_rating"]:
                continue
            if args["max_rating"] is not None and rating > args["max_rating"]:
                continue
        filtered.append(merged)
    filtered.sort(key=lambda x: qt.safe_float(x.get("score")) or 0.0, reverse=True)
    return filtered


def apply_expertise_filters(rows: List[dict], args: dict) -> List[dict]:
    filtered = []
    for obj in rows:
        exp_lat, exp_lon = qt.expertise_effective_coords(obj)
        include, dist = qt.distance_filter_passes(
            args["user_lat"], args["user_lon"], args["max_distance"], exp_lat, exp_lon
        )
        if not include:
            continue
        if dist is not None:
            obj["distance_miles"] = dist
        filtered.append(obj)
    filtered.sort(key=lambda x: qt.safe_float(x.get("score")) or 0.0, reverse=True)
    return filtered


def apply_wish_filters(rows: List[dict], args: dict) -> List[dict]:
    filtered = []
    for obj in rows:
        wish_lat, wish_lon = qt.wish_effective_coords(obj)
        include, dist = qt.distance_filter_passes(
            args["user_lat"], args["user_lon"], args["max_distance"], wish_lat, wish_lon
        )
        if not include:
            continue
        if dist is not None:
            obj["distance_miles"] = dist
        filtered.append(obj)
    filtered.sort(key=lambda x: qt.safe_float(x.get("score")) or 0.0, reverse=True)
    return filtered


def hit_passes_demo_cutoff(row: dict) -> bool:
    breakdown = row.get("score_breakdown") or {}
    dense = qt.safe_float(breakdown.get("dense_score"))
    semantic = qt.safe_float(breakdown.get("semantic_score"))
    lexical = qt.safe_float(breakdown.get("lexical_score"))
    fused = qt.safe_float(row.get("score")) or 0.0
    if dense is not None and dense >= LEGACY_SEMANTIC_PASS_MIN:
        return True
    if semantic is not None and semantic >= LEGACY_SEMANTIC_PASS_MIN:
        return True
    if lexical is not None and lexical >= LEGACY_LEXICAL_PASS_MIN:
        return True
    return fused >= qt.MIN_SIMILARITY_SCORE


def annotate_cutoff(rows: List[dict]) -> List[dict]:
    if not rows:
        return rows
    for row in rows:
        row["passes_relevance_cutoff"] = hit_passes_demo_cutoff(row)
    if not any(r.get("passes_relevance_cutoff") for r in rows):
        best = max(rows, key=lambda r: qt.safe_float(r.get("score")) or 0.0)
        best["passes_relevance_cutoff"] = True
    return rows


def python_rrf_scores(sem_scores: List[Optional[float]], lex_scores: List[Optional[float]], uids: List[str]) -> List[float]:
    """Legacy Python RRF over semantic vs fuzzy-lexical ranks, normalized to ~[0, 1]."""
    n = len(sem_scores)
    if n == 0:
        return []

    sem_order = sorted(
        range(n),
        key=lambda i: (-(sem_scores[i] or 0.0), -(lex_scores[i] or 0.0), uids[i]),
    )
    lex_order = sorted(
        range(n),
        key=lambda i: (-(lex_scores[i] or 0.0), -(sem_scores[i] or 0.0), uids[i]),
    )
    rank_sem = [0] * n
    rank_lex = [0] * n
    for pos, i in enumerate(sem_order):
        rank_sem[i] = pos + 1
    for pos, i in enumerate(lex_order):
        rank_lex[i] = pos + 1

    k = LEGACY_RRF_K
    max_raw = (2.0 / (k + 1.0)) if k > 0 else 0.0
    out = []
    for i in range(n):
        raw = (1.0 / (k + rank_sem[i])) + (1.0 / (k + rank_lex[i]))
        out.append(min(1.0, (raw / max_raw) if max_raw > 0 else 0.0))
    return out


def attach_channel_scores(
    rows: List[dict],
    query: str,
    uid_field: str,
    lexical_fn: Callable[[str, dict], float],
    dense_map: Dict[str, float],
    sparse_map: Dict[str, float],
    include_business_boosts: bool,
) -> List[dict]:
    dense_norm = {uid: clamp01(score) for uid, score in dense_map.items()}
    sparse_norm = minmax_norm_map(sparse_map)

    sem_list = []
    lex_list = []
    uid_list = []
    prepared = []

    for row in rows:
        uid = row.get(uid_field)
        dense_raw = dense_map.get(uid) if uid else None
        sparse_raw = sparse_map.get(uid) if uid else None
        dense = dense_norm.get(uid) if uid else None
        sparse = sparse_norm.get(uid) if uid else None
        lexical = lexical_fn(query, row)
        semantic = dense

        token_boosts = business_lexical_details(query, row) if include_business_boosts else {
            "token_tag": 0.0,
            "token_name": 0.0,
            "token_tagline": 0.0,
            "token_bio": 0.0,
            "phrase_name": 0.0,
            "phrase_tag": 0.0,
            "total_lexical_boost": 0.0,
        }
        scaled_lex_boost = 0.0 if include_business_boosts else lexical * 0.25
        additive = (semantic or 0.0) + token_boosts["total_lexical_boost"] + scaled_lex_boost

        dense_sparse = clamp01(row.get("score"))
        breakdown = row.get("score_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}
        has_sparse = bool(uid and uid in sparse_map)
        breakdown.update(
            {
                "has_sparse_score": has_sparse,
                "dense_score_raw": dense_raw,
                "sparse_score_raw": sparse_raw,
                "dense_score": dense,
                "sparse_score": sparse if has_sparse else None,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "lexical_fuzzy_score": lexical,
                "dense_sparse_score": dense_sparse,
                "legacy_additive_score": additive,
                "legacy_boosts": {
                    **token_boosts,
                    "scaled_lexical_boost": scaled_lex_boost,
                },
            }
        )
        row["score_breakdown"] = breakdown
        prepared.append(row)
        sem_list.append(semantic or 0.0)
        lex_list.append(lexical or 0.0)
        uid_list.append(str(uid or ""))

    fused_sem_lex = python_rrf_scores(sem_list, lex_list, uid_list)
    for row, fused in zip(prepared, fused_sem_lex):
        row["score_breakdown"]["semantic_lexical_score"] = fused

    return prepared


def classify_search_result(row: dict) -> str:
    """Mutually exclusive buckets: sparse > exact > semantic > other."""
    breakdown = row.get("score_breakdown") or {}
    if breakdown.get("has_sparse_score"):
        return "sparse"
    if breakdown.get("has_exact_match"):
        return "exact"
    semantic = qt.safe_float(breakdown.get("semantic_score"))
    if semantic is not None and semantic > SEMANTIC_CATEGORY_MIN:
        return "semantic"
    return "other"


def build_search_categories(rows: List[dict]) -> List[dict]:
    """Annotate rows and return collapsible category metadata for the client."""
    counts = {"sparse": 0, "exact": 0, "semantic": 0, "other": 0}
    for row in rows:
        category = classify_search_result(row)
        row["search_result_category"] = category
        row["passes_relevance_cutoff"] = True
        counts[category] += 1
    return [
        {"id": cat_id, "title": title, "count": counts[cat_id]}
        for cat_id, title in SEARCH_CATEGORY_META
    ]


def jsonify_categorized(rows: List[dict]):
    categories = build_search_categories(rows)
    return jsonify({"results": rows, "search_categories": categories})


def retrieve_hybrid_with_channels(
    collection_name: str,
    uid_field: str,
    query: str,
    candidate_limit: int,
    fetch_rows: Callable[[List[str]], Dict[str, dict]],
    lexical_fn: Callable[[str, dict], float],
) -> List[dict]:
    vector = qt.embed_text(query)
    bm25_query = Document(text=query or "", model=qt.BM25_MODEL)

    hybrid_hits = qt.qdrant_hybrid_search(collection_name, query, vector, candidate_limit)
    dense_hits = qdrant_named_search(collection_name, vector, qt.DENSE_VECTOR_NAME, candidate_limit)
    sparse_hits = qdrant_named_search(collection_name, bm25_query, qt.BM25_VECTOR_NAME, candidate_limit)

    dense_map = scores_by_uid(dense_hits, uid_field)
    sparse_map = scores_by_uid(sparse_hits, uid_field)

    uids = [h.payload.get(uid_field) for h in hybrid_hits if getattr(h, "payload", None)]
    sql_rows = fetch_rows(uids)
    rows = qt.merge_hybrid_hits(hybrid_hits, uid_field, sql_rows)
    attach_channel_scores(
        rows,
        query,
        uid_field,
        lexical_fn,
        dense_map,
        sparse_map,
        include_business_boosts=(uid_field == "business_uid"),
    )
    apply_exact_match_boost(rows, query, uid_field)
    annotate_cutoff(rows)
    return rows


def search_business_rows(args: dict) -> List[dict]:
    ensure_synced("business")
    query = args["query"]
    if qt.is_browse_query(query):
        filtered = qt.fetch_browse_businesses(
            args["user_lat"], args["user_lon"], args["max_distance"], args["min_rating"], args["max_rating"]
        )
        return filtered[: args["final_limit"]]

    rows = retrieve_hybrid_with_channels(
        "businesses",
        "business_uid",
        query,
        args["candidate_limit"],
        fetch_business_rows,
        business_lexical_score,
    )
    rows = apply_business_filters(rows, args)
    return rows[: args["final_limit"]]


def search_expertise_rows(args: dict) -> List[dict]:
    ensure_synced("expertise")
    query = args["query"]
    if qt.is_browse_query(query):
        return qt.fetch_browse_expertise(args["user_lat"], args["user_lon"], args["max_distance"])[
            : args["final_limit"]
        ]

    rows = retrieve_hybrid_with_channels(
        "expertise",
        "profile_expertise_uid",
        query,
        args["candidate_limit"],
        fetch_expertise_rows,
        expertise_lexical_score,
    )
    rows = apply_expertise_filters(rows, args)
    return rows[: args["final_limit"]]


def search_wish_rows(args: dict) -> List[dict]:
    ensure_synced("wish")
    query = args["query"]
    if qt.is_browse_query(query):
        return qt.fetch_browse_wishes(args["user_lat"], args["user_lon"], args["max_distance"])[
            : args["final_limit"]
        ]

    rows = retrieve_hybrid_with_channels(
        "wishes",
        "profile_wish_uid",
        query,
        args["candidate_limit"],
        fetch_wish_rows,
        wish_lexical_score,
    )
    rows = apply_wish_filters(rows, args)
    return rows[: args["final_limit"]]


@app.route("/search_business", methods=["GET"])
def search_business():
    return jsonify_categorized(search_business_rows(parse_search_args()))


@app.route("/search_expertise", methods=["GET"])
def search_expertise():
    return jsonify_categorized(search_expertise_rows(parse_search_args()))


@app.route("/search_wishes", methods=["GET"])
def search_wishes():
    return jsonify_categorized(search_wish_rows(parse_search_args()))


@app.route("/search_global", methods=["GET"])
def search_global():
    args = parse_search_args()
    query = args["query"]

    if qt.is_browse_query(query):
        qt.prepare_search_indexes()
        business_results = [
            {**row, "itemType": "businesses"}
            for row in qt.fetch_browse_businesses(
                args["user_lat"], args["user_lon"], args["max_distance"], args["min_rating"], args["max_rating"]
            )
        ]
        expertise_results = [
            {**row, "itemType": "expertise"}
            for row in qt.fetch_browse_expertise(args["user_lat"], args["user_lon"], args["max_distance"])
        ]
        seeking_results = [
            {**row, "itemType": "seeking"}
            for row in qt.fetch_browse_wishes(args["user_lat"], args["user_lon"], args["max_distance"])
        ]
        combined = business_results + expertise_results + seeking_results
        return jsonify_categorized(combined[: args["final_limit"]])

    business_results = [{**row, "itemType": "businesses"} for row in search_business_rows(args)]
    expertise_results = [{**row, "itemType": "expertise"} for row in search_expertise_rows(args)]
    seeking_results = [{**row, "itemType": "seeking"} for row in search_wish_rows(args)]

    for item in business_results:
        item["global_score"] = (qt.safe_float(item.get("score")) or 0.0) * qt.GLOBAL_BUSINESS_WEIGHT
    for item in expertise_results:
        item["global_score"] = (qt.safe_float(item.get("score")) or 0.0) * qt.GLOBAL_EXPERTISE_WEIGHT
    for item in seeking_results:
        item["global_score"] = (qt.safe_float(item.get("score")) or 0.0) * qt.GLOBAL_SEEKING_WEIGHT

    combined = business_results + expertise_results + seeking_results
    combined.sort(key=lambda x: qt.safe_float(x.get("global_score")) or 0.0, reverse=True)
    return jsonify_categorized(combined[: args["final_limit"]])


@app.route("/search_suggest", methods=["GET"])
def search_suggest():
    return qt.search_suggest()


if __name__ == "__main__":
    qt.prepare_search_indexes()
    print(f"Search compare demo on http://0.0.0.0:{DEMO_PORT}")
    print("Drop-in routes: /search_business /search_expertise /search_wishes /search_global /search_suggest")
    app.run(host="0.0.0.0", port=DEMO_PORT)
