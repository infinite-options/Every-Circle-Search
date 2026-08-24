import os
import pymysql
import uuid
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Modifier,
    PointStruct,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseVectorParams,
    VectorParams,
)

# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))

MYSQL = dict(
    host=os.environ["RDS_HOST"],
    port=int(os.environ["RDS_PORT"]),
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PW"],
    database=os.environ["RDS_DB"]
)

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
EMBED_DIM = 384
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"
COLLECTION_NAMES = ("businesses", "wishes", "expertise")

# CUDA can be advertised as available yet fail at runtime ("no kernel image for device")
# when the PyTorch CUDA build does not match the GPU/driver. Default to CPU; set
# SEARCH_EMBEDDING_DEVICE=cuda on hosts where GPU + torch are known good.
SEARCH_EMBEDDING_DEVICE = (
    os.getenv("SEARCH_EMBEDDING_DEVICE") or os.getenv("TORCH_EMBED_DEVICE") or "cpu"
).strip().lower()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
embedder = SentenceTransformer(MODEL_NAME, device=SEARCH_EMBEDDING_DEVICE)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# In-memory sync fingerprints (uid -> version). Reset when hybrid collections are recreated.
biz_map = {}
wish_map = {}
exp_map = {}


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except:
        return default


MIN_SIMILARITY_SCORE = _env_float("MIN_SIMILARITY_SCORE", 0.35)
# Qdrant RRF uses k=60 by default; used only to normalize fused scores to ~[0, 1].
RRF_K = _env_float("RRF_K", 60.0)
HYBRID_TOP_K = int(_env_float("HYBRID_TOP_K", 100))
HYBRID_TOP_K_MAX = int(_env_float("HYBRID_TOP_K_MAX", 500))
GLOBAL_BUSINESS_WEIGHT = _env_float("GLOBAL_BUSINESS_WEIGHT", 1.15)
GLOBAL_EXPERTISE_WEIGHT = _env_float("GLOBAL_EXPERTISE_WEIGHT", 0.85)
GLOBAL_SEEKING_WEIGHT = _env_float("GLOBAL_SEEKING_WEIGHT", 0.85)
SEARCH_DEFAULT_LIMIT = int(_env_float("SEARCH_DEFAULT_LIMIT", 120))
# Category + exact-match boost (mirrors SearchCompareDemo).
SEMANTIC_CATEGORY_MIN = _env_float("SEMANTIC_CATEGORY_MIN", 0.4)
EXACT_MATCH_MIN_TOKEN_LEN = int(os.getenv("EXACT_MATCH_MIN_TOKEN_LEN", "3"))
CONTAINS_MATCH_MIN_TOKEN_LEN = int(os.getenv("CONTAINS_MATCH_MIN_TOKEN_LEN", "4"))
EXACT_MATCH_BOOST_FACTOR = _env_float("EXACT_MATCH_BOOST_FACTOR", 1.20)

SEARCH_CATEGORY_META = (
    ("sparse", "Sparse (BM25)"),
    ("exact", "Exact match"),
    ("semantic", f"Semantic only (>{SEMANTIC_CATEGORY_MIN:g})"),
    ("other", "Other"),
)


def hybrid_candidate_limit(final_limit):
    """Bound Qdrant hybrid retrieval; never pull the full collection."""
    target = max(int(final_limit or HYBRID_TOP_K), HYBRID_TOP_K)
    return max(1, min(target, HYBRID_TOP_K_MAX))


def normalize_rrf_score(raw_score):
    """Normalize two-channel Qdrant RRF score into approximately [0, 1].

    Must use the same k as qdrant_hybrid_search (RRF_K). FusionQuery(RRF) defaults
    to k=2 on the server; we pass RRF_K explicitly so scores do not all clamp to 1.0.
    """
    max_raw = (2.0 / (RRF_K + 1.0)) if RRF_K > 0 else 0.0
    raw = safe_float(raw_score) or 0.0
    if max_raw <= 0:
        return 0.0
    return min(1.0, raw / max_raw)


def qdrant_hybrid_search(collection_name, query_text, query_vector, limit):
    """
    Dense + BM25 prefetch, fused with Qdrant native RRF.
    Requires fastembed for local BM25 Document encoding.
    """
    rrf_k = max(1, int(RRF_K))
    response = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=limit,
            ),
            Prefetch(
                query=Document(text=query_text or "", model=BM25_MODEL),
                using=BM25_VECTOR_NAME,
                limit=limit,
            ),
        ],
        query=RrfQuery(rrf=Rrf(k=rrf_k)),
        limit=limit,
        with_payload=True,
    )
    points = getattr(response, "points", None)
    if points is not None:
        return points
    if isinstance(response, dict):
        return response.get("points", [])
    return []


def build_point_vectors(text):
    searchable = "" if text is None else str(text)
    return {
        DENSE_VECTOR_NAME: embed_text(searchable),
        BM25_VECTOR_NAME: Document(text=searchable, model=BM25_MODEL),
    }


def hit_passes_relevance_cutoff(hit):
    score = safe_float(getattr(hit, "score", None)) or 0.0
    return score >= MIN_SIMILARITY_SCORE


def filter_relevant_hits(hits):
    """
    Annotate each hit with passes_relevance_cutoff (keep full list for "Show more").
    If none pass, mark the single best hit so the default UI is not empty.
    """
    if not hits:
        return []

    for hit in hits:
        passes = hit_passes_relevance_cutoff(hit)
        payload = getattr(hit, "payload", None)
        if isinstance(payload, dict):
            payload["passes_relevance_cutoff"] = passes

    any_pass = any(
        isinstance(getattr(h, "payload", None), dict) and h.payload.get("passes_relevance_cutoff") for h in hits
    )
    if not any_pass and hits:
        best = max(hits, key=lambda h: safe_float(getattr(h, "score", None)) or 0.0)
        payload = getattr(best, "payload", None)
        if isinstance(payload, dict):
            payload["passes_relevance_cutoff"] = True

    return hits


def filter_rescored_candidates(merged_candidates):
    class _Hit:
        def __init__(self, payload):
            self.payload = payload
            self.score = payload.get("score", 0.0)

    return [h.payload for h in filter_relevant_hits([_Hit(m) for m in merged_candidates])]


def merge_hybrid_hits(results, uid_field, additional_info):
    """Attach normalized Qdrant RRF scores and merge SQL enrichment rows."""
    merged_candidates = []
    for r in results:
        payload = getattr(r, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        uid = payload.get(uid_field)
        if additional_info is not None and (not uid or uid not in additional_info):
            continue

        raw = safe_float(getattr(r, "score", None)) or 0.0
        norm = normalize_rrf_score(raw)
        merged = {**payload}
        if uid and additional_info and uid in additional_info:
            merged.update(additional_info[uid])
        merged["score"] = norm
        merged["score_breakdown"] = {
            "fusion": "rrf",
            "rescore_mode": "qdrant_hybrid",
            "rrf_raw": raw,
            "final_score": norm,
        }
        merged_candidates.append(merged)

    return filter_rescored_candidates(merged_candidates)


def clamp01(value):
    num = safe_float(value)
    if num is None:
        return None
    return max(0.0, min(1.0, num))


def minmax_norm_map(score_map):
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


def qdrant_named_search(collection_name, query, using, limit):
    response = qdrant.query_points(
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


def scores_by_uid(hits, uid_field):
    out = {}
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        if not isinstance(payload, dict):
            continue
        uid = payload.get(uid_field)
        score = safe_float(getattr(hit, "score", None))
        if uid and score is not None:
            out[uid] = score
    return out


def row_match_text(row, uid_field):
    """Same fields that get indexed: name/bio/tags for businesses, title+desc otherwise."""
    if uid_field == "business_uid":
        parts = [
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


def detect_exact_match(query, row, uid_field):
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


def apply_exact_match_boost(rows, query, uid_field):
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
        base = safe_float(row.get("score")) or 0.0
        boosted = base * EXACT_MATCH_BOOST_FACTOR
        row["score"] = boosted
        breakdown["score_before_exact_match"] = base
        breakdown["final_score"] = boosted


def annotate_hybrid_channels(rows, query, uid_field, collection_name, candidate_limit):
    """
    Attach dense/sparse channel flags for categorization, then exact-match boost.
    Sparse membership comes from a BM25-only query; semantic_score from dense cosine.
    """
    if not rows:
        return rows

    vector = embed_text(query)
    bm25_query = Document(text=query or "", model=BM25_MODEL)
    dense_hits = qdrant_named_search(collection_name, vector, DENSE_VECTOR_NAME, candidate_limit)
    sparse_hits = qdrant_named_search(collection_name, bm25_query, BM25_VECTOR_NAME, candidate_limit)
    dense_map = scores_by_uid(dense_hits, uid_field)
    sparse_map = scores_by_uid(sparse_hits, uid_field)
    dense_norm = {uid: clamp01(score) for uid, score in dense_map.items()}
    sparse_norm = minmax_norm_map(sparse_map)

    for row in rows:
        uid = row.get(uid_field)
        breakdown = row.get("score_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}
            row["score_breakdown"] = breakdown
        has_sparse = bool(uid and uid in sparse_map)
        dense = dense_norm.get(uid) if uid else None
        sparse = sparse_norm.get(uid) if uid else None
        breakdown.update(
            {
                "has_sparse_score": has_sparse,
                "dense_score_raw": dense_map.get(uid) if uid else None,
                "sparse_score_raw": sparse_map.get(uid) if uid else None,
                "dense_score": dense,
                "sparse_score": sparse if has_sparse else None,
                "semantic_score": dense,
                "dense_sparse_score": clamp01(row.get("score")),
            }
        )

    apply_exact_match_boost(rows, query, uid_field)
    return rows


def classify_search_result(row):
    """Mutually exclusive buckets: sparse > exact > semantic > other."""
    breakdown = row.get("score_breakdown") or {}
    if breakdown.get("has_sparse_score"):
        return "sparse"
    if breakdown.get("has_exact_match"):
        return "exact"
    semantic = safe_float(breakdown.get("semantic_score"))
    if semantic is not None and semantic > SEMANTIC_CATEGORY_MIN:
        return "semantic"
    return "other"


def build_search_categories(rows):
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


def jsonify_categorized(rows):
    categories = build_search_categories(rows)
    return jsonify({"results": rows, "search_categories": categories})


def csv_to_tokens(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(csv_to_tokens(v))
        return out
    parts = [p.strip().lower() for p in str(value).split(",") if p and p.strip()]
    return parts


# ---------------------------------------------------------
# SAFE CONVERSION HELPERS (bulletproof)
# ---------------------------------------------------------

def owner_profile_is_publicly_visible(row):
    """Hide content whose owner profile is taken down / pending review / acknowledged."""
    if not row:
        return True
    try:
        moderated = int(row.get("profile_personal_moderated") or 0)
    except (TypeError, ValueError):
        moderated = 0
    return moderated == 0


def content_row_is_publicly_visible(row, content_moderated_key):
    if not owner_profile_is_publicly_visible(row):
        return False
    try:
        moderated = int(row.get(content_moderated_key) or 0)
    except (TypeError, ValueError):
        moderated = 0
    return moderated == 0


def safe_float(value):
    """
    Safely convert value to float.
    Returns None for invalid, empty, or malformed numeric input.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    try:
        return float(s)
    except:
        return None


def safe_int(value):
    """
    Safely convert value to int.
    Returns None for invalid or empty input.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value

    s = str(value).strip()
    if s == "":
        return None

    try:
        return int(s)
    except:
        return None


# ---------------------------------------------------------
# LIMIT LOGIC
# ---------------------------------------------------------
def get_limit(param, max_results):
    if param is None or param == "":
        return SEARCH_DEFAULT_LIMIT

    value = str(param).strip().upper()

    if value == "ALL":
        return max_results

    if value.isdigit():
        return int(value)

    return SEARCH_DEFAULT_LIMIT


# ---------------------------------------------------------
# Haversine Distance (safe)
# ---------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Returns distance in miles.
    Returns None if coordinates are missing or invalid.
    """

    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)

    if None in (lat1, lon1, lat2, lon2):
        return None

    R = 3958.8  # Earth radius (miles)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def _coords_usable(lat, lon):
    lat = safe_float(lat)
    lon = safe_float(lon)
    if lat is None or lon is None:
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    return True


def expertise_effective_coords(row):
    """Use offering address coords when set; otherwise seller home coords."""
    lat = safe_float(row.get("profile_expertise_latitude"))
    lon = safe_float(row.get("profile_expertise_longitude"))
    if _coords_usable(lat, lon):
        return lat, lon
    return safe_float(row.get("profile_personal_latitude")), safe_float(
        row.get("profile_personal_longitude")
    )


def wish_effective_coords(row):
    """Use seeking address coords when set; otherwise seeker home coords."""
    lat = safe_float(row.get("profile_wish_latitude"))
    lon = safe_float(row.get("profile_wish_longitude"))
    if _coords_usable(lat, lon):
        return lat, lon
    return safe_float(row.get("profile_personal_latitude")), safe_float(
        row.get("profile_personal_longitude")
    )


def distance_filter_passes(user_lat, user_lon, max_distance, target_lat, target_lon):
    """
    Returns (include_row, distance_miles).
    When max_distance is set, rows without target coordinates are excluded.
    """
    if user_lat is None or user_lon is None:
        return True, None
    if not _coords_usable(target_lat, target_lon):
        if max_distance is not None:
            return False, None
        return True, None
    dist = haversine_miles(user_lat, user_lon, target_lat, target_lon)
    if max_distance is not None:
        if dist is None:
            return False, None
        if dist > max_distance:
            return False, dist
    return True, dist


def is_browse_query(query):
    """Empty query = browse all public/active catalog items (no semantic search)."""
    return not (query or "").strip()


def fetch_browse_businesses(user_lat, user_lon, max_distance, min_rating, max_rating):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT
            b.*,
            GROUP_CONCAT(DISTINCT bs.bs_service_name) AS all_service_names,
            GROUP_CONCAT(DISTINCT bs.bs_tags) AS all_service_tags,
            GROUP_CONCAT(DISTINCT t.tag_name) AS all_custom_tags
        FROM business b
        LEFT JOIN business_services bs ON bs.bs_business_id = b.business_uid
        LEFT JOIN business_tags bt ON bt.bt_business_id = b.business_uid
        LEFT JOIN tags t ON t.tag_uid = bt.bt_tag_id
        WHERE b.business_is_active = 1
        GROUP BY b.business_uid
        """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        row["business_latitude"] = safe_float(row.get("business_latitude"))
        row["business_longitude"] = safe_float(row.get("business_longitude"))
        row["business_google_rating"] = safe_float(row.get("business_google_rating"))
        row["bs_service_names"] = csv_to_tokens(row.get("all_service_names"))
        row["bs_tags"] = csv_to_tokens(row.get("all_service_tags"))
        row["custom_tags"] = csv_to_tokens(row.get("all_custom_tags"))
        row["score"] = 1.0
        row["score_breakdown"] = {"browse_mode": True, "final_score": 1.0}

        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            row.get("business_latitude"),
            row.get("business_longitude"),
        )
        if not include:
            continue
        if dist is not None:
            row["distance_miles"] = dist

        rating = safe_float(row.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        results.append(row)
    return results


def fetch_browse_expertise(user_lat, user_lon, max_distance):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
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
        WHERE profile_expertise.profile_expertise_is_public = 1
          AND COALESCE(profile_expertise.profile_expertise_moderated, 0) = 0
          AND COALESCE(profile_personal.profile_personal_moderated, 0) = 0
        """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        row["profile_expertise_latitude"] = safe_float(row.get("profile_expertise_latitude"))
        row["profile_expertise_longitude"] = safe_float(row.get("profile_expertise_longitude"))
        row["score"] = 1.0
        row["score_breakdown"] = {"browse_mode": True, "final_score": 1.0}

        exp_lat, exp_lon = expertise_effective_coords(row)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            exp_lat,
            exp_lon,
        )
        if not include:
            continue
        if dist is not None:
            row["distance_miles"] = dist

        results.append(row)
    return results


def fetch_browse_wishes(user_lat, user_lon, max_distance):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
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
        WHERE profile_wish.profile_wish_is_public = 1
          AND COALESCE(profile_wish.profile_wish_moderated, 0) = 0
          AND COALESCE(profile_personal.profile_personal_moderated, 0) = 0
        """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        row["profile_wish_latitude"] = safe_float(row.get("profile_wish_latitude"))
        row["profile_wish_longitude"] = safe_float(row.get("profile_wish_longitude"))
        row["score"] = 1.0
        row["score_breakdown"] = {"browse_mode": True, "final_score": 1.0}

        wish_lat, wish_lon = wish_effective_coords(row)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            wish_lat,
            wish_lon,
        )
        if not include:
            continue
        if dist is not None:
            row["distance_miles"] = dist

        results.append(row)
    return results


# ---------------------------------------------------------
# EMBEDDING + MYSQL HELPERS
# ---------------------------------------------------------
def embed_text(text: str):
    s = "" if text is None else str(text)
    return embedder.encode(s, convert_to_numpy=True).tolist()

def make_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

def mysql_connect():
    return pymysql.connect(**MYSQL)

# ---------------------------------------------------------
# VERIFY QDRANT INSERT (optional, kept safe)
# ---------------------------------------------------------
def verify_qdrant_insert(collection, uid_key, uid_value):
    try:
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter={"must": [{"key": uid_key, "match": {"value": uid_value}}]},
            limit=1
        )
        return len(points) > 0
    except:
        return False


def get_existing_columns(cursor, table_name):
    """
    Return a set of existing column names for a table.
    """
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    rows = cursor.fetchall()
    cols = set()
    for row in rows:
        name = row.get("Field") if isinstance(row, dict) else None
        if name:
            cols.add(name)
    return cols


def pick_sync_timestamp_column(existing_columns, candidates):
    """
    Pick first existing timestamp-ish column from candidates.
    """
    for col in candidates:
        if col in existing_columns:
            return col
    return None


# ---------------------------------------------------------
# ENSURE COLLECTIONS (dense + BM25 sparse)
# ---------------------------------------------------------
def collection_supports_hybrid(collection_name):
    """True when collection has named dense + bm25 sparse vectors."""
    try:
        info = qdrant.get_collection(collection_name)
        params = info.config.params
        vectors = params.vectors
        sparse = params.sparse_vectors
        if not sparse or BM25_VECTOR_NAME not in sparse:
            return False
        if isinstance(vectors, dict):
            return DENSE_VECTOR_NAME in vectors
        return False
    except Exception:
        return False


_hybrid_collections_verified = False


def ensure_collections():
    """
    Ensure hybrid collections exist.
    Returns True if any collection was created or recreated (caller should reset sync maps).
    """
    recreated = False
    for col in COLLECTION_NAMES:
        if qdrant.collection_exists(col):
            if collection_supports_hybrid(col):
                continue
            print(f"♻️ Recreating Qdrant collection '{col}' for dense+bm25 hybrid...")
            qdrant.delete_collection(col)
            recreated = True

        print(f"🆕 Creating Qdrant collection '{col}' (dense+bm25)...")
        qdrant.create_collection(
            collection_name=col,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                BM25_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        recreated = True
        print(f"✅ Collection '{col}' ready (hybrid).")

    if not recreated:
        print("✅ Hybrid collections ready (dense+bm25).")
    return recreated


def reset_sync_maps():
    global biz_map, wish_map, exp_map
    biz_map = {}
    wish_map = {}
    exp_map = {}


def prepare_search_indexes():
    """Ensure hybrid schema once per process; clear sync maps after recreate."""
    global _hybrid_collections_verified
    if _hybrid_collections_verified:
        return
    if ensure_collections():
        reset_sync_maps()
    _hybrid_collections_verified = True


# ---------------------------------------------------------
# BUSINESS SYNC
# ---------------------------------------------------------
def sync_businesses(biz_map):
    print("\n==============================")
    print("📦 SYNCING BUSINESSES")
    print("==============================")

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    business_columns = get_existing_columns(cur, "business")
    business_sync_ts_col = pick_sync_timestamp_column(
        business_columns,
        [
            # every_circle.business (actual schema — no generic updated_at)
            "business_updated_at",
            "updated_at",
            "business_joined_timestamp",
            "business_last_updated_at",
            "business_created_at",
        ],
    )

    business_select_fields = [
        "business_uid",
        "business_name",
        "business_short_bio",
        "business_tag_line",
        "business_city",
        "business_state",
        "business_country",
        "business_latitude",
        "business_longitude",
        "business_phone_number",
        "business_phone_number_is_public",
        "business_email_id",
        "business_email_id_is_public",
        "business_images_url",
        "business_images_is_public",
        "business_profile_img",
        "business_profile_img_is_public",
        "business_owner_fn",
        "business_owner_ln",
        "business_price_level",
        "business_google_rating",
        "business_reward_type",
        "business_reward_amount",
    ]
    if business_sync_ts_col:
        business_select_fields.append(business_sync_ts_col)

    cur.execute(
        f"""
        SELECT
            {", ".join(business_select_fields)}
        FROM business
        WHERE business_is_active = 1
        """
    )
    rows = cur.fetchall()

    cur.execute("SELECT bs_business_id, bs_service_name, bs_tags FROM business_services")
    service_rows = cur.fetchall()

    cur.execute("""
        SELECT bt.bt_business_id, t.tag_name
        FROM business_tags bt
        JOIN tags t ON t.tag_uid = bt.bt_tag_id
    """)
    business_tag_rows = cur.fetchall()
    conn.close()

    # Map business_uid → list of service tags and service names
    service_map = {}
    service_name_map = {}
    # Map business_uid -> custom business tags (business_tags -> tags)
    custom_tag_map = {}
    # Fingerprint tag text per business so bs_tags changes trigger reindex
    service_fingerprint = {}
    for s in service_rows:
        bid = s["bs_business_id"]
        if bid not in service_map:
            service_map[bid] = []
        if bid not in service_name_map:
            service_name_map[bid] = []

        service_name_raw = s.get("bs_service_name")
        if service_name_raw:
            service_name_map[bid].append(str(service_name_raw).strip().lower())
        tags_raw = s["bs_tags"]
        if tags_raw:
            service_map[bid].extend(
                [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
            )

    for row in business_tag_rows:
        bid = row["bt_business_id"]
        if bid not in custom_tag_map:
            custom_tag_map[bid] = []
        tag_name = row.get("tag_name")
        if tag_name:
            custom_tag_map[bid].append(str(tag_name).strip().lower())

    for bid, tags in service_map.items():
        names = service_name_map.get(bid, [])
        custom_tags = custom_tag_map.get(bid, [])
        service_fingerprint[bid] = "|".join(sorted(tags + names + custom_tags))

    current_state = {}
    for r in rows:
        uid = r["business_uid"]
        biz_updated = str(r.get(business_sync_ts_col) or "")
        tags_fp = service_fingerprint.get(uid, "")
        current_state[uid] = f"{biz_updated}::{tags_fp}"

    # INSERT or UPDATE operations
    for row in rows:
        uid = row["business_uid"]
        row["bs_tags"] = service_map.get(uid, [])
        row["bs_service_names"] = service_name_map.get(uid, [])
        row["custom_tags"] = custom_tag_map.get(uid, [])

        is_new = uid not in biz_map
        is_updated = (not is_new and biz_map[uid] != current_state[uid])

        if is_new:
            print(f"🆕 New business detected: {uid}")
        elif is_updated:
            print(f"🔄 Updated business detected: {uid}")

        if is_new or is_updated:
            upsert_business(row)
            success = verify_qdrant_insert("businesses", "business_uid", uid)
            print(("✔" if success else "❌") + f" {uid} — {row['business_name']}")

        biz_map[uid] = current_state[uid]

    # Handle deleted businesses
    for old_uid in list(biz_map.keys()):
        if old_uid not in current_state:
            print(f"🗑 Removing business: {old_uid}")
            qdrant.delete("businesses", points_selector=[make_uuid(old_uid)])
            biz_map.pop(old_uid, None)

    return biz_map


# ---------------------------------------------------------
# UPSERT BUSINESS
# ---------------------------------------------------------
def upsert_business(row):
    uid = row["business_uid"]

    # parse tagline tags
    tagline_tags = []
    if row.get("business_tag_line"):
        tagline_tags = [
            t.strip().lower()
            for t in row["business_tag_line"].split(",")
            if t.strip()
        ]

    # normalize service tags from business_services.bs_tags
    service_tags = []
    for t in row.get("bs_tags", []):
        if t is None:
            continue
        s = str(t).strip().lower()
        if s:
            service_tags.append(s)

    # normalize service names from business_services.bs_service_name
    service_names = []
    for t in row.get("bs_service_names", []):
        if t is None:
            continue
        s = str(t).strip().lower()
        if s:
            service_names.append(s)

    # normalize custom business tags (business_tags -> tags)
    custom_tags = []
    for t in row.get("custom_tags", []):
        if t is None:
            continue
        s = str(t).strip().lower()
        if s:
            custom_tags.append(s)

    # create searchable text (include tags so category/cuisine intent is captured)
    searchable_terms = [
        row.get("business_name") or "",
        row.get("business_short_bio") or "",
        row.get("business_tag_line") or "",
        " ".join(tagline_tags),
        " ".join(service_tags),
        " ".join(service_names),
        " ".join(custom_tags),
    ]
    text = " ".join([part for part in searchable_terms if part and str(part).strip()])

    # sanitize numeric values
    row["business_latitude"] = safe_float(row.get("business_latitude"))
    row["business_longitude"] = safe_float(row.get("business_longitude"))
    row["business_price_level"] = safe_int(row.get("business_price_level"))
    row["business_google_rating"] = safe_float(row.get("business_google_rating"))
    row["business_reward_amount"] = safe_float(row.get("business_reward_amount"))

    payload = {
        **row,
        "tags": tagline_tags,
        "bs_tags": service_tags,
        "bs_service_names": service_names,
        "custom_tags": custom_tags,
    }

    qdrant.upsert(
        collection_name="businesses",
        points=[
            PointStruct(
                id=make_uuid(uid),
                vector=build_point_vectors(text),
                payload=payload
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH BUSINESS
# ---------------------------------------------------------
@app.route("/search_business", methods=["GET"])
def search_business():
    global biz_map
    prepare_search_indexes()
    biz_map = sync_businesses(biz_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")

    # FILTER PARAMETERS (safe)
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))

    final_limit = get_limit(limit_param, 99999)
    candidate_limit = hybrid_candidate_limit(final_limit)

    if is_browse_query(query):
        filtered = fetch_browse_businesses(user_lat, user_lon, max_distance, min_rating, max_rating)
        return jsonify_categorized(filtered[:final_limit])

    vector = embed_text(query)
    results = qdrant_hybrid_search("businesses", query, vector, candidate_limit)

    business_uids = [r.payload.get("business_uid") for r in results if getattr(r, "payload", None)]
    additional_info = {}

    # fetch SQL details
    if business_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(business_uids))

        cur.execute(
            f"""
            SELECT *
            FROM business
            WHERE business_uid IN ({placeholders})
              AND business_is_active = 1
        """,
            business_uids,
        )

        rows = cur.fetchall()
        conn.close()

        for row in rows:
            # sanitize numeric fields here too
            row["business_latitude"] = safe_float(row.get("business_latitude"))
            row["business_longitude"] = safe_float(row.get("business_longitude"))
            row["business_google_rating"] = safe_float(row.get("business_google_rating"))
            row["business_reward_amount"] = safe_float(row.get("business_reward_amount"))
            row["business_price_level"] = safe_int(row.get("business_price_level"))

            additional_info[row["business_uid"]] = row

    boosted_candidates = merge_hybrid_hits(results, "business_uid", additional_info)
    annotate_hybrid_channels(
        boosted_candidates, query, "business_uid", "businesses", candidate_limit
    )

    # -----------------------------------------------------
    # APPLY FILTERS + ADD DISTANCE (SAFE)
    # -----------------------------------------------------
    filtered = []
    for merged in boosted_candidates:
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            merged.get("business_latitude"),
            merged.get("business_longitude"),
        )
        if not include:
            continue
        if dist is not None:
            merged["distance_miles"] = dist

        # Safely filter by rating
        rating = safe_float(merged.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        filtered.append(merged)

    filtered.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify_categorized(filtered[:final_limit])

# ---------------------------------------------------------
# WISHES SYNC
# ---------------------------------------------------------
def sync_wishes(wish_map):
    print("\n==============================")
    print("💫 SYNCING WISHES")
    print("==============================")

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    wish_columns = get_existing_columns(cur, "profile_wish")
    wish_sync_ts_col = pick_sync_timestamp_column(
        wish_columns,
        [
            # profile_wish (actual schema — no generic updated_at)
            "profile_wish_updated_at",
            "updated_at",
            "profile_wish_last_updated_at",
            "profile_wish_created_at",
        ],
    )
    wish_select_fields = [
        "profile_wish_uid",
        "profile_wish_title",
        "profile_wish_description",
    ]
    if wish_sync_ts_col:
        wish_select_fields.append(wish_sync_ts_col)

    cur.execute(
        f"""
        SELECT {", ".join(wish_select_fields)}
        FROM profile_wish
        WHERE profile_wish_is_public = 1
          AND COALESCE(profile_wish_moderated, 0) = 0
        """
    )
    rows = cur.fetchall()
    conn.close()

    current_state = {
        r["profile_wish_uid"]: str(r.get(wish_sync_ts_col) or "")
        for r in rows
    }

    for row in rows:
        uid = row["profile_wish_uid"]
        is_new = uid not in wish_map
        is_updated = (not is_new and wish_map[uid] != current_state[uid])

        if is_new:
            print(f"🆕 New wish detected: {uid}")
        elif is_updated:
            print(f"🔄 Updated wish detected: {uid}")

        if is_new or is_updated:
            upsert_wish(row)
            success = verify_qdrant_insert("wishes", "profile_wish_uid", uid)
            print(("✔" if success else "❌") + f" {uid} — {row['profile_wish_title']}")

        wish_map[uid] = current_state[uid]

    # Remove deleted wishes
    for old_uid in list(wish_map.keys()):
        if old_uid not in current_state:
            print(f"🗑 Removing wish: {old_uid}")
            qdrant.delete("wishes", points_selector=[make_uuid(old_uid)])
            wish_map.pop(old_uid, None)

    return wish_map


# ---------------------------------------------------------
# UPSERT WISH (no numeric conversions required)
# ---------------------------------------------------------
def upsert_wish(row):
    uid = row["profile_wish_uid"]
    text = f"{row['profile_wish_title']} {row.get('profile_wish_description') or ''}"

    qdrant.upsert(
        collection_name="wishes",
        points=[
            PointStruct(
                id=make_uuid(uid),
                vector=build_point_vectors(text),
                payload=row
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH WISHES
# ---------------------------------------------------------
@app.route("/search_wishes", methods=["GET"])
def search_wishes():
    global wish_map
    prepare_search_indexes()
    wish_map = sync_wishes(wish_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))

    final_limit = get_limit(limit_param, 99999)
    candidate_limit = hybrid_candidate_limit(final_limit)

    if is_browse_query(query):
        response = fetch_browse_wishes(user_lat, user_lon, max_distance)
        return jsonify_categorized(response[:final_limit])

    vector = embed_text(query)
    results = qdrant_hybrid_search("wishes", query, vector, candidate_limit)

    wish_uids = [r.payload.get("profile_wish_uid") for r in results if getattr(r, "payload", None)]

    additional_info = {}

    if wish_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(wish_uids))

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
            wish_uids,
        )

        rows = cur.fetchall()
        conn.close()

        for row in rows:
            # sanitize location numeric fields
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
            row["profile_wish_latitude"] = safe_float(row.get("profile_wish_latitude"))
            row["profile_wish_longitude"] = safe_float(row.get("profile_wish_longitude"))

            additional_info[row["profile_wish_uid"]] = row

    boosted_candidates = merge_hybrid_hits(results, "profile_wish_uid", additional_info)
    annotate_hybrid_channels(
        boosted_candidates, query, "profile_wish_uid", "wishes", candidate_limit
    )

    response = []
    for obj in boosted_candidates:
        wish_lat, wish_lon = wish_effective_coords(obj)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            wish_lat,
            wish_lon,
        )
        if not include:
            continue
        if dist is not None:
            obj["distance_miles"] = dist

        response.append(obj)

    response.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify_categorized(response[:final_limit])


# ---------------------------------------------------------
# EXPERTISE SYNC
# ---------------------------------------------------------
def sync_expertise(exp_map):
    print("\n==============================")
    print("🎓 SYNCING EXPERTISE")
    print("==============================")

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)

    exp_columns = get_existing_columns(cur, "profile_expertise")
    exp_sync_ts_col = pick_sync_timestamp_column(
        exp_columns,
        [
            # profile_expertise (actual schema — no generic updated_at)
            "profile_expertise_updated_at",
            "updated_at",
            "profile_expertise_last_updated_at",
            "profile_expertise_created_at",
        ],
    )
    exp_select_fields = [
        "profile_expertise_uid",
        "profile_expertise_title",
        "profile_expertise_description",
    ]
    if exp_sync_ts_col:
        exp_select_fields.append(exp_sync_ts_col)

    cur.execute(
        f"""
        SELECT {", ".join(exp_select_fields)}
        FROM profile_expertise
        WHERE profile_expertise_is_public = 1
          AND COALESCE(profile_expertise_moderated, 0) = 0
        """
    )
    rows = cur.fetchall()
    conn.close()

    current_state = {
        r["profile_expertise_uid"]: str(r.get(exp_sync_ts_col) or "")
        for r in rows
    }

    for row in rows:
        uid = row["profile_expertise_uid"]
        is_new = uid not in exp_map
        is_updated = (not is_new and exp_map[uid] != current_state[uid])

        if is_new:
            print(f"🆕 New expertise detected: {uid}")
        elif is_updated:
            print(f"🔄 Updated expertise detected: {uid}")

        if is_new or is_updated:
            upsert_expertise(row)
            success = verify_qdrant_insert("expertise", "profile_expertise_uid", uid)
            print(("✔" if success else "❌") + f" {uid} — {row['profile_expertise_title']}")

        exp_map[uid] = current_state[uid]

    # remove deleted
    for old_uid in list(exp_map.keys()):
        if old_uid not in current_state:
            print(f"🗑 Removing expertise: {old_uid}")
            qdrant.delete("expertise", points_selector=[make_uuid(old_uid)])
            exp_map.pop(old_uid, None)

    return exp_map


# ---------------------------------------------------------
# UPSERT EXPERTISE
# ---------------------------------------------------------
def upsert_expertise(row):
    uid = row["profile_expertise_uid"]
    text = f"{row['profile_expertise_title']} {row.get('profile_expertise_description') or ''}"

    qdrant.upsert(
        collection_name="expertise",
        points=[
            PointStruct(
                id=make_uuid(uid),
                vector=build_point_vectors(text),
                payload=row
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH EXPERTISE
# ---------------------------------------------------------
@app.route("/search_expertise", methods=["GET"])
def search_expertise():
    global exp_map
    prepare_search_indexes()
    exp_map = sync_expertise(exp_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))

    final_limit = get_limit(limit_param, 99999)
    candidate_limit = hybrid_candidate_limit(final_limit)

    if is_browse_query(query):
        response = fetch_browse_expertise(user_lat, user_lon, max_distance)
        return jsonify_categorized(response[:final_limit])

    vector = embed_text(query)
    results = qdrant_hybrid_search("expertise", query, vector, candidate_limit)

    exp_uids = [r.payload.get("profile_expertise_uid") for r in results if getattr(r, "payload", None)]
    additional_info = {}

    if exp_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(exp_uids))

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
            exp_uids,
        )

        rows = cur.fetchall()
        conn.close()

        for row in rows:
            # sanitize numeric fields
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
            row["profile_expertise_latitude"] = safe_float(row.get("profile_expertise_latitude"))
            row["profile_expertise_longitude"] = safe_float(row.get("profile_expertise_longitude"))

            additional_info[row["profile_expertise_uid"]] = row

    boosted_candidates = merge_hybrid_hits(results, "profile_expertise_uid", additional_info)
    annotate_hybrid_channels(
        boosted_candidates, query, "profile_expertise_uid", "expertise", candidate_limit
    )

    response = []
    for obj in boosted_candidates:
        exp_lat, exp_lon = expertise_effective_coords(obj)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            exp_lat,
            exp_lon,
        )
        if not include:
            continue
        if dist is not None:
            obj["distance_miles"] = dist

        response.append(obj)

    response.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify_categorized(response[:final_limit])


# ---------------------------------------------------------
# SEARCH GLOBAL (business + expertise + seeking)
# ---------------------------------------------------------
@app.route("/search_global", methods=["GET"])
def search_global():
    global biz_map, exp_map, wish_map
    prepare_search_indexes()
    biz_map = sync_businesses(biz_map)
    exp_map = sync_expertise(exp_map)
    wish_map = sync_wishes(wish_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))

    final_limit = get_limit(limit_param, 99999)
    candidate_limit = hybrid_candidate_limit(final_limit)

    if is_browse_query(query):
        business_results = [{**row, "itemType": "businesses"} for row in fetch_browse_businesses(user_lat, user_lon, max_distance, min_rating, max_rating)]
        expertise_results = [{**row, "itemType": "expertise"} for row in fetch_browse_expertise(user_lat, user_lon, max_distance)]
        seeking_results = [{**row, "itemType": "seeking"} for row in fetch_browse_wishes(user_lat, user_lon, max_distance)]
        combined = business_results + expertise_results + seeking_results
        return jsonify_categorized(combined[:final_limit])

    vector = embed_text(query)

    # --- businesses ---
    biz_hits = qdrant_hybrid_search("businesses", query, vector, candidate_limit)
    biz_uids = [r.payload.get("business_uid") for r in biz_hits if getattr(r, "payload", None)]
    biz_rows = {}
    if biz_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(biz_uids))
        cur.execute(
            f"""
            SELECT *
            FROM business
            WHERE business_uid IN ({placeholders})
              AND business_is_active = 1
        """,
            biz_uids,
        )
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            row["business_google_rating"] = safe_float(row.get("business_google_rating"))
            row["business_reward_amount"] = safe_float(row.get("business_reward_amount"))
            row["business_price_level"] = safe_int(row.get("business_price_level"))
            row["business_latitude"] = safe_float(row.get("business_latitude"))
            row["business_longitude"] = safe_float(row.get("business_longitude"))
            biz_rows[row["business_uid"]] = row

    biz_merged = merge_hybrid_hits(biz_hits, "business_uid", biz_rows)
    annotate_hybrid_channels(biz_merged, query, "business_uid", "businesses", candidate_limit)

    business_results = []
    for merged in biz_merged:
        merged["itemType"] = "businesses"

        rating = safe_float(merged.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            merged.get("business_latitude"),
            merged.get("business_longitude"),
        )
        if not include:
            continue
        if dist is not None:
            merged["distance_miles"] = dist

        business_results.append(merged)

    business_results.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)

    # --- expertise ---
    exp_hits = qdrant_hybrid_search("expertise", query, vector, candidate_limit)
    exp_uids = [r.payload.get("profile_expertise_uid") for r in exp_hits if getattr(r, "payload", None)]
    exp_rows = {}
    if exp_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(exp_uids))
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
            exp_uids,
        )
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
            row["profile_expertise_latitude"] = safe_float(row.get("profile_expertise_latitude"))
            row["profile_expertise_longitude"] = safe_float(row.get("profile_expertise_longitude"))
            exp_rows[row["profile_expertise_uid"]] = row

    exp_merged = merge_hybrid_hits(exp_hits, "profile_expertise_uid", exp_rows)
    annotate_hybrid_channels(
        exp_merged, query, "profile_expertise_uid", "expertise", candidate_limit
    )

    expertise_results = []
    for merged in exp_merged:
        merged["itemType"] = "expertise"
        exp_lat, exp_lon = expertise_effective_coords(merged)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            exp_lat,
            exp_lon,
        )
        if not include:
            continue
        if dist is not None:
            merged["distance_miles"] = dist
        expertise_results.append(merged)

    # --- seeking (wishes) ---
    wish_hits = qdrant_hybrid_search("wishes", query, vector, candidate_limit)
    wish_uids = [r.payload.get("profile_wish_uid") for r in wish_hits if getattr(r, "payload", None)]
    wish_rows = {}
    if wish_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(wish_uids))
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
            wish_uids,
        )
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
            row["profile_wish_latitude"] = safe_float(row.get("profile_wish_latitude"))
            row["profile_wish_longitude"] = safe_float(row.get("profile_wish_longitude"))
            wish_rows[row["profile_wish_uid"]] = row

    wish_merged = merge_hybrid_hits(wish_hits, "profile_wish_uid", wish_rows)
    annotate_hybrid_channels(wish_merged, query, "profile_wish_uid", "wishes", candidate_limit)

    seeking_results = []
    for merged in wish_merged:
        merged["itemType"] = "seeking"
        wish_lat, wish_lon = wish_effective_coords(merged)
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            wish_lat,
            wish_lon,
        )
        if not include:
            continue
        if dist is not None:
            merged["distance_miles"] = dist
        seeking_results.append(merged)

    for item in business_results:
        base = safe_float(item.get("score")) or 0.0
        item["global_score"] = base * GLOBAL_BUSINESS_WEIGHT
    for item in expertise_results:
        base = safe_float(item.get("score")) or 0.0
        item["global_score"] = base * GLOBAL_EXPERTISE_WEIGHT
    for item in seeking_results:
        base = safe_float(item.get("score")) or 0.0
        item["global_score"] = base * GLOBAL_SEEKING_WEIGHT

    combined = business_results + expertise_results + seeking_results
    combined.sort(key=lambda x: safe_float(x.get("global_score")) or 0.0, reverse=True)
    return jsonify_categorized(combined[:final_limit])


# ---------------------------------------------------------
# SEARCH SUGGEST (autocomplete) — swappable data sources
# ---------------------------------------------------------
SEARCH_SUGGEST_MIN_LEN = int(_env_float("SEARCH_SUGGEST_MIN_LEN", 2))
SEARCH_SUGGEST_DEFAULT_LIMIT = int(_env_float("SEARCH_SUGGEST_DEFAULT_LIMIT", 8))
SEARCH_SUGGEST_MAX_LIMIT = int(_env_float("SEARCH_SUGGEST_MAX_LIMIT", 20))
SEARCH_SUGGEST_SOURCE = (os.getenv("SEARCH_SUGGEST_SOURCE") or "tags").strip().lower()


def fetch_tag_suggestions(query, limit):
    """
    Tag-backed autocomplete. Replace or extend via SEARCH_SUGGEST_SOURCES.
    """
    q = (query or "").strip().lower()
    if len(q) < SEARCH_SUGGEST_MIN_LEN:
        return []

    prefix_pattern = f"{q}%"
    contains_pattern = f"%{q}%"

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        """
        SELECT
            t.tag_name,
            COUNT(DISTINCT bt.bt_business_id) AS business_count
        FROM tags t
        LEFT JOIN business_tags bt ON bt.bt_tag_id = t.tag_uid
        WHERE LOWER(t.tag_name) LIKE %s
        GROUP BY t.tag_uid, t.tag_name
        ORDER BY
            CASE WHEN LOWER(t.tag_name) LIKE %s THEN 0 ELSE 1 END,
            business_count DESC,
            t.tag_name ASC
        LIMIT %s
        """,
        (contains_pattern, prefix_pattern, limit),
    )
    rows = cur.fetchall()
    conn.close()

    suggestions = []
    for row in rows:
        tag_name = (row.get("tag_name") or "").strip()
        if not tag_name:
            continue
        suggestions.append(
            {
                "text": tag_name,
                "source": "tags",
                "count": int(row.get("business_count") or 0),
            }
        )
    return suggestions


SEARCH_SUGGEST_SOURCES = {
    "tags": fetch_tag_suggestions,
}


def get_search_suggestions(query, limit, source=None):
    resolved_source = (source or SEARCH_SUGGEST_SOURCE or "tags").strip().lower()
    provider = SEARCH_SUGGEST_SOURCES.get(resolved_source)
    if not provider:
        return [], resolved_source
    return provider(query, limit), resolved_source


@app.route("/search_suggest", methods=["GET"])
def search_suggest():
    query = (request.args.get("q") or "").strip()
    source = (request.args.get("source") or "").strip().lower() or None

    try:
        limit = int(request.args.get("limit", SEARCH_SUGGEST_DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = SEARCH_SUGGEST_DEFAULT_LIMIT
    limit = max(1, min(limit, SEARCH_SUGGEST_MAX_LIMIT))

    suggestions, resolved_source = get_search_suggestions(query, limit, source=source)
    return jsonify(
        {
            "query": query,
            "source": resolved_source,
            "suggestions": suggestions,
        }
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    prepare_search_indexes()
    reset_sync_maps()

    app.run(host="0.0.0.0", port=5001)
