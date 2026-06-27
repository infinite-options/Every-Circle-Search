import os
import pymysql
import uuid
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

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


def qdrant_vector_search(collection_name, query_vector, limit):
    """
    Compatibility wrapper across qdrant-client versions.
    Older versions expose `search`, newer versions expose `query_points`.
    Returns a list of scored points in both cases.
    """
    if hasattr(qdrant, "search"):
        return qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )

    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        points = getattr(response, "points", None)
        if points is not None:
            return points
        if isinstance(response, dict):
            return response.get("points", [])
        return []

    raise AttributeError("Qdrant client does not support vector search methods (search/query_points)")


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except:
        return default


MIN_SIMILARITY_SCORE = _env_float("MIN_SIMILARITY_SCORE", 0.25)
RELATIVE_SCORE_FACTOR = _env_float("RELATIVE_SCORE_FACTOR", 0.60)
LEXICAL_MIN_SCORE = _env_float("LEXICAL_MIN_SCORE", 0.30)
GLOBAL_BUSINESS_WEIGHT = _env_float("GLOBAL_BUSINESS_WEIGHT", 1.15)
GLOBAL_EXPERTISE_WEIGHT = _env_float("GLOBAL_EXPERTISE_WEIGHT", 0.85)
SEARCH_DEFAULT_LIMIT = int(_env_float("SEARCH_DEFAULT_LIMIT", 120))
# Soft proximity boost when user home coords are sent but no distance filter is active.
PROXIMITY_BOOST_MILES = _env_float("PROXIMITY_BOOST_MILES", 5.0)
PROXIMITY_BOOST_FACTOR = _env_float("PROXIMITY_BOOST_FACTOR", 1.12)

# Business hybrid re-score: "rrf" = reciprocal rank fusion (semantic vs fuzzy-lexical); "legacy" = sem + token boost
RESCORE_MODE = (os.getenv("RESCORE_MODE") or "rrf").strip().lower()
RRF_K = _env_float("RRF_K", 60.0)


def filter_relevant_hits(hits):
    """
    Trim low-relevance tail results to reduce irrelevant matches.
    Keeps hits that satisfy BOTH:
      1) absolute minimum score
      2) relative score vs top hit in the same result set
    """
    if not hits:
        return []

    absolute_min = MIN_SIMILARITY_SCORE
    relative_factor = RELATIVE_SCORE_FACTOR

    top_score = max([(safe_float(getattr(h, "score", None)) or 0.0) for h in hits] + [0.0])
    relative_min = top_score * relative_factor

    filtered = []
    for hit in hits:
        score = safe_float(getattr(hit, "score", None)) or 0.0
        if score >= absolute_min and score >= relative_min:
            filtered.append(hit)

    # Never return empty just because thresholds were too strict; keep best hit.
    if not filtered and hits:
        return hits[:1]

    return filtered


def normalize_tokens(text):
    if text is None:
        return []
    s = str(text).strip().lower()
    if not s:
        return []
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return [t for t in cleaned.split() if t]


def business_lexical_details(query, payload):
    """
    Return lexical contribution breakdown for business relevance.
    """
    q_tokens = normalize_tokens(query)
    if not q_tokens:
        return {
            "token_tag": 0.0,
            "token_name": 0.0,
            "token_tagline": 0.0,
            "token_bio": 0.0,
            "phrase_name": 0.0,
            "phrase_tag": 0.0,
            "total_lexical_boost": 0.0,
        }

    name_tokens = normalize_tokens(payload.get("business_name"))
    bio_tokens = normalize_tokens(payload.get("business_short_bio"))
    tag_line_tokens = normalize_tokens(payload.get("business_tag_line"))

    tag_tokens = []
    for t in payload.get("tags", []) or []:
        tag_tokens.extend(normalize_tokens(t))
    for t in payload.get("bs_tags", []) or []:
        tag_tokens.extend(normalize_tokens(t))
    for t in payload.get("bs_service_names", []) or []:
        tag_tokens.extend(normalize_tokens(t))
    for t in payload.get("custom_tags", []) or []:
        tag_tokens.extend(normalize_tokens(t))

    name_set = set(name_tokens)
    bio_set = set(bio_tokens)
    tag_line_set = set(tag_line_tokens)
    tag_set = set(tag_tokens)

    token_tag = 0.0
    token_name = 0.0
    token_tagline = 0.0
    token_bio = 0.0
    for q in q_tokens:
        if q in tag_set:
            token_tag += 0.22
        if q in name_set:
            token_name += 0.16
        if q in tag_line_set:
            token_tagline += 0.12
        if q in bio_set:
            token_bio += 0.06

    # Phrase-level tiny bonus when raw query appears in key text
    phrase_name = 0.0
    phrase_tag = 0.0
    q_str = " ".join(q_tokens)
    if q_str:
        name_str = " ".join(name_tokens)
        tag_str = " ".join(tag_tokens)
        if q_str and q_str in name_str:
            phrase_name += 0.08
        if q_str and q_str in tag_str:
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


def business_lexical_boost(query, payload):
    details = business_lexical_details(query, payload)
    return details["total_lexical_boost"]


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


def fuzzy_norm(query, text):
    if not query or not text:
        return 0.0
    return fuzz.token_set_ratio(str(query), str(text)) / 100.0


def business_lexical_score(query, row):
    name = row.get("business_name") or ""
    tagline = row.get("business_tag_line") or ""
    bio = row.get("business_short_bio") or ""
    # SQL lexical path uses all_* keys; Qdrant payload uses bs_* / custom_tags.
    service_names = csv_to_tokens(row.get("all_service_names") or row.get("bs_service_names"))
    service_tags = csv_to_tokens(row.get("all_service_tags") or row.get("bs_tags"))
    custom_tags = csv_to_tokens(row.get("all_custom_tags") or row.get("custom_tags"))

    score = 0.0
    score += 0.30 * fuzzy_norm(query, name)
    score += 0.10 * fuzzy_norm(query, tagline)
    score += 0.08 * fuzzy_norm(query, bio)
    score += 0.22 * fuzzy_norm(query, " ".join(service_names))
    score += 0.18 * fuzzy_norm(query, " ".join(service_tags))
    score += 0.22 * fuzzy_norm(query, " ".join(custom_tags))

    return min(1.0, score)


def apply_hybrid_rescore(query, merged_candidates, lexical_score_fn, uid_field, lexical_details_fn=None):
    """
    Produce final `score` and `score_breakdown` via semantic + fuzzy-lexical hybrid re-score.

    - rrf (default): reciprocal rank fusion of Qdrant semantic order and fuzzy lexical order.
    - legacy: semantic cosine plus lexical boost (token details for business, scaled fuzzy for others).
    """
    if not merged_candidates:
        return merged_candidates

    if RESCORE_MODE == "legacy":
        for merged in merged_candidates:
            sem_score = safe_float(merged.pop("_sem_score", None))
            if sem_score is None:
                sem_score = safe_float(merged.get("score")) or 0.0
            lex_score = lexical_score_fn(query, merged)
            if lexical_details_fn:
                lexical_details = lexical_details_fn(query, merged)
                final_score = sem_score + lexical_details["total_lexical_boost"]
            else:
                lexical_details = {}
                final_score = sem_score + lex_score * 0.25
            merged["score"] = final_score
            breakdown = {
                "rescore_mode": "legacy",
                "semantic_score": sem_score,
                "lexical_fuzzy_score": lex_score,
                "final_score": final_score,
            }
            if lexical_details:
                breakdown.update(lexical_details)
            merged["score_breakdown"] = breakdown
        return merged_candidates

    n = len(merged_candidates)
    sem_scores = []
    lex_scores = []
    for merged in merged_candidates:
        sem = safe_float(merged.pop("_sem_score", None))
        if sem is None:
            sem = safe_float(merged.get("score")) or 0.0
        sem_scores.append(sem)
        lex_scores.append(lexical_score_fn(query, merged))

    uid_key = lambda i: str(merged_candidates[i].get(uid_field) or "")
    sem_order = sorted(
        range(n),
        key=lambda i: (-(sem_scores[i] or 0.0), -(lex_scores[i] or 0.0), uid_key(i)),
    )
    lex_order = sorted(
        range(n),
        key=lambda i: (-(lex_scores[i] or 0.0), -(sem_scores[i] or 0.0), uid_key(i)),
    )

    rank_sem = [0] * n
    rank_lex = [0] * n
    for pos, i in enumerate(sem_order):
        rank_sem[i] = pos + 1
    for pos, i in enumerate(lex_order):
        rank_lex[i] = pos + 1

    k = RRF_K
    max_raw = (2.0 / (k + 1.0)) if k > 0 else 0.0

    for i, merged in enumerate(merged_candidates):
        raw_rrf = (1.0 / (k + rank_sem[i])) + (1.0 / (k + rank_lex[i]))
        norm_rrf = min(1.0, (raw_rrf / max_raw) if max_raw > 0 else 0.0)
        extra = lexical_details_fn(query, merged) if lexical_details_fn else {}
        merged["score"] = norm_rrf
        merged["score_breakdown"] = {
            "rescore_mode": "rrf",
            "semantic_score": sem_scores[i],
            "lexical_fuzzy_score": lex_scores[i],
            **extra,
            "rrf_k": k,
            "rrf_rank_semantic": rank_sem[i],
            "rrf_rank_lexical": rank_lex[i],
            "rrf_raw": raw_rrf,
            "final_score": norm_rrf,
        }
    return merged_candidates


def apply_business_rescore(query, merged_candidates):
    return apply_hybrid_rescore(
        query,
        merged_candidates,
        business_lexical_score,
        "business_uid",
        business_lexical_details,
    )


def apply_expertise_rescore(query, merged_candidates):
    return apply_hybrid_rescore(
        query,
        merged_candidates,
        expertise_lexical_score,
        "profile_expertise_uid",
    )


def apply_wish_rescore(query, merged_candidates):
    return apply_hybrid_rescore(
        query,
        merged_candidates,
        wish_lexical_score,
        "profile_wish_uid",
    )


def filter_rescored_candidates(merged_candidates):
    class _Hit:
        def __init__(self, payload):
            self.payload = payload
            self.score = payload.get("score", 0.0)

    return [h.payload for h in filter_relevant_hits([_Hit(m) for m in merged_candidates])]


def expertise_lexical_score(query, row):
    title = row.get("profile_expertise_title") or ""
    desc = row.get("profile_expertise_description") or ""
    details = row.get("profile_expertise_details") or ""
    return min(1.0, 0.55 * fuzzy_norm(query, title) + 0.30 * fuzzy_norm(query, desc) + 0.15 * fuzzy_norm(query, details))


def wish_lexical_score(query, row):
    title = row.get("profile_wish_title") or ""
    desc = row.get("profile_wish_description") or ""
    return min(1.0, 0.60 * fuzzy_norm(query, title) + 0.40 * fuzzy_norm(query, desc))

# ---------------------------------------------------------
# SAFE CONVERSION HELPERS (bulletproof)
# ---------------------------------------------------------
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


def apply_home_proximity_boost(row, user_lat, user_lon, max_distance, lat_key, lon_key):
    """
    When home coords are provided without an explicit distance filter, boost rows
    within PROXIMITY_BOOST_MILES and flag them for the client UI.
    """
    row["location_boosted"] = False
    if user_lat is None or user_lon is None or max_distance is not None:
        return

    target_lat = row.get(lat_key)
    target_lon = row.get(lon_key)
    if not _coords_usable(target_lat, target_lon):
        return

    dist = haversine_miles(user_lat, user_lon, target_lat, target_lon)
    if dist is not None:
        row["distance_miles"] = dist

    if dist is None or dist > PROXIMITY_BOOST_MILES:
        return

    base = safe_float(row.get("score")) or 0.0
    boosted = base * PROXIMITY_BOOST_FACTOR
    row["score"] = boosted
    row["location_boosted"] = True

    breakdown = row.get("score_breakdown")
    if isinstance(breakdown, dict):
        breakdown["proximity_boost"] = True
        breakdown["proximity_boost_miles"] = dist
        breakdown["proximity_boost_factor"] = PROXIMITY_BOOST_FACTOR
        breakdown["final_score"] = boosted


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
               profile_personal_tag_line, profile_personal_tag_line_is_public
        FROM profile_expertise
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_expertise_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_expertise.profile_expertise_is_public = 1
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
               profile_personal_tag_line, profile_personal_tag_line_is_public
        FROM profile_wish
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_wish_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_wish.profile_wish_is_public = 1
        """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        row["score"] = 1.0
        row["score_breakdown"] = {"browse_mode": True, "final_score": 1.0}

        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            row.get("profile_personal_latitude"),
            row.get("profile_personal_longitude"),
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
# ENSURE COLLECTIONS
# ---------------------------------------------------------
def ensure_collections():
    for col in ["businesses", "wishes", "expertise"]:
        if not qdrant.collection_exists(col):
            print(f"🆕 Creating Qdrant collection '{col}'...")
            qdrant.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        print(f"✅ Collection '{col}' ready.")


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
                vector=embed_text(text),
                payload=payload
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH BUSINESS (FULLY FIXED)
# ---------------------------------------------------------
def search_business_lexical():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))
    final_limit = get_limit(limit_param, 99999)

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
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
    """)
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["business_latitude"] = safe_float(row.get("business_latitude"))
        row["business_longitude"] = safe_float(row.get("business_longitude"))
        row["business_google_rating"] = safe_float(row.get("business_google_rating"))
        row["score"] = business_lexical_score(query, row)
        row["bs_service_names"] = csv_to_tokens(row.get("all_service_names"))
        row["bs_tags"] = csv_to_tokens(row.get("all_service_tags"))
        row["custom_tags"] = csv_to_tokens(row.get("all_custom_tags"))

        if row["score"] < LEXICAL_MIN_SCORE:
            continue

        if user_lat is not None and user_lon is not None:
            dist = haversine_miles(user_lat, user_lon, row.get("business_latitude"), row.get("business_longitude"))
            row["distance_miles"] = dist
            if max_distance is not None and dist is not None and dist > max_distance:
                continue

        rating = safe_float(row.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        ranked.append(row)

    ranked.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(ranked[:final_limit])


@app.route("/search_business", methods=["GET"])
def search_business():
    global biz_map
    biz_map = sync_businesses(biz_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")

    # FILTER PARAMETERS (safe)
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    if is_browse_query(query):
        filtered = fetch_browse_businesses(user_lat, user_lon, max_distance, min_rating, max_rating)
        for merged in filtered:
            apply_home_proximity_boost(
                merged,
                user_lat,
                user_lon,
                max_distance,
                "business_latitude",
                "business_longitude",
            )
        return jsonify(filtered[:final_limit])

    vector = embed_text(query)

    # search qdrant
    results = qdrant_vector_search("businesses", query_vector=vector, limit=max_results)

    business_uids = [r.payload.get("business_uid") for r in results]
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

    # Build merged candidates; re-score (RRF or legacy) before relevance filtering.
    merged_candidates = []
    for r in results:
        uid = r.payload.get("business_uid")
        if business_uids and uid not in additional_info:
            continue
        merged = {**r.payload, "_sem_score": safe_float(r.score) or 0.0}
        if uid in additional_info:
            merged.update(additional_info[uid])
            merged["_sem_score"] = safe_float(r.score) or 0.0
        merged_candidates.append(merged)

    apply_business_rescore(query, merged_candidates)

    boosted_candidates = filter_rescored_candidates(merged_candidates)

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

        apply_home_proximity_boost(
            merged,
            user_lat,
            user_lon,
            max_distance,
            "business_latitude",
            "business_longitude",
        )

        # Safely filter by rating
        rating = safe_float(merged.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        filtered.append(merged)

    filtered.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(filtered[:final_limit])

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
                vector=embed_text(text),
                payload=row
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH WISHES (safe, but no numeric conversions needed)
# ---------------------------------------------------------
def search_wishes_lexical():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    final_limit = get_limit(limit_param, 99999)

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT profile_wish.*,
               user_email_id,
               profile_personal_first_name, profile_personal_last_name,
               profile_personal_email_is_public, profile_personal_phone_number,
               profile_personal_phone_number_is_public,
               profile_personal_city, profile_personal_state, profile_personal_country,
               profile_personal_location_is_public,
               profile_personal_latitude, profile_personal_longitude,
               profile_personal_image, profile_personal_image_is_public,
               profile_personal_tag_line, profile_personal_tag_line_is_public
        FROM profile_wish
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_wish_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_wish.profile_wish_is_public = 1
    """)
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        row["score"] = wish_lexical_score(query, row)
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        ranked.append(row)

    ranked.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(ranked[:final_limit])


@app.route("/search_wishes", methods=["GET"])
def search_wishes():
    global wish_map
    wish_map = sync_wishes(wish_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    if is_browse_query(query):
        response = fetch_browse_wishes(user_lat, user_lon, max_distance)
        return jsonify(response[:final_limit])

    vector = embed_text(query)

    results = qdrant_vector_search("wishes", query_vector=vector, limit=max_results)

    wish_uids = [r.payload.get("profile_wish_uid") for r in results]

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
                   profile_personal_tag_line, profile_personal_tag_line_is_public
            FROM profile_wish
            LEFT JOIN every_circle.profile_personal
                ON profile_personal_uid = profile_wish_profile_personal_id
            LEFT JOIN every_circle.users
                ON user_uid = profile_personal_user_id
            WHERE profile_wish_uid IN ({placeholders})
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

            additional_info[row["profile_wish_uid"]] = row

    merged_candidates = []
    for r in results:
        uid = r.payload.get("profile_wish_uid")
        if wish_uids and uid not in additional_info:
            continue
        merged = {**r.payload, "_sem_score": safe_float(r.score) or 0.0}
        if uid in additional_info:
            merged.update(additional_info[uid])
            merged["_sem_score"] = safe_float(r.score) or 0.0
        merged_candidates.append(merged)

    apply_wish_rescore(query, merged_candidates)
    boosted_candidates = filter_rescored_candidates(merged_candidates)

    response = []
    for obj in boosted_candidates:
        include, dist = distance_filter_passes(
            user_lat,
            user_lon,
            max_distance,
            obj.get("profile_personal_latitude"),
            obj.get("profile_personal_longitude"),
        )
        if not include:
            continue
        if dist is not None:
            obj["distance_miles"] = dist

        response.append(obj)

    response.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(response[:final_limit])


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
                vector=embed_text(text),
                payload=row
            )
        ]
    )


# ---------------------------------------------------------
# SEARCH EXPERTISE
# ---------------------------------------------------------
def search_expertise_lexical():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    final_limit = get_limit(limit_param, 99999)

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT profile_expertise.*,
               user_email_id,
               profile_personal_first_name, profile_personal_last_name,
               profile_personal_email_is_public, profile_personal_phone_number,
               profile_personal_phone_number_is_public,
               profile_personal_city, profile_personal_state, profile_personal_country,
               profile_personal_location_is_public,
               profile_personal_latitude, profile_personal_longitude,
               profile_personal_image, profile_personal_image_is_public,
               profile_personal_tag_line, profile_personal_tag_line_is_public
        FROM profile_expertise
        LEFT JOIN every_circle.profile_personal
            ON profile_personal_uid = profile_expertise_profile_personal_id
        LEFT JOIN every_circle.users
            ON user_uid = profile_personal_user_id
        WHERE profile_expertise.profile_expertise_is_public = 1
    """)
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        row["score"] = expertise_lexical_score(query, row)
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        ranked.append(row)

    ranked.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(ranked[:final_limit])


@app.route("/search_expertise", methods=["GET"])
def search_expertise():
    global exp_map
    exp_map = sync_expertise(exp_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    if is_browse_query(query):
        response = fetch_browse_expertise(user_lat, user_lon, max_distance)
        return jsonify(response[:final_limit])

    vector = embed_text(query)

    results = qdrant_vector_search("expertise", query_vector=vector, limit=max_results)

    exp_uids = [r.payload.get("profile_expertise_uid") for r in results]
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
                   profile_personal_tag_line, profile_personal_tag_line_is_public
            FROM profile_expertise
            LEFT JOIN every_circle.profile_personal
                ON profile_personal_uid = profile_expertise_profile_personal_id
            LEFT JOIN every_circle.users
                ON user_uid = profile_personal_user_id
            WHERE profile_expertise_uid IN ({placeholders})
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

    merged_candidates = []
    for r in results:
        uid = r.payload.get("profile_expertise_uid")
        if exp_uids and uid not in additional_info:
            continue
        merged = {**r.payload, "_sem_score": safe_float(r.score) or 0.0}
        if uid in additional_info:
            merged.update(additional_info[uid])
            merged["_sem_score"] = safe_float(r.score) or 0.0
        merged_candidates.append(merged)

    apply_expertise_rescore(query, merged_candidates)
    boosted_candidates = filter_rescored_candidates(merged_candidates)

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
    return jsonify(response[:final_limit])


# ---------------------------------------------------------
# SEARCH GLOBAL (business + expertise only)
# ---------------------------------------------------------
@app.route("/search_global", methods=["GET"])
def search_global():
    global biz_map, exp_map
    biz_map = sync_businesses(biz_map)
    exp_map = sync_expertise(exp_map)

    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    if is_browse_query(query):
        business_results = [{**row, "itemType": "businesses"} for row in fetch_browse_businesses(user_lat, user_lon, max_distance, min_rating, max_rating)]
        for item in business_results:
            apply_home_proximity_boost(
                item,
                user_lat,
                user_lon,
                max_distance,
                "business_latitude",
                "business_longitude",
            )
        expertise_results = [{**row, "itemType": "expertise"} for row in fetch_browse_expertise(user_lat, user_lon, max_distance)]
        seeking_results = [{**row, "itemType": "seeking"} for row in fetch_browse_wishes(user_lat, user_lon, max_distance)]
        combined = business_results + expertise_results + seeking_results
        return jsonify(combined[:final_limit])

    vector = embed_text(query)

    # --- businesses ---
    biz_hits = qdrant_vector_search("businesses", query_vector=vector, limit=max_results)
    biz_uids = [r.payload.get("business_uid") for r in biz_hits]
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

    merged_business_results = []
    for r in biz_hits:
        uid = r.payload.get("business_uid")
        if biz_uids and uid not in biz_rows:
            continue
        merged = {"itemType": "businesses", **r.payload, "_sem_score": safe_float(r.score) or 0.0}
        if uid in biz_rows:
            merged.update(biz_rows[uid])
            merged["_sem_score"] = safe_float(r.score) or 0.0

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

        merged_business_results.append(merged)

    apply_business_rescore(query, merged_business_results)

    business_results = filter_rescored_candidates(merged_business_results)
    for item in business_results:
        apply_home_proximity_boost(
            item,
            user_lat,
            user_lon,
            max_distance,
            "business_latitude",
            "business_longitude",
        )
    business_results.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)

    # --- expertise ---
    exp_hits = qdrant_vector_search("expertise", query_vector=vector, limit=max_results)
    exp_uids = [r.payload.get("profile_expertise_uid") for r in exp_hits]
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
                   profile_personal_tag_line, profile_personal_tag_line_is_public
            FROM profile_expertise
            LEFT JOIN every_circle.profile_personal
                ON profile_personal_uid = profile_expertise_profile_personal_id
            LEFT JOIN every_circle.users
                ON user_uid = profile_personal_user_id
            WHERE profile_expertise_uid IN ({placeholders})
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

    merged_expertise_results = []
    for r in exp_hits:
        uid = r.payload.get("profile_expertise_uid")
        if exp_uids and uid not in exp_rows:
            continue
        merged = {
            "itemType": "expertise",
            **r.payload,
            "_sem_score": safe_float(r.score) or 0.0,
        }
        if uid in exp_rows:
            merged.update(exp_rows[uid])
            merged["_sem_score"] = safe_float(r.score) or 0.0

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

        merged_expertise_results.append(merged)

    apply_expertise_rescore(query, merged_expertise_results)
    expertise_results = filter_rescored_candidates(merged_expertise_results)

    # Global ranking: prefer business intent by default, but keep expertise when truly relevant.
    for item in business_results:
        base = safe_float(item.get("score")) or 0.0
        item["global_score"] = base * GLOBAL_BUSINESS_WEIGHT
    for item in expertise_results:
        base = safe_float(item.get("score")) or 0.0
        item["global_score"] = base * GLOBAL_EXPERTISE_WEIGHT

    combined = business_results + expertise_results
    combined.sort(key=lambda x: safe_float(x.get("global_score")) or 0.0, reverse=True)
    return jsonify(combined[:final_limit])


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
    ensure_collections()

    global biz_map, wish_map, exp_map
    biz_map = {}
    wish_map = {}
    exp_map = {}

    app.run(host="0.0.0.0", port=5001)
