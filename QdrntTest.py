import os
import pymysql
import uuid
import math
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HOME"] = "/home/ec2-user/.cache/huggingface"

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

app = Flask(__name__)
embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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
        return 5

    value = str(param).strip().upper()

    if value == "ALL":
        return max_results

    if value.isdigit():
        return int(value)

    return 5


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


# ---------------------------------------------------------
# EMBEDDING + MYSQL HELPERS
# ---------------------------------------------------------
def embed_text(text: str):
    return embedder.encode(text).tolist()

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

    cur.execute("""
        SELECT 
            business_uid,
            business_name,
            business_short_bio,
            business_tag_line,
            business_city,
            business_state,
            business_country,
            business_latitude,
            business_longitude,
            business_phone_number,
            business_phone_number_is_public,
            business_email_id,
            business_email_id_is_public,
            business_images_url,
            business_images_is_public,
            business_owner_fn,
            business_owner_ln,
            business_price_level,
            business_google_rating,
            business_reward_type,
            business_reward_amount,
            updated_at
        FROM business
    """)
    rows = cur.fetchall()

    cur.execute("SELECT bs_business_id, bs_tags FROM business_services")
    service_rows = cur.fetchall()
    conn.close()

    # Map business_uid → list of service tags
    service_map = {}
    for s in service_rows:
        bid = s["bs_business_id"]
        if bid not in service_map:
            service_map[bid] = []

        tags_raw = s["bs_tags"]
        if tags_raw:
            service_map[bid].extend(
                [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
            )

    current_state = {r["business_uid"]: str(r["updated_at"]) for r in rows}

    # INSERT or UPDATE operations
    for row in rows:
        uid = row["business_uid"]
        row["bs_tags"] = service_map.get(uid, [])

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

    # create searchable text
    text = (
        f"{row['business_name']} "
        f"{row.get('business_short_bio') or ''} "
        f"{row.get('business_tag_line') or ''}"
    )

    # parse tagline tags
    tagline_tags = []
    if row.get("business_tag_line"):
        tagline_tags = [
            t.strip().lower()
            for t in row["business_tag_line"].split(",")
            if t.strip()
        ]

    # sanitize numeric values
    row["business_latitude"] = safe_float(row.get("business_latitude"))
    row["business_longitude"] = safe_float(row.get("business_longitude"))
    row["business_price_level"] = safe_int(row.get("business_price_level"))
    row["business_google_rating"] = safe_float(row.get("business_google_rating"))
    row["business_reward_amount"] = safe_float(row.get("business_reward_amount"))

    payload = {
        **row,
        "tags": tagline_tags,
        "bs_tags": row.get("bs_tags", [])
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
@app.route("/search_business", methods=["GET"])
def search_business():
    global biz_map
    biz_map = sync_businesses(biz_map)

    query = request.args.get("q", "")
    limit_param = request.args.get("limit")

    # FILTER PARAMETERS (safe)
    user_lat = safe_float(request.args.get("user_lat"))
    user_lon = safe_float(request.args.get("user_lon"))
    max_distance = safe_float(request.args.get("max_distance"))
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    vector = embed_text(query)

    # search qdrant
    results = qdrant.search("businesses", query_vector=vector, limit=max_results)

    business_uids = [r.payload.get("business_uid") for r in results]
    additional_info = {}

    # fetch SQL details
    if business_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(business_uids))

        cur.execute(f"""
            SELECT *
            FROM business
            WHERE business_uid IN ({placeholders})
        """, business_uids)

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

    # -----------------------------------------------------
    # APPLY FILTERS + ADD DISTANCE (SAFE)
    # -----------------------------------------------------
    filtered = []
    for r in results:
        uid = r.payload.get("business_uid")

        merged = {"score": r.score, **r.payload}

        if uid in additional_info:
            merged.update(additional_info[uid])

        # Compute distance safely
        if user_lat is not None and user_lon is not None:
            dist = haversine_miles(
                user_lat,
                user_lon,
                merged.get("business_latitude"),
                merged.get("business_longitude")
            )
            merged["distance_miles"] = dist

            if max_distance is not None and dist is not None:
                if dist > max_distance:
                    continue

        # Safely filter by rating
        rating = safe_float(merged.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

        filtered.append(merged)

        if len(filtered) >= final_limit:
            break

    return jsonify(filtered)

# ---------------------------------------------------------
# WISHES SYNC
# ---------------------------------------------------------
def sync_wishes(wish_map):
    print("\n==============================")
    print("💫 SYNCING WISHES")
    print("==============================")

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT profile_wish_uid, profile_wish_title, profile_wish_description, updated_at
        FROM profile_wish
    """)
    rows = cur.fetchall()
    conn.close()

    current_state = {r["profile_wish_uid"]: str(r["updated_at"]) for r in rows}

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
@app.route("/search_wishes", methods=["GET"])
def search_wishes():
    global wish_map
    wish_map = sync_wishes(wish_map)

    query = request.args.get("q", "")
    limit_param = request.args.get("limit")

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    vector = embed_text(query)

    results = qdrant.search("wishes", query_vector=vector, limit=max_results)
    results = results[:final_limit]

    wish_uids = [r.payload.get("profile_wish_uid") for r in results]

    additional_info = {}

    if wish_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(wish_uids))

        cur.execute(f"""
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
        """, wish_uids)

        rows = cur.fetchall()
        conn.close()

        for row in rows:
            # sanitize location numeric fields
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))

            additional_info[row["profile_wish_uid"]] = row

    response = []
    for r in results:
        uid = r.payload.get("profile_wish_uid")
        obj = {"score": r.score, **r.payload}

        if uid in additional_info:
            obj.update(additional_info[uid])

        response.append(obj)

    return jsonify(response)


# ---------------------------------------------------------
# EXPERTISE SYNC
# ---------------------------------------------------------
def sync_expertise(exp_map):
    print("\n==============================")
    print("🎓 SYNCING EXPERTISE")
    print("==============================")

    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT profile_expertise_uid, profile_expertise_title,
               profile_expertise_description, updated_at
        FROM profile_expertise
    """)
    rows = cur.fetchall()
    conn.close()

    current_state = {r["profile_expertise_uid"]: str(r["updated_at"]) for r in rows}

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
@app.route("/search_expertise", methods=["GET"])
def search_expertise():
    global exp_map
    exp_map = sync_expertise(exp_map)

    query = request.args.get("q", "")
    limit_param = request.args.get("limit")

    max_results = 99999
    final_limit = get_limit(limit_param, max_results)

    vector = embed_text(query)

    results = qdrant.search("expertise", query_vector=vector, limit=max_results)
    results = results[:final_limit]

    exp_uids = [r.payload.get("profile_expertise_uid") for r in results]
    additional_info = {}

    if exp_uids:
        conn = mysql_connect()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(exp_uids))

        cur.execute(f"""
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
        """, exp_uids)

        rows = cur.fetchall()
        conn.close()

        for row in rows:
            # sanitize numeric fields
            row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
            row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))

            additional_info[row["profile_expertise_uid"]] = row

    response = []
    for r in results:
        uid = r.payload.get("profile_expertise_uid")
        obj = {"score": r.score, **r.payload}

        if uid in additional_info:
            obj.update(additional_info[uid])

        response.append(obj)

    return jsonify(response)


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
