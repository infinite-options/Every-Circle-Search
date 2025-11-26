# QdrntTest.py — Infinite Options RDS + Qdrant + OpenAI Tags (Live Sync Version)

import os
import pymysql
import json
import re
import uuid
import threading
import time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI

# -------------------------
# Load environment
# -------------------------
load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------------
# MySQL config (RDS)
# -------------------------
MYSQL = dict(
    host=os.environ["RDS_HOST"],
    port=int(os.environ["RDS_PORT"]),
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PW"],
    database=os.environ["RDS_DB"]
)

# -------------------------
# Config
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
TOP_K = int(os.getenv("TOP_K", "5"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Setup
# -------------------------
app = Flask(__name__)
embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# -------------------------
# Helpers
# -------------------------
def embed_text(text: str):
    return embedder.encode(text).tolist()

def make_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

def mysql_connect():
    return pymysql.connect(**MYSQL)

# -------------------------
# Recreate collections
# -------------------------
def recreate_collections():
    for col in ["wishes", "expertise", "businesses"]:
        qdrant.recreate_collection(
            collection_name=col,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
    print("✅ Qdrant collections recreated.")

# -------------------------
# Full Sync on Startup
# -------------------------
def sync_wishes():
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT profile_wish_uid, profile_wish_title, profile_wish_description, updated_at FROM profile_wish")
    rows = cur.fetchall()
    conn.close()

    points = []
    for row in rows:
        text = f"{row['profile_wish_title']} - {row['profile_wish_description'] or ''}"
        points.append(PointStruct(
            id=make_uuid(row['profile_wish_uid']),
            vector=embed_text(text),
            payload=row
        ))
    qdrant.upsert(collection_name="wishes", points=points)
    print(f"✅ Synced {len(points)} wishes.")
    return {r["profile_wish_uid"]: str(r.get("updated_at")) for r in rows}

def sync_expertise():
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT profile_expertise_uid, profile_expertise_title, profile_expertise_description, updated_at FROM profile_expertise")
    rows = cur.fetchall()
    conn.close()

    points = []
    for row in rows:
        text = f"{row['profile_expertise_title']} - {row['profile_expertise_description'] or ''}"
        points.append(PointStruct(
            id=make_uuid(row['profile_expertise_uid']),
            vector=embed_text(text),
            payload=row
        ))
    qdrant.upsert(collection_name="expertise", points=points)
    print(f"✅ Synced {len(points)} expertise entries.")
    return {r["profile_expertise_uid"]: str(r.get("updated_at")) for r in rows}

def sync_businesses():
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT business_uid, business_name, business_short_bio, business_tag_line, business_city, business_state, business_country, business_google_rating, updated_at FROM business")
    rows = cur.fetchall()
    conn.close()

    points = []
    for row in rows:
        text = f"{row['business_name']} - {row['business_short_bio'] or ''} - {row['business_tag_line'] or ''}"
        tags = [t.strip().lower() for t in (row.get("business_tag_line") or "").split(",") if t.strip()]
        payload = row | {"tags": tags}

        rating = payload.get("business_google_rating")
        if rating is not None:
            try:
                payload["business_google_rating"] = float(rating)
            except (TypeError, ValueError):
                payload["business_google_rating"] = 0.0

        points.append(PointStruct(
            id=make_uuid(row['business_uid']),
            vector=embed_text(text),
            payload=payload
        ))
    qdrant.upsert(collection_name="businesses", points=points)
    print(f"✅ Synced {len(points)} businesses.")
    return {r["business_uid"]: str(r.get("updated_at")) for r in rows}

# -------------------------
# Incremental Live Sync (Polling)
# -------------------------
def upsert_business(uid):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT business_uid, business_name, business_short_bio, business_tag_line, business_city, business_state, business_country, business_google_rating, updated_at FROM business WHERE business_uid=%s", (uid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        qdrant.delete(collection_name="businesses", points_selector={"points": [make_uuid(uid)]})
        print(f"🗑️ Deleted business {uid} from Qdrant.")
        return

    text = f"{row['business_name']} - {row['business_short_bio'] or ''} - {row['business_tag_line'] or ''}"
    tags = [t.strip().lower() for t in (row.get("business_tag_line") or "").split(",") if t.strip()]
    payload = row | {"tags": tags}

    rating = payload.get("business_google_rating")
    if rating is not None:
        try:
            payload["business_google_rating"] = float(rating)
        except (TypeError, ValueError):
            payload["business_google_rating"] = 0.0

    qdrant.upsert(collection_name="businesses", points=[PointStruct(id=make_uuid(uid), vector=embed_text(text), payload=payload)])
    print(f"🔁 Updated business {row['business_name']}")

def upsert_wish(uid):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT profile_wish_uid, profile_wish_title, profile_wish_description, updated_at FROM profile_wish WHERE profile_wish_uid=%s", (uid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        qdrant.delete(collection_name="wishes", points_selector={"points": [make_uuid(uid)]})
        print(f"🗑️ Deleted wish {uid} from Qdrant.")
        return

    text = f"{row['profile_wish_title']} - {row['profile_wish_description'] or ''}"
    qdrant.upsert(collection_name="wishes", points=[PointStruct(id=make_uuid(uid), vector=embed_text(text), payload=row)])
    print(f"🔁 Updated wish {row['profile_wish_title']}")

def upsert_expertise(uid):
    conn = mysql_connect()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT profile_expertise_uid, profile_expertise_title, profile_expertise_description, updated_at FROM profile_expertise WHERE profile_expertise_uid=%s", (uid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        qdrant.delete(collection_name="expertise", points_selector={"points": [make_uuid(uid)]})
        print(f"🗑️ Deleted expertise {uid} from Qdrant.")
        return

    text = f"{row['profile_expertise_title']} - {row['profile_expertise_description'] or ''}"
    qdrant.upsert(collection_name="expertise", points=[PointStruct(id=make_uuid(uid), vector=embed_text(text), payload=row)])
    print(f"🔁 Updated expertise {row['profile_expertise_title']}")

# -------------------------
# Polling Thread
# -------------------------
def poll_table(name, key_field, updated_field, upsert_func, table_map, interval=60):
    print(f"🕒 Started live sync for {name}")
    while True:
        try:
            conn = mysql_connect()
            cur = conn.cursor(pymysql.cursors.DictCursor)
            cur.execute(f"SELECT {key_field}, {updated_field} FROM {name}")
            rows = cur.fetchall()
            conn.close()

            current_state = {r[key_field]: str(r[updated_field]) for r in rows}
            all_keys = set(table_map.keys()) | set(current_state.keys())

            for uid in all_keys:
                if uid not in current_state:
                    qdrant.delete(collection_name=name.rstrip('s'), points_selector={"points": [make_uuid(uid)]})
                    print(f"🗑️ {name} deleted: {uid}")
                    table_map.pop(uid, None)
                elif uid not in table_map or table_map[uid] != current_state[uid]:
                    upsert_func(uid)
                    table_map[uid] = current_state[uid]

        except Exception as e:
            print(f"⚠️ Poll error ({name}):", e)
        time.sleep(interval)

# -------------------------
# Search Endpoint (with filters)
# -------------------------
@app.route("/search_business", methods=["GET"])
def search_business():
    query = request.args.get("q", "")
    tag = request.args.get("tag", "")
    location = request.args.get("location", "")
    min_rating = request.args.get("min_rating", "")
    limit = int(request.args.get("limit", TOP_K))

    vector = embed_text(query) if query else None
    must_filters = []

    if tag:
        must_filters.append({"key": "tags", "match": {"value": tag.lower()}})
    if location:
        must_filters.append({"key": "business_city", "match": {"value": location}})
    if min_rating:
        try:
            must_filters.append({"key": "business_google_rating", "range": {"gte": float(min_rating)}})
        except ValueError:
            pass

    scroll_filter = {"must": must_filters} if must_filters else None

    if vector:
        results = qdrant.search("businesses", query_vector=vector, query_filter=scroll_filter, limit=limit)
        return jsonify([{"score": r.score, **(r.payload or {})} for r in results])
    elif must_filters:
        results, _ = qdrant.scroll("businesses", scroll_filter=scroll_filter, limit=limit)
        return jsonify([r.payload for r in results])
    else:
        return jsonify({"error": "Must provide either q, tag, or filter"}), 400

# -------------------------
# Tag Generation
# -------------------------
def generate_tags(name, description, max_tags=10):
    prompt = f"""
You are a tagging assistant. Generate concise, single-word search tags.

Rules:
- ONE WORD PER TAG
- lowercase only
- Prefer common search terms
- Avoid repeating the name
- Return ONLY a JSON array of 5–10 strings

Name: {name}
Description: {description}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You generate practical, single-word search tags."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        tags = json.loads(content)
        if not isinstance(tags, list):
            tags = [str(tags)]
    except Exception:
        tags = [t.strip() for t in re.split(r"[,;\n]+", content) if t.strip()]
    return tags[:max_tags]

@app.route("/generate_tags", methods=["POST"])
def api_generate_tags():
    data = request.json or {}
    name = data.get("name", "")
    description = data.get("description", "")
    max_tags = int(data.get("max_tags", 10))
    if not description and not name:
        return jsonify({"error": "Name or description required"}), 400
    tags = generate_tags(name, description, max_tags=max_tags)
    return jsonify({"tags": tags})

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    recreate_collections()
    wish_map = sync_wishes()
    exp_map = sync_expertise()
    biz_map = sync_businesses()

    print("✅ Collections synced. Launching Flask + background sync threads...")

    # Background sync threads
    threading.Thread(target=poll_table, args=("business", "business_uid", "updated_at", upsert_business, biz_map), daemon=True).start()
    threading.Thread(target=poll_table, args=("profile_wish", "profile_wish_uid", "updated_at", upsert_wish, wish_map), daemon=True).start()
    threading.Thread(target=poll_table, args=("profile_expertise", "profile_expertise_uid", "updated_at", upsert_expertise, exp_map), daemon=True).start()

    print("🚀 Live sync running. Flask server starting on port 5001...")
    app.run(host="0.0.0.0", port=5001)
