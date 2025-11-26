# App.py — every_circle.business (live query, env-driven config) + rating & distance (miles)

import os
import pymysql
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from math import radians, cos, sin, asin, sqrt

# -------------------------
# Load environment
# -------------------------
load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------------
# MySQL (keep your current EC2-working config style)
# -------------------------
MYSQL = dict(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DATABASE", "every_circle")
)

# -------------------------
# Config from .env
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-MiniLM-L3-v2")  # used smaller model
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))
TOP_K = int(os.getenv("TOP_K", 5))

# -------------------------
# Model
# -------------------------
model = SentenceTransformer(MODEL_NAME)

# -------------------------
# Attributes (added latitude/longitude)
# -------------------------
ATTRIBUTES = [
    "business_uid",
    "business_name",
    "business_tag_line",
    "business_short_bio",
    "business_address_line_1",
    "business_city",
    "business_state",
    "business_country",
    "business_google_rating",
    "business_price_level",
    "business_latitude",      # NEW
    "business_longitude"      # NEW
]

TEXT_FIELDS_FOR_EMBEDDING = [
    "business_name",
    "business_tag_line",
    "business_short_bio",
]

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)

def get_db_connection():
    return pymysql.connect(**MYSQL)

def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points (miles)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 3959.0 * c  # Earth radius (miles)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    city_filter = request.args.get("city")
    min_rating = request.args.get("min_rating", type=float)  # rating filter
    user_lat = request.args.get("lat", type=float)           # distance filter
    user_lon = request.args.get("lon", type=float)
    radius_miles = request.args.get("radius_miles", default=5.0, type=float)

    # Step 1: encode query
    query_vector = model.encode(query)

    # Step 2: build SQL with filters (city + rating)
    sql = f"SELECT {', '.join(ATTRIBUTES)} FROM business WHERE 1=1"
    params = []

    if city_filter:
        sql += " AND business_city = %s"
        params.append(city_filter)

    if min_rating is not None:
        sql += " AND business_google_rating >= %s"
        params.append(min_rating)

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    conn.close()

    # Step 3: embed & score results + optional distance filter
    docs = []
    for row in rows:
        doc = dict(zip(ATTRIBUTES, row))

        # semantic score
        text_blob = " ".join(str(doc.get(f, "") or "") for f in TEXT_FIELDS_FOR_EMBEDDING)
        if text_blob.strip():
            doc_vector = model.encode(text_blob)
            score = float(
                np.dot(query_vector, doc_vector) /
                (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            )
        else:
            score = -1.0
        doc["score"] = score

        # distance filter (only if user lat/lon provided)
        if user_lat is not None and user_lon is not None:
            try:
                b_lat = float(doc.get("business_latitude"))
                b_lon = float(doc.get("business_longitude"))
                distance = haversine_miles(user_lat, user_lon, b_lat, b_lon)
                doc["distance_miles"] = round(distance, 2)
                if distance > radius_miles:
                    continue  # skip results outside radius
            except Exception:
                # If coords missing/invalid and user asked for distance filtering, skip
                continue
        else:
            doc["distance_miles"] = None

        docs.append(doc)

    # Step 4: sort & return top-k
    if user_lat is not None and user_lon is not None:
        # Nearest first, then higher semantic score
        docs = sorted(docs, key=lambda x: (x["distance_miles"], -x["score"]))
    else:
        docs = sorted(docs, key=lambda x: -x["score"])

    return jsonify(docs[:TOP_K])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
