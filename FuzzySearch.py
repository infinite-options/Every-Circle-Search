import os
import math
import pymysql
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from thefuzz import fuzz

load_dotenv()

MYSQL = dict(
    host=os.environ["RDS_HOST"],
    port=int(os.environ["RDS_PORT"]),
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PW"],
    database=os.environ["RDS_DB"],
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


LEXICAL_MIN_SCORE = _env_float("LEXICAL_MIN_SCORE", 0.10)
GLOBAL_BUSINESS_WEIGHT = _env_float("GLOBAL_BUSINESS_WEIGHT", 1.15)
GLOBAL_EXPERTISE_WEIGHT = _env_float("GLOBAL_EXPERTISE_WEIGHT", 0.85)


def mysql_connect():
    return pymysql.connect(**MYSQL)


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def get_limit(param, max_results):
    if param is None or param == "":
        return 120
    value = str(param).strip().upper()
    if value == "ALL":
        return max_results
    if value.isdigit():
        return int(value)
    return 120


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)
    if None in (lat1, lon1, lat2, lon2):
        return None
    r = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def csv_to_tokens(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(csv_to_tokens(v))
        return out
    return [p.strip().lower() for p in str(value).split(",") if p and p.strip()]


def fuzzy_norm(query, text):
    if not query or not text:
        return 0.0
    return fuzz.token_set_ratio(str(query), str(text)) / 100.0


def business_lexical_score(query, row):
    name = row.get("business_name") or ""
    tagline = row.get("business_tag_line") or ""
    bio = row.get("business_short_bio") or ""
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


def business_lexical_breakdown(query, row):
    name = row.get("business_name") or ""
    tagline = row.get("business_tag_line") or ""
    bio = row.get("business_short_bio") or ""
    service_names = csv_to_tokens(row.get("all_service_names") or row.get("bs_service_names"))
    service_tags = csv_to_tokens(row.get("all_service_tags") or row.get("bs_tags"))
    custom_tags = csv_to_tokens(row.get("all_custom_tags") or row.get("custom_tags"))

    name_score = 0.30 * fuzzy_norm(query, name)
    tagline_score = 0.10 * fuzzy_norm(query, tagline)
    bio_score = 0.08 * fuzzy_norm(query, bio)
    service_name_score = 0.22 * fuzzy_norm(query, " ".join(service_names))
    service_tag_score = 0.18 * fuzzy_norm(query, " ".join(service_tags))
    custom_tag_score = 0.22 * fuzzy_norm(query, " ".join(custom_tags))
    final_score = min(1.0, name_score + tagline_score + bio_score + service_name_score + service_tag_score + custom_tag_score)

    return {
        "name_score": name_score,
        "tagline_score": tagline_score,
        "bio_score": bio_score,
        "service_name_score": service_name_score,
        "service_tag_score": service_tag_score,
        "custom_tag_score": custom_tag_score,
        "semantic_score": 0.0,
        "total_lexical_boost": final_score,
        "final_score": final_score,
    }


def expertise_lexical_score(query, row):
    title = row.get("profile_expertise_title") or ""
    desc = row.get("profile_expertise_description") or ""
    details = row.get("profile_expertise_details") or ""
    return min(1.0, 0.55 * fuzzy_norm(query, title) + 0.30 * fuzzy_norm(query, desc) + 0.15 * fuzzy_norm(query, details))


def expertise_lexical_breakdown(query, row):
    title = row.get("profile_expertise_title") or ""
    desc = row.get("profile_expertise_description") or ""
    details = row.get("profile_expertise_details") or ""
    title_score = 0.55 * fuzzy_norm(query, title)
    description_score = 0.30 * fuzzy_norm(query, desc)
    details_score = 0.15 * fuzzy_norm(query, details)
    final_score = min(1.0, title_score + description_score + details_score)
    return {
        "title_score": title_score,
        "description_score": description_score,
        "details_score": details_score,
        "semantic_score": 0.0,
        "total_lexical_boost": final_score,
        "final_score": final_score,
    }


def wish_lexical_score(query, row):
    title = row.get("profile_wish_title") or ""
    desc = row.get("profile_wish_description") or ""
    return min(1.0, 0.60 * fuzzy_norm(query, title) + 0.40 * fuzzy_norm(query, desc))


def wish_lexical_breakdown(query, row):
    title = row.get("profile_wish_title") or ""
    desc = row.get("profile_wish_description") or ""
    title_score = 0.60 * fuzzy_norm(query, title)
    description_score = 0.40 * fuzzy_norm(query, desc)
    final_score = min(1.0, title_score + description_score)
    return {
        "title_score": title_score,
        "description_score": description_score,
        "semantic_score": 0.0,
        "total_lexical_boost": final_score,
        "final_score": final_score,
    }


@app.route("/search_business", methods=["GET"])
def search_business():
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
        GROUP BY b.business_uid
        """
    )
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["business_google_rating"] = safe_float(row.get("business_google_rating"))
        row["business_latitude"] = safe_float(row.get("business_latitude"))
        row["business_longitude"] = safe_float(row.get("business_longitude"))
        score_breakdown = business_lexical_breakdown(query, row)
        row["score"] = score_breakdown["final_score"]
        row["score_breakdown"] = score_breakdown
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


@app.route("/search_expertise", methods=["GET"])
def search_expertise():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    final_limit = get_limit(limit_param, 99999)

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
        """
    )
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        score_breakdown = expertise_lexical_breakdown(query, row)
        row["score"] = score_breakdown["final_score"]
        row["score_breakdown"] = score_breakdown
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        ranked.append(row)

    ranked.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(ranked[:final_limit])


@app.route("/search_wishes", methods=["GET"])
def search_wishes():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    final_limit = get_limit(limit_param, 99999)

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
        """
    )
    rows = cur.fetchall()
    conn.close()

    ranked = []
    for row in rows:
        row["profile_personal_latitude"] = safe_float(row.get("profile_personal_latitude"))
        row["profile_personal_longitude"] = safe_float(row.get("profile_personal_longitude"))
        score_breakdown = wish_lexical_breakdown(query, row)
        row["score"] = score_breakdown["final_score"]
        row["score_breakdown"] = score_breakdown
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        ranked.append(row)

    ranked.sort(key=lambda x: safe_float(x.get("score")) or 0.0, reverse=True)
    return jsonify(ranked[:final_limit])


@app.route("/search_global", methods=["GET"])
def search_global():
    query = (request.args.get("q", "") or "").strip()
    limit_param = request.args.get("limit")
    min_rating = safe_float(request.args.get("min_rating"))
    max_rating = safe_float(request.args.get("max_rating"))
    final_limit = get_limit(limit_param, 99999)

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
        GROUP BY b.business_uid
        """
    )
    biz_rows = cur.fetchall()

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
        """
    )
    exp_rows = cur.fetchall()
    conn.close()

    business_results = []
    for row in biz_rows:
        row["business_google_rating"] = safe_float(row.get("business_google_rating"))
        row["itemType"] = "businesses"
        score_breakdown = business_lexical_breakdown(query, row)
        row["score"] = score_breakdown["final_score"]
        row["score_breakdown"] = score_breakdown
        row["bs_service_names"] = csv_to_tokens(row.get("all_service_names"))
        row["bs_tags"] = csv_to_tokens(row.get("all_service_tags"))
        row["custom_tags"] = csv_to_tokens(row.get("all_custom_tags"))
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        rating = safe_float(row.get("business_google_rating"))
        if rating is not None:
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue
        row["global_score"] = (safe_float(row.get("score")) or 0.0) * GLOBAL_BUSINESS_WEIGHT
        business_results.append(row)

    expertise_results = []
    for row in exp_rows:
        row["itemType"] = "expertise"
        score_breakdown = expertise_lexical_breakdown(query, row)
        row["score"] = score_breakdown["final_score"]
        row["score_breakdown"] = score_breakdown
        if row["score"] < LEXICAL_MIN_SCORE:
            continue
        row["global_score"] = (safe_float(row.get("score")) or 0.0) * GLOBAL_EXPERTISE_WEIGHT
        expertise_results.append(row)

    combined = business_results + expertise_results
    combined.sort(key=lambda x: safe_float(x.get("global_score")) or 0.0, reverse=True)
    return jsonify(combined[:final_limit])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
