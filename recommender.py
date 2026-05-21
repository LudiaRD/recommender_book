import re
import numpy as np
import pandas as pd


def clean_query_text(x):
    x = "" if pd.isna(x) else str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def minmax_array(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    if np.nanmax(x) == np.nanmin(x):
        return np.zeros(len(x))
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def recommend_cold_user_by_preference_text(
    preference_text,
    content_items,
    tfidf_vectorizer,
    tfidf_matrix,
    availability_summary=None,
    top_n=10,
    candidate_pool=300,
):
    rec = content_items.copy().reset_index(drop=True)
    rec["content_idx"] = rec.index
    rec["book_id"] = rec["book_id"].astype(str)

    for c in ["loan_count", "borrower_count"]:
        if c not in rec.columns:
            rec[c] = 0
        rec[c] = pd.to_numeric(rec[c], errors="coerce").fillna(0)

    query_vec = tfidf_vectorizer.transform([clean_query_text(preference_text)])
    preference_raw = tfidf_matrix.dot(query_vec.T).toarray().ravel()
    rec["preference_score"] = minmax_array(preference_raw)

    popularity_raw = np.log1p(rec["loan_count"].values)
    rec["popularity_score"] = minmax_array(popularity_raw)
    rec["novelty_score"] = minmax_array(1 / (1 + popularity_raw))

    if availability_summary is not None and not availability_summary.empty:
        rec = rec.merge(availability_summary, on="book_id", how="left")
    else:
        rec["copy_count"] = 0
        rec["access_type_sample"] = ""

    rec["copy_count"] = pd.to_numeric(rec["copy_count"], errors="coerce").fillna(0)
    rec["availability_score"] = minmax_array(rec["copy_count"].values)

    rec["cold_user_score"] = (
        0.70 * rec["preference_score"]
        + 0.12 * rec["popularity_score"]
        + 0.13 * rec["novelty_score"]
        + 0.05 * rec["availability_score"]
    )

    rec["dedupe_key"] = (
        rec["title_raw"].fillna("").map(clean_query_text)
        + "|"
        + rec["year_clean"].fillna("").astype(str)
        + "|"
        + rec["class_code"].fillna("").astype(str)
    )

    rec = (
        rec.sort_values(["cold_user_score", "preference_score"], ascending=False)
        .drop_duplicates("dedupe_key")
        .head(candidate_pool)
        .copy()
    )

    # Filter tambahan agar tidak menampilkan buku tanpa judul dan tanpa eksemplar
    rec = rec[
        rec["title_raw"].fillna("").astype(str).str.strip().ne("")
        & (rec["copy_count"] > 0)
    ].copy()

    rec = rec.head(top_n).copy()
    rec["rank"] = range(1, len(rec) + 1)

    def make_reason(row):
        reasons = []

        if row.get("preference_score", 0) >= 0.60:
            if bool(row.get("has_synopsis", False)):
                reasons.append("sesuai dengan preferensi pengguna berdasarkan metadata dan sinopsis")
            else:
                reasons.append("sesuai dengan preferensi pengguna berdasarkan metadata bibliografis")

        if row.get("novelty_score", 0) >= 0.60:
            reasons.append("memberikan unsur kebaruan karena tidak hanya berdasarkan buku populer")

        if row.get("copy_count", 0) > 0:
            reasons.append(f"memiliki {int(row.get('copy_count', 0))} eksemplar pada metadata koleksi")

        if str(row.get("class_group", "")).strip() != "":
            reasons.append(f"berada pada kelompok klasifikasi {row.get('class_group')}")

        if len(reasons) == 0:
            reasons.append("dipilih berdasarkan kombinasi preferensi teks, popularitas, kebaruan, dan ketersediaan koleksi")

        return "Direkomendasikan karena " + "; ".join(reasons[:4]) + "."

    rec["explanation"] = rec.apply(make_reason, axis=1)
    rec["recommendation_mode"] = "cold_user_preference_text"
    rec["input_preference"] = preference_text

    output_cols = [
        "recommendation_mode",
        "input_preference",
        "rank",
        "book_id",
        "title_raw",
        "author_raw",
        "publisher",
        "year_clean",
        "category",
        "class_code",
        "class_group",
        "has_synopsis",
        "content_source_level",
        "sinopsis",
        "copy_count",
        "access_type_sample",
        "loan_count",
        "borrower_count",
        "preference_score",
        "popularity_score",
        "novelty_score",
        "availability_score",
        "cold_user_score",
        "explanation",
    ]

    output_cols = [c for c in output_cols if c in rec.columns]
    return rec[output_cols].reset_index(drop=True)
