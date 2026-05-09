# =========================================================
# STREAMLIT HYBRID BOOK RECOMMENDER APP
# CF + CBF/NLP + KG Semantic Boost + Novelty + Query Search
# With deduplication and richer explanation
# =========================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sistem Rekomendasi Buku Hybrid",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
        color: #111827;
    }
    .subtitle {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 24px;
    }
    .book-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 14px;
    }
    .book-title {
        font-size: 19px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 6px;
    }
    .book-meta {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .score-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background-color: #EEF2FF;
        color: #3730A3;
        font-size: 13px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }
    .score-pill-green {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background-color: #ECFDF5;
        color: #065F46;
        font-size: 13px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }
    .reason-box {
        padding: 10px 12px;
        border-radius: 10px;
        background-color: #F9FAFB;
        color: #374151;
        font-size: 14px;
        margin-top: 10px;
        line-height: 1.5;
    }
    .small-note {
        color: #6B7280;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. PATH CONFIG
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"


# =========================================================
# 4. UTILITY FUNCTIONS
# =========================================================

def ensure_col(df, col, default=""):
    if col not in df.columns:
        df[col] = default
    return df


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_for_key(x):
    x = clean_text(x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_title_for_dedupe(x):
    """
    Membuat key judul agar judul yang sama/mirip tidak tampil berulang.
    """
    x = normalize_for_key(x)

    # Hilangkan kata-kata umum dari deskripsi bibliografis
    stop_patterns = [
        r"\bcet\b",
        r"\beditor\b",
        r"\bed\b",
        r"\bvol\b",
        r"\bjilid\b",
        r"\bhlm\b",
        r"\bhal\b"
    ]

    for pat in stop_patterns:
        x = re.sub(pat, " ", x)

    x = re.sub(r"\s+", " ", x).strip()
    return x


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v = s.min()
    max_v = s.max()

    if max_v == min_v:
        return pd.Series(np.ones(len(s)), index=s.index)

    return (s - min_v) / (max_v - min_v)


def normalize_weights(weight_dict):
    total = sum(weight_dict.values())
    if total <= 0:
        n = len(weight_dict)
        return {k: 1 / n for k in weight_dict}

    return {k: v / total for k, v in weight_dict.items()}


def make_class_group_from_code(code):
    if pd.isna(code):
        return ""

    code = str(code)
    match = re.search(r"\b\d{3}\b", code)

    if not match:
        return ""

    n = int(match.group(0))

    if 0 <= n <= 99:
        return "000 Karya umum"
    elif 100 <= n <= 199:
        return "100 Filsafat dan psikologi"
    elif 200 <= n <= 299:
        return "200 Agama"
    elif 300 <= n <= 399:
        return "300 Ilmu sosial"
    elif 400 <= n <= 499:
        return "400 Bahasa"
    elif 500 <= n <= 599:
        return "500 Ilmu murni"
    elif 600 <= n <= 699:
        return "600 Ilmu terapan"
    elif 700 <= n <= 799:
        return "700 Seni dan rekreasi"
    elif 800 <= n <= 899:
        return "800 Sastra"
    elif 900 <= n <= 999:
        return "900 Sejarah dan geografi"

    return ""


def extract_simple_topic(title):
    """
    Mengambil topik sederhana dari judul agar explanation tidak terlalu generik.
    """
    if pd.isna(title):
        return ""

    title = str(title).lower()

    topic_keywords = [
        "kewarganegaraan",
        "pendidikan",
        "manajemen",
        "ekonomi",
        "hukum",
        "politik",
        "administrasi",
        "literasi",
        "perpustakaan",
        "teknologi",
        "komunikasi",
        "sosial",
        "budaya",
        "agama",
        "psikologi",
        "bahasa",
        "sastra",
        "sejarah",
        "keuangan",
        "bisnis",
        "pemerintahan",
        "digital"
    ]

    found = [kw for kw in topic_keywords if kw in title]

    if len(found) == 0:
        return ""

    return ", ".join(found[:2])


def score_level(value):
    value = safe_float(value)

    if value >= 0.75:
        return "sangat kuat"
    elif value >= 0.50:
        return "cukup kuat"
    elif value > 0:
        return "tambahan"
    else:
        return "tidak dominan"


# =========================================================
# 5. LOAD DATA
# =========================================================

@st.cache_data(show_spinner="Membaca data rekomendasi...")
def load_data():
    metadata_path = DATA_DIR / "book_metadata_master_enriched.csv"
    hybrid_path = DATA_DIR / "hybrid_candidates_scores.csv"
    final_path = DATA_DIR / "final_recommendations_topn.csv"
    evaluation_path = DATA_DIR / "evaluation_report.csv"
    coverage_path = DATA_DIR / "coverage_report.csv"

    if not metadata_path.exists():
        st.error(f"File metadata tidak ditemukan: {metadata_path}")
        st.stop()

    book_metadata = pd.read_csv(metadata_path)

    if hybrid_path.exists():
        hybrid_candidates = pd.read_csv(hybrid_path)
    elif final_path.exists():
        hybrid_candidates = pd.read_csv(final_path)
    else:
        st.error("File hybrid_candidates_scores.csv atau final_recommendations_topn.csv tidak ditemukan.")
        st.stop()

    evaluation_report = pd.read_csv(evaluation_path) if evaluation_path.exists() else pd.DataFrame()
    coverage_report = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

    return book_metadata, hybrid_candidates, evaluation_report, coverage_report


book_metadata, hybrid_candidates_raw, evaluation_report, coverage_report = load_data()


# =========================================================
# 6. PREPARE METADATA
# =========================================================

@st.cache_data(show_spinner="Menyiapkan metadata buku...")
def prepare_metadata(book_metadata):
    book_metadata = book_metadata.copy()

    required_meta_cols = [
        "book_id",
        "title_raw",
        "author_raw",
        "author_clean",
        "publisher",
        "publisher_clean",
        "year_clean",
        "class_code",
        "class_group",
        "category",
        "content_text_clean",
        "loan_count",
        "borrower_count",
        "metadata_source",
        "access_type"
    ]

    for col in required_meta_cols:
        default = 0 if col in ["loan_count", "borrower_count"] else ""
        book_metadata = ensure_col(book_metadata, col, default)

    book_metadata["book_id"] = book_metadata["book_id"].astype(str)

    text_cols = [
        "title_raw",
        "author_raw",
        "author_clean",
        "publisher",
        "publisher_clean",
        "year_clean",
        "class_code",
        "class_group",
        "category",
        "content_text_clean",
        "metadata_source",
        "access_type"
    ]

    for col in text_cols:
        book_metadata[col] = book_metadata[col].fillna("").astype(str)

    book_metadata["loan_count"] = pd.to_numeric(
        book_metadata["loan_count"],
        errors="coerce"
    ).fillna(0)

    book_metadata["borrower_count"] = pd.to_numeric(
        book_metadata["borrower_count"],
        errors="coerce"
    ).fillna(0)

    # Jika class_group kosong, buat dari class_code
    mask_empty_group = book_metadata["class_group"].str.strip().eq("")
    book_metadata.loc[mask_empty_group, "class_group"] = (
        book_metadata.loc[mask_empty_group, "class_code"]
        .apply(make_class_group_from_code)
    )

    # Buat search text yang lebih kaya
    book_metadata["search_text"] = (
        book_metadata["title_raw"].fillna("").astype(str) + " " +
        book_metadata["author_raw"].fillna("").astype(str) + " " +
        book_metadata["publisher"].fillna("").astype(str) + " " +
        book_metadata["year_clean"].fillna("").astype(str) + " " +
        book_metadata["class_code"].fillna("").astype(str) + " " +
        book_metadata["class_group"].fillna("").astype(str) + " " +
        book_metadata["category"].fillna("").astype(str) + " " +
        book_metadata["content_text_clean"].fillna("").astype(str)
    ).apply(normalize_for_key)

    book_metadata["title_dedupe_key"] = book_metadata["title_raw"].apply(normalize_title_for_dedupe)
    book_metadata.loc[
        book_metadata["title_dedupe_key"].eq(""),
        "title_dedupe_key"
    ] = book_metadata["book_id"]

    book_metadata["rec_dedupe_key"] = (
        book_metadata["title_dedupe_key"].astype(str) + "|" +
        book_metadata["year_clean"].fillna("").astype(str) + "|" +
        book_metadata["class_code"].fillna("").astype(str)
    )

    book_metadata = book_metadata.drop_duplicates(subset=["book_id"], keep="first").copy()

    return book_metadata


book_metadata = prepare_metadata(book_metadata)


# =========================================================
# 7. PREPARE HYBRID CANDIDATES
# =========================================================

@st.cache_data(show_spinner="Menyiapkan kandidat rekomendasi hybrid...")
def prepare_hybrid_candidates(hybrid_candidates_raw, book_metadata):
    candidates = hybrid_candidates_raw.copy()

    required_candidate_cols = [
        "user_id_hash",
        "book_id",
        "cf_score",
        "cbf_score",
        "kg_score",
        "novelty_score",
        "hybrid_score",
        "explanation"
    ]

    for col in required_candidate_cols:
        default = 0.0 if "score" in col else ""
        candidates = ensure_col(candidates, col, default)

    candidates["user_id_hash"] = candidates["user_id_hash"].fillna("").astype(str)
    candidates["book_id"] = candidates["book_id"].fillna("").astype(str)

    score_cols = ["cf_score", "cbf_score", "kg_score", "novelty_score", "hybrid_score"]

    for col in score_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0.0)

    # Hapus metadata lama dari candidates agar tidak muncul suffix _x/_y saat merge
    metadata_cols_to_remove = [
        "title_raw",
        "author_raw",
        "author_clean",
        "publisher",
        "publisher_clean",
        "year_clean",
        "class_code",
        "class_group",
        "category",
        "content_text_clean",
        "loan_count",
        "borrower_count",
        "metadata_source",
        "access_type",
        "title_dedupe_key",
        "rec_dedupe_key",
        "search_text"
    ]

    keep_cols = [
        c for c in candidates.columns
        if c not in metadata_cols_to_remove
    ]

    candidates = candidates[keep_cols].copy()

    meta_cols = [
        "book_id",
        "title_raw",
        "author_raw",
        "author_clean",
        "publisher",
        "publisher_clean",
        "year_clean",
        "class_code",
        "class_group",
        "category",
        "loan_count",
        "borrower_count",
        "metadata_source",
        "access_type",
        "title_dedupe_key",
        "rec_dedupe_key",
        "search_text"
    ]

    meta = book_metadata[meta_cols].drop_duplicates(subset=["book_id"]).copy()

    candidates = candidates.merge(
        meta,
        on="book_id",
        how="left"
    )

    # Fallback jika metadata tidak ditemukan
    for col in meta_cols:
        if col != "book_id" and col not in candidates.columns:
            candidates[col] = ""

    text_cols = [
        "title_raw",
        "author_raw",
        "author_clean",
        "publisher",
        "publisher_clean",
        "year_clean",
        "class_code",
        "class_group",
        "category",
        "metadata_source",
        "access_type",
        "title_dedupe_key",
        "rec_dedupe_key",
        "search_text"
    ]

    for col in text_cols:
        candidates[col] = candidates[col].fillna("").astype(str)

    candidates["loan_count"] = pd.to_numeric(candidates["loan_count"], errors="coerce").fillna(0)
    candidates["borrower_count"] = pd.to_numeric(candidates["borrower_count"], errors="coerce").fillna(0)

    # Jika novelty_score belum ada, hitung dari loan_count
    if candidates["novelty_score"].sum() == 0:
        max_pop = np.log1p(candidates["loan_count"].max())
        if max_pop > 0:
            candidates["novelty_score"] = 1 - (np.log1p(candidates["loan_count"]) / max_pop)
        else:
            candidates["novelty_score"] = 1.0

    return candidates


hybrid_candidates = prepare_hybrid_candidates(hybrid_candidates_raw, book_metadata)


# =========================================================
# 8. BUILD TF-IDF SEARCH INDEX
# =========================================================

@st.cache_resource(show_spinner="Membangun indeks pencarian TF-IDF...")
def build_tfidf_index(metadata_df):
    texts = metadata_df["search_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = build_tfidf_index(book_metadata)


@st.cache_data(show_spinner=False)
def compute_query_scores(query, metadata_df):
    query = normalize_for_key(query)

    result = metadata_df[["book_id"]].copy()

    if query == "":
        result["query_score"] = 0.0
        return result

    query_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()

    result["query_score"] = scores
    result["query_score"] = safe_minmax(result["query_score"])

    return result


# =========================================================
# 9. EXPLANATION GENERATOR
# =========================================================

def generate_richer_explanation(row):
    """
    Explanation berbasis kontribusi CF, CBF, KG, novelty, query, dan metadata.
    """
    title = row.get("title_raw", "")
    class_group = row.get("class_group", "")

    cf = safe_float(row.get("cf_score", 0))
    cbf = safe_float(row.get("cbf_score", 0))
    kg = safe_float(row.get("kg_score", 0))
    novelty = safe_float(row.get("novelty_score", 0))
    query_score = safe_float(row.get("query_score", 0))

    topic = extract_simple_topic(title)

    reasons = []

    # Keyword/query explanation
    if query_score >= 0.70:
        if topic:
            reasons.append(
                f"sangat sesuai dengan kata kunci pencarian dan berkaitan dengan topik {topic}"
            )
        else:
            reasons.append(
                "sangat sesuai dengan kata kunci pencarian pada metadata buku"
            )
    elif query_score >= 0.30:
        reasons.append(
            "memiliki kecocokan metadata dengan kata kunci pencarian"
        )

    # CBF explanation
    if cbf >= 0.70:
        if topic:
            reasons.append(
                f"metadata buku sangat mirip dengan pola minat pengguna, terutama pada topik {topic}"
            )
        else:
            reasons.append(
                "metadata buku sangat mirip dengan buku yang pernah dipinjam pengguna"
            )
    elif cbf >= 0.35:
        if topic:
            reasons.append(
                f"memiliki kedekatan konten dengan topik {topic}"
            )
        else:
            reasons.append(
                "memiliki kedekatan konten dengan riwayat peminjaman pengguna"
            )

    # KG explanation
    if kg >= 0.70 and str(class_group).strip() != "":
        reasons.append(
            f"memiliki relasi semantik yang kuat pada kelompok klasifikasi {class_group}"
        )
    elif kg >= 0.30 and str(class_group).strip() != "":
        reasons.append(
            f"berada pada kelompok klasifikasi yang relevan dengan riwayat pengguna ({class_group})"
        )

    # CF explanation
    if cf >= 0.70:
        reasons.append(
            "diprediksi relevan berdasarkan pola peminjaman pengguna lain yang serupa"
        )
    elif cf >= 0.30:
        reasons.append(
            "mendapat sinyal tambahan dari collaborative filtering"
        )

    # Novelty explanation
    if novelty >= 0.75:
        reasons.append(
            "termasuk koleksi yang belum terlalu populer sehingga menambah kebaruan rekomendasi"
        )
    elif novelty >= 0.50:
        reasons.append(
            "memberikan variasi terhadap daftar rekomendasi"
        )

    # Fallback
    if len(reasons) == 0:
        if str(class_group).strip() != "":
            reasons.append(
                f"masih berada dalam area topik {class_group} dan memiliki kecocokan berdasarkan skor hybrid"
            )
        else:
            reasons.append(
                "memiliki kecocokan berdasarkan kombinasi skor collaborative filtering, content-based filtering, knowledge graph, dan novelty"
            )

    # Batasi maksimal 3 alasan agar tampilan tetap ringkas
    reasons = reasons[:3]

    return "Direkomendasikan karena " + "; ".join(reasons) + "."


def get_dominant_component(row):
    score_map = {
        "CF": safe_float(row.get("cf_score", 0)),
        "CBF": safe_float(row.get("cbf_score", 0)),
        "KG": safe_float(row.get("kg_score", 0)),
        "Novelty": safe_float(row.get("novelty_score", 0)),
        "Keyword": safe_float(row.get("query_score", 0))
    }

    return max(score_map, key=score_map.get)


# =========================================================
# 10. RECOMMENDATION ENGINE
# =========================================================

def build_recommendation(
    query,
    mode,
    selected_user,
    selected_class,
    selected_source,
    top_n,
    weights,
    max_per_class=3
):
    query_scores = compute_query_scores(query, book_metadata)

    # Mode 1: personal user + keyword
    if mode == "Berdasarkan user + kata kunci" and selected_user:
        base = hybrid_candidates[
            hybrid_candidates["user_id_hash"].astype(str) == str(selected_user)
        ].copy()

        if base.empty:
            st.warning("User tidak memiliki kandidat rekomendasi. Sistem memakai mode kata kunci umum.")
            base = book_metadata[["book_id"]].copy()
            base["user_id_hash"] = selected_user
            base["cf_score"] = 0.0
            base["cbf_score"] = 0.0
            base["kg_score"] = 0.0
            base["hybrid_score"] = 0.0
            base["novelty_score"] = 0.0

            base = base.merge(
                book_metadata.drop_duplicates("book_id"),
                on="book_id",
                how="left"
            )

    # Mode 2: search umum berbasis kata kunci
    else:
        base = book_metadata.copy()
        base["user_id_hash"] = ""
        base["cf_score"] = 0.0
        base["cbf_score"] = 0.0
        base["kg_score"] = 0.0
        base["hybrid_score"] = 0.0

        max_pop = np.log1p(base["loan_count"].max())
        if max_pop > 0:
            base["novelty_score"] = 1 - (np.log1p(base["loan_count"]) / max_pop)
        else:
            base["novelty_score"] = 1.0

    # Gabungkan query score
    base = base.merge(
        query_scores,
        on="book_id",
        how="left"
    )

    base["query_score"] = base["query_score"].fillna(0.0)

    # Filter kata kunci: jika user mengisi query, ambil yang query_score > 0
    if normalize_for_key(query) != "":
        base = base[base["query_score"] > 0].copy()

    # Filter klasifikasi
    if selected_class:
        base = base[
            base["class_group"].fillna("").astype(str).isin(selected_class)
        ].copy()

    # Filter sumber metadata
    if selected_source:
        base = base[
            base["metadata_source"].fillna("").astype(str).isin(selected_source)
        ].copy()

    if base.empty:
        return pd.DataFrame()

    # Normalisasi skor pada subset hasil agar bobot user fair
    score_cols = ["cf_score", "cbf_score", "kg_score", "novelty_score", "query_score"]

    for col in score_cols:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)
        base[col] = safe_minmax(base[col])

    base["dynamic_score"] = (
        weights["cf"] * base["cf_score"] +
        weights["cbf"] * base["cbf_score"] +
        weights["kg"] * base["kg_score"] +
        weights["novelty"] * base["novelty_score"] +
        weights["query"] * base["query_score"]
    )

    # Jika hybrid_score tersedia, bisa digunakan sebagai secondary sorting
    base["hybrid_score"] = pd.to_numeric(
        base.get("hybrid_score", 0),
        errors="coerce"
    ).fillna(0.0)

    # Deduplikasi berdasarkan judul + tahun + kelas
    before_dedup = len(base)

    base = (
        base
        .sort_values(
            ["dynamic_score", "hybrid_score", "novelty_score"],
            ascending=[False, False, False]
        )
        .drop_duplicates(subset=["rec_dedupe_key"], keep="first")
        .copy()
    )

    after_dedup = len(base)
    removed_duplicates = before_dedup - after_dedup

    # Diversity sederhana: batasi jumlah buku dari class_group yang sama
    selected_rows = []
    class_counter = {}

    for _, row in base.iterrows():
        cls = str(row.get("class_group", "") or "")

        if cls != "" and class_counter.get(cls, 0) >= max_per_class:
            continue

        selected_rows.append(row)
        class_counter[cls] = class_counter.get(cls, 0) + 1

        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) == 0:
        return pd.DataFrame()

    result = pd.DataFrame(selected_rows)
    result["rank"] = range(1, len(result) + 1)

    result["dominant_component"] = result.apply(get_dominant_component, axis=1)
    result["explanation"] = result.apply(generate_richer_explanation, axis=1)
    result["removed_duplicate_count"] = removed_duplicates

    return result


# =========================================================
# 11. SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Pengaturan Rekomendasi")

    mode = st.radio(
        "Mode rekomendasi",
        ["Berdasarkan kata kunci", "Berdasarkan user + kata kunci"],
        index=0
    )

    query = st.text_input(
        "Masukkan kata kunci / judul / topik buku",
        placeholder="Contoh: kewarganegaraan, manajemen, literasi digital"
    )

    top_n = st.slider(
        "Jumlah rekomendasi",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    max_per_class = st.slider(
        "Maksimal buku per kelompok klasifikasi",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    st.divider()

    selected_user = ""

    if mode == "Berdasarkan user + kata kunci":
        user_options = (
            hybrid_candidates["user_id_hash"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        typed_user = st.text_input(
            "Cari / masukkan user_id_hash",
            placeholder="Contoh: U_00404cf27d71b0"
        )

        if typed_user.strip() != "":
            selected_user = typed_user.strip()
        else:
            selected_user = st.selectbox(
                "Atau pilih user dari daftar",
                options=user_options[:500] if len(user_options) > 500 else user_options
            )

    st.divider()

    st.subheader("Filter Metadata")

    class_options = sorted(
        [
            x for x in book_metadata["class_group"].dropna().astype(str).unique()
            if x.strip() != ""
        ]
    )

    selected_class = st.multiselect(
        "Kelompok klasifikasi",
        options=class_options
    )

    source_options = sorted(
        [
            x for x in book_metadata["metadata_source"].dropna().astype(str).unique()
            if x.strip() != ""
        ]
    )

    selected_source = st.multiselect(
        "Sumber metadata",
        options=source_options
    )

    st.divider()

    st.subheader("Pembobotan Model")

    w_cf = st.slider("Collaborative Filtering", 0, 100, 45)
    w_cbf = st.slider("Content-Based Filtering / NLP", 0, 100, 30)
    w_kg = st.slider("Knowledge Graph", 0, 100, 10)
    w_novelty = st.slider("Novelty / Long-tail", 0, 100, 5)
    w_query = st.slider("Keyword Relevance", 0, 100, 10)

    weights = normalize_weights({
        "cf": w_cf,
        "cbf": w_cbf,
        "kg": w_kg,
        "novelty": w_novelty,
        "query": w_query
    })

    st.caption(
        f"Bobot aktif: CF={weights['cf']:.2f}, "
        f"CBF={weights['cbf']:.2f}, "
        f"KG={weights['kg']:.2f}, "
        f"Novelty={weights['novelty']:.2f}, "
        f"Keyword={weights['query']:.2f}"
    )


# =========================================================
# 12. MAIN PAGE
# =========================================================

st.markdown(
    "<div class='main-title'>📚 Sistem Rekomendasi Buku Hybrid</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Rekomendasi buku berbasis Collaborative Filtering, Content-Based Filtering, Knowledge Graph, Novelty, dan Keyword Relevance.</div>",
    unsafe_allow_html=True
)


# =========================================================
# 13. SUMMARY METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Jumlah Buku", f"{book_metadata['book_id'].nunique():,}")

with m2:
    st.metric("Jumlah Kandidat", f"{len(hybrid_candidates):,}")

with m3:
    user_count = hybrid_candidates["user_id_hash"].nunique()
    st.metric("Jumlah User", f"{user_count:,}")

with m4:
    if not coverage_report.empty and "catalog_coverage" in coverage_report.columns:
        st.metric("Catalog Coverage", f"{coverage_report['catalog_coverage'].iloc[0]:.2%}")
    else:
        st.metric("Catalog Coverage", "-")


# =========================================================
# 14. BUILD RECOMMENDATIONS
# =========================================================

recommendations = build_recommendation(
    query=query,
    mode=mode,
    selected_user=selected_user,
    selected_class=selected_class,
    selected_source=selected_source,
    top_n=top_n,
    weights=weights,
    max_per_class=max_per_class
)


# =========================================================
# 15. DISPLAY RESULT
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Rekomendasi", "📊 Detail Skor", "🔍 Analisis Duplikasi", "📈 Evaluasi Model"]
)

with tab1:
    st.subheader("Hasil Rekomendasi")

    if recommendations.empty:
        st.info("Belum ada rekomendasi yang cocok. Coba ubah kata kunci, bobot, atau filter.")
    else:
        removed_dup = int(recommendations["removed_duplicate_count"].iloc[0])
        st.caption(f"Jumlah kandidat mirip/duplikat yang dihapus sebelum Top-N: {removed_dup:,}")

        for _, row in recommendations.iterrows():
            title = str(row.get("title_raw", "") or "")
            author = str(row.get("author_raw", "") or "")
            publisher = str(row.get("publisher", "") or "")
            year = str(row.get("year_clean", "") or "")
            class_group = str(row.get("class_group", "") or "")
            explanation = str(row.get("explanation", "") or "")
            dynamic_score = safe_float(row.get("dynamic_score", 0))
            dominant_component = str(row.get("dominant_component", "") or "")

            if title.strip() == "" or title.lower() in ["nan", "none", "<na>"]:
                title = "Judul tidak tersedia"

            st.markdown(
                f"""
                <div class="book-card">
                    <div class="book-title">{int(row['rank'])}. {title}</div>
                    <div class="book-meta">
                        <b>Pengarang:</b> {author if author else '-'}<br>
                        <b>Penerbit:</b> {publisher if publisher else '-'} |
                        <b>Tahun:</b> {year if year else '-'} |
                        <b>Kelas:</b> {class_group if class_group else '-'}
                    </div>

                    <span class="score-pill-green">Final Score: {dynamic_score:.3f}</span>
                    <span class="score-pill">Dominan: {dominant_component}</span>
                    <span class="score-pill">CF: {safe_float(row.get('cf_score', 0)):.3f}</span>
                    <span class="score-pill">CBF: {safe_float(row.get('cbf_score', 0)):.3f}</span>
                    <span class="score-pill">KG: {safe_float(row.get('kg_score', 0)):.3f}</span>
                    <span class="score-pill">Novelty: {safe_float(row.get('novelty_score', 0)):.3f}</span>
                    <span class="score-pill">Keyword: {safe_float(row.get('query_score', 0)):.3f}</span>

                    <div class="reason-box">
                        <b>Alasan rekomendasi:</b><br>
                        {explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab2:
    st.subheader("Detail Skor Rekomendasi")

    if recommendations.empty:
        st.info("Tidak ada data skor untuk ditampilkan.")
    else:
        display_cols = [
            "rank",
            "book_id",
            "title_raw",
            "author_raw",
            "publisher",
            "year_clean",
            "class_group",
            "cf_score",
            "cbf_score",
            "kg_score",
            "novelty_score",
            "query_score",
            "dynamic_score",
            "dominant_component",
            "explanation"
        ]

        for col in display_cols:
            if col not in recommendations.columns:
                recommendations[col] = ""

        st.dataframe(
            recommendations[display_cols],
            use_container_width=True,
            hide_index=True
        )

        csv = recommendations[display_cols].to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download hasil rekomendasi CSV",
            data=csv,
            file_name="hasil_rekomendasi_buku.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Analisis Deduplikasi Rekomendasi")

    st.markdown(
        """
        Deduplikasi dilakukan berdasarkan kombinasi **judul yang dinormalisasi + tahun + kode klasifikasi**.
        Tujuannya agar satu judul atau judul yang sangat mirip tidak muncul berulang dalam daftar Top-N.
        """
    )

    if recommendations.empty:
        st.info("Tidak ada data deduplikasi untuk ditampilkan.")
    else:
        dedupe_cols = [
            "rank",
            "book_id",
            "title_raw",
            "year_clean",
            "class_code",
            "class_group",
            "rec_dedupe_key",
            "dynamic_score"
        ]

        for col in dedupe_cols:
            if col not in recommendations.columns:
                recommendations[col] = ""

        st.dataframe(
            recommendations[dedupe_cols],
            use_container_width=True,
            hide_index=True
        )

        class_dist = (
            recommendations["class_group"]
            .fillna("")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        class_dist.columns = ["class_group", "count"]

        st.subheader("Distribusi Kelompok Klasifikasi pada Hasil Top-N")
        st.dataframe(class_dist, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Evaluasi Model Offline")

    if evaluation_report.empty:
        st.info("File evaluation_report.csv belum tersedia.")
    else:
        st.dataframe(
            evaluation_report,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Coverage Report")

    if coverage_report.empty:
        st.info("File coverage_report.csv belum tersedia.")
    else:
        st.dataframe(
            coverage_report,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 16. FOOTER
# =========================================================

st.caption(
    "Aplikasi ini melakukan re-ranking dinamis berdasarkan bobot yang dipilih pengguna. "
    "Model utama tetap berasal dari hasil preprocessing dan model learning sebelumnya."
)
