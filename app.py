import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

from recommender import recommend_cold_user_by_preference_text


st.set_page_config(
    page_title="Sistem Rekomendasi Buku Perpustakaan",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Sistem Rekomendasi Buku Berbasis Preferensi Pengguna")

st.markdown(
    """
    Aplikasi ini membantu pengguna memilih buku berdasarkan teks preferensi.
    Contoh input: *Saya suka novel petualangan, psikologi keluarga, motivasi, dan cerita kehidupan.*
    """
)


def find_file(possible_paths):
    """
    Mencari file dari beberapa kemungkinan lokasi.
    Berguna karena file bisa berada di root repo atau di folder artifacts.
    """
    for path in possible_paths:
        p = Path(path)
        if p.exists():
            return p
    return None

def clean_display_value(value, default="-"):
    """
    Membersihkan nilai kosong, NaN, None, dan string 'nan'
    agar tidak tampil di UI.
    """
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    value = str(value).strip()

    if value.lower() in ["", "nan", "none", "null", "<na>", "nat"]:
        return default

    if value.endswith(".0"):
        try:
            return str(int(float(value)))
        except Exception:
            return value

    return value


def get_ddc_group(class_code):
    """
    Mengubah kode DDC menjadi kelompok klasifikasi utama.
    Contoh:
    641.5 -> 600 - Ilmu Terapan
    899.221 -> 800 - Sastra
    """
    code = clean_display_value(class_code, default="")
    if code == "":
        return "-"

    digits = "".join([c for c in code if c.isdigit()])

    if len(digits) < 3:
        return "-"

    try:
        ddc = int(digits[:3])
    except Exception:
        return "-"

    if 0 <= ddc <= 99:
        return "000 - Karya Umum"
    elif 100 <= ddc <= 199:
        return "100 - Filsafat dan Psikologi"
    elif 200 <= ddc <= 299:
        return "200 - Agama"
    elif 300 <= ddc <= 399:
        return "300 - Ilmu Sosial"
    elif 400 <= ddc <= 499:
        return "400 - Bahasa"
    elif 500 <= ddc <= 599:
        return "500 - Ilmu Murni"
    elif 600 <= ddc <= 699:
        return "600 - Ilmu Terapan/Teknologi"
    elif 700 <= ddc <= 799:
        return "700 - Seni dan Rekreasi"
    elif 800 <= ddc <= 899:
        return "800 - Sastra"
    elif 900 <= ddc <= 999:
        return "900 - Sejarah dan Geografi"

    return "-"


def get_best_call_number(row):
    """
    Mengambil nomor panggil terbaik.
    Prioritas:
    1. call_number
    2. class_code
    """
    call_number = clean_display_value(row.get("call_number", ""), default="")
    class_code = clean_display_value(row.get("class_code", ""), default="")

    if call_number != "":
        return call_number

    if class_code != "":
        return class_code

    return "-"

@st.cache_resource
def load_artifacts():
    # -----------------------------------------------------
    # 1. Cari lokasi file content_items
    # -----------------------------------------------------
    content_path = find_file([
        "artifacts/content_items.csv",
        "content_items.csv",
        "artifacts/content_items.parquet",
        "content_items.parquet",
    ])

    availability_path = find_file([
        "artifacts/availability_summary.csv",
        "availability_summary.csv",
        "artifacts/availability_summary.parquet",
        "availability_summary.parquet",
    ])

    vectorizer_path = find_file([
        "artifacts/tfidf_vectorizer.joblib",
        "tfidf_vectorizer.joblib",
    ])

    matrix_path = find_file([
        "artifacts/tfidf_matrix.npz",
        "tfidf_matrix.npz",
    ])

    missing = []
    if content_path is None:
        missing.append("content_items.csv / content_items.parquet")
    if availability_path is None:
        missing.append("availability_summary.csv / availability_summary.parquet")
    if vectorizer_path is None:
        missing.append("tfidf_vectorizer.joblib")
    if matrix_path is None:
        missing.append("tfidf_matrix.npz")

    if missing:
        raise FileNotFoundError(
            "File berikut belum ditemukan di repo: " + ", ".join(missing)
        )

    # -----------------------------------------------------
    # 2. Load content_items
    # -----------------------------------------------------
    if content_path.suffix.lower() == ".parquet":
        content_items = pd.read_parquet(content_path)
    else:
        content_items = pd.read_csv(content_path)

    # -----------------------------------------------------
    # 3. Load availability_summary
    # -----------------------------------------------------
    if availability_path.suffix.lower() == ".parquet":
        availability_summary = pd.read_parquet(availability_path)
    else:
        availability_summary = pd.read_csv(availability_path)

    # -----------------------------------------------------
    # 4. Load TF-IDF artifacts
    # -----------------------------------------------------
    tfidf_vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = load_npz(matrix_path)

    # -----------------------------------------------------
    # 5. Validasi jumlah baris
    # -----------------------------------------------------
    if len(content_items) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"Jumlah baris content_items ({len(content_items)}) "
            f"tidak sama dengan jumlah baris tfidf_matrix ({tfidf_matrix.shape[0]}). "
            "Pastikan content_items.csv dan tfidf_matrix.npz dibuat dari proses yang sama."
        )

    return content_items, availability_summary, tfidf_vectorizer, tfidf_matrix


try:
    content_items, availability_summary, tfidf_vectorizer, tfidf_matrix = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak model: {e}")

    st.info(
        """
        Pastikan file berikut tersedia di root repo atau folder `artifacts/`:

        - `content_items.csv`
        - `availability_summary.csv`
        - `tfidf_vectorizer.joblib`
        - `tfidf_matrix.npz`
        """
    )
    st.stop()


with st.sidebar:
    st.header("Pengaturan")
    top_n = st.slider(
        "Jumlah rekomendasi",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

    st.caption("Model menggunakan kemiripan teks preferensi dengan metadata dan sinopsis buku.")

    st.divider()
    st.write("**Status artefak:**")
    st.write(f"Jumlah buku: {len(content_items):,}")
    st.write(f"Ukuran TF-IDF: {tfidf_matrix.shape[0]:,} x {tfidf_matrix.shape[1]:,}")


preference_text = st.text_area(
    "Masukkan preferensi buku yang Anda inginkan",
    height=130,
    placeholder="Contoh: Saya suka novel petualangan, perjuangan, psikologi keluarga, motivasi, dan cerita kehidupan."
)

button = st.button("Cari Rekomendasi Buku", type="primary")


if button:
    if not preference_text.strip():
        st.warning("Silakan isi preferensi buku terlebih dahulu.")
        st.stop()

    with st.spinner("Sedang mencari rekomendasi terbaik..."):
        recommendations = recommend_cold_user_by_preference_text(
            preference_text=preference_text,
            content_items=content_items,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            availability_summary=availability_summary,
            top_n=top_n,
            candidate_pool=300,
        )

    if recommendations.empty:
        st.warning("Belum ditemukan rekomendasi yang sesuai. Coba gunakan kata kunci lain.")
        st.stop()

    st.success(f"Ditemukan {len(recommendations)} rekomendasi buku.")

    for _, row in recommendations.iterrows():
        title = clean_display_value(row.get("title_raw", "-"))
        author = clean_display_value(row.get("author_raw", "-"))
        publisher = clean_display_value(row.get("publisher", "-"))
        year = clean_display_value(row.get("year_clean", "-"))
        category = clean_display_value(row.get("category", "-"))
        class_code = clean_display_value(row.get("class_code", "-"))
        class_group = clean_display_value(row.get("class_group", ""), default="")
        call_number = get_best_call_number(row)

        if class_group == "":
            class_group = get_ddc_group(class_code)

        copy_count = int(float(row.get("copy_count", 0) or 0))
        explanation = clean_display_value(row.get("explanation", "-"))
        with st.container(border=True):
            st.subheader(f"{int(row.get('rank', 0))}. {row.get('title_raw', '-')}")
            st.write(f"**Penulis:** {author}")
            st.write(f"**Penerbit/Tahun:** {publisher} / {year}")
            st.write(f"**Nomor Panggil:** {call_number}")
            st.write(f"**Kode DDC:** {class_code}")
            st.write(f"**Kelompok Klasifikasi DDC:** {class_group}")
            st.write(f"**Lokasi/Jenis Koleksi:** {category}")
            st.write(f"**Jumlah Eksemplar:** {copy_count}")
            st.write(f"**Alasan Rekomendasi:** {explanation}")

            sinopsis = str(row.get("sinopsis", "")).strip()
            if sinopsis and sinopsis.lower() not in ["nan", "none"]:
                with st.expander("Lihat sinopsis"):
                    st.write(sinopsis)

    st.divider()

    display_cols = [
        "rank",
        "title_raw",
        "author_raw",
        "publisher",
        "year_clean",
        "category",
        "class_group",
        "copy_count",
        "preference_score",
        "cold_user_score",
        "explanation",
    ]
    display_cols = [c for c in display_cols if c in recommendations.columns]

    st.subheader("Tabel Rekomendasi")
    st.dataframe(recommendations[display_cols], use_container_width=True)

    csv_data = recommendations.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ Download hasil rekomendasi CSV",
        data=csv_data,
        file_name="hasil_rekomendasi_buku.csv",
        mime="text/csv"
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        recommendations.to_excel(writer, index=False, sheet_name="rekomendasi")
    excel_buffer.seek(0)

    st.download_button(
        label="⬇️ Download hasil rekomendasi Excel",
        data=excel_buffer,
        file_name="hasil_rekomendasi_buku.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Isi preferensi buku, lalu klik tombol **Cari Rekomendasi Buku**.")
