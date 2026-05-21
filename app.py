import io
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


@st.cache_resource
def load_artifacts():
    content_items = pd.read_parquet("artifacts/content_items.parquet")
    availability_summary = pd.read_parquet("artifacts/availability_summary.parquet")
    tfidf_vectorizer = joblib.load("artifacts/tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz("artifacts/tfidf_matrix.npz")
    return content_items, availability_summary, tfidf_vectorizer, tfidf_matrix


try:
    content_items, availability_summary, tfidf_vectorizer, tfidf_matrix = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak model: {e}")
    st.stop()


with st.sidebar:
    st.header("Pengaturan")
    top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=20, value=10, step=1)
    st.caption("Model menggunakan kemiripan teks preferensi dengan metadata dan sinopsis buku.")


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
        with st.container(border=True):
            st.subheader(f"{int(row['rank'])}. {row.get('title_raw', '-')}")
            st.write(f"**Penulis:** {row.get('author_raw', '-')}")
            st.write(f"**Penerbit/Tahun:** {row.get('publisher', '-')} / {row.get('year_clean', '-')}")
            st.write(f"**Kategori/Klasifikasi:** {row.get('category', '-')} / {row.get('class_group', '-')}")
            st.write(f"**Jumlah eksemplar:** {int(row.get('copy_count', 0))}")
            st.write(f"**Alasan rekomendasi:** {row.get('explanation', '-')}")

            sinopsis = str(row.get("sinopsis", "")).strip()
            if sinopsis:
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
