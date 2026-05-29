"""
Property Similarity Finder — Streamlit app.
Run: streamlit run similarity_app.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Property Similarity Finder", layout="wide")


@st.cache_resource
def load_artifacts():
    sim = np.load(ARTIFACTS / "sim_unified.npy")
    idx = pd.read_csv(ARTIFACTS / "sim_index_unified.csv", dtype={"postal_code": str})
    idx["postal_code"] = idx["postal_code"].str.zfill(6)

    clusters = pd.read_csv(ARTIFACTS / "cluster_labels.csv", dtype={"postal_code": str})
    clusters["postal_code"] = clusters["postal_code"].str.zfill(6)

    names = pd.read_csv(ARTIFACTS / "cluster_names.csv")
    names = names[["cluster_id", "auto_label"]].rename(columns={"auto_label": "cluster_label"})

    universe = pd.read_csv(ARTIFACTS / "prop_universe.csv", dtype={"postal_code": str})
    universe["postal_code"] = universe["postal_code"].str.zfill(6)

    clusters = clusters.merge(names, on="cluster_id", how="left")
    clusters["cluster_label"] = clusters["cluster_label"].fillna("Outlier (unique property)")

    return sim, idx, clusters, universe


def get_similar(postal_code, top_n, boost, exclude_residential, min_coverage, sim, idx, clusters, universe):
    pc = str(postal_code).strip().zfill(6)
    rows = idx[idx["postal_code"] == pc]["row_idx"].values
    if not len(rows):
        return None, None, None, None

    row_i = rows[0]
    scores = sim[row_i].copy()

    query_info = clusters[clusters["postal_code"] == pc]
    query_cluster = int(query_info["cluster_id"].values[0]) if len(query_info) else -1
    query_label = query_info["cluster_label"].values[0] if len(query_info) else "Unknown"

    if query_cluster >= 0:
        cluster_arr = clusters.set_index("postal_code")["cluster_id"].reindex(
            idx["postal_code"].values
        ).values
        scores[cluster_arr == query_cluster] += boost

    scores = scores.clip(0, 1)

    score_map = dict(zip(idx["postal_code"].values, scores))

    results = universe[universe["postal_code"] != pc].copy()
    results = results.merge(
        clusters[["postal_code", "cluster_id", "cluster_label", "outlier_score"]],
        on="postal_code", how="left"
    )
    results["similarity"] = results["postal_code"].map(score_map)
    results["same_cluster"] = results["cluster_id"] == query_cluster
    results["matrix_count_int"] = results["matrix_count"].fillna(0).astype(int)
    results["data_coverage"] = results["matrix_count_int"].map(
        {0: "Low", 1: "Low", 2: "Medium", 3: "High"}
    ).fillna("Low")

    if exclude_residential:
        results = results[results["zoning"].fillna("").str.lower() != "residential"]

    # Minimum coverage filter
    if min_coverage == "Medium+":
        results = results[results["matrix_count_int"] >= 2]
    elif min_coverage == "High only":
        results = results[results["matrix_count_int"] == 3]

    results = (
        results.sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    results.index += 1
    results.index.name = "rank"

    display_cols = [
        "postal_code", "building_name", "street_address", "planning_area",
        "zoning", "gfa_sqm", "cluster_label", "similarity", "same_cluster", "data_coverage",
    ]
    return results[display_cols].reset_index(), query_label, query_cluster, query_info


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Property Similarity Finder")
st.caption("Finds the most similar industrial/commercial properties by postal code.")

sim, idx, clusters, universe = load_artifacts()

with st.sidebar:
    st.header("Query Settings")
    postal_input = st.text_input("Postal Code", placeholder="e.g. 569058", max_chars=6)
    top_n = st.slider("Number of results", min_value=5, max_value=20, value=10)
    boost = st.slider("Same-cluster boost", min_value=0.0, max_value=0.30, value=0.15, step=0.05,
                      help="Adds this value to properties in the same cluster as the query.")
    min_coverage = st.selectbox(
        "Minimum result coverage",
        options=["All", "Medium+", "High only"],
        index=1,
        help="Filter results by how many feature blocks (physical / location / amenity) are available. "
             "'Medium+' requires at least 2 of 3 blocks — recommended to avoid data-sparse matches."
    )
    exclude_res = st.checkbox("Exclude residential zoning", value=True)
    run = st.button("Find Similar Properties", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Universe: {len(universe):,} properties")

if run:
    if not postal_input.strip():
        st.warning("Enter a postal code to search.")
    else:
        out = get_similar(
            postal_input, top_n, boost, exclude_res, min_coverage,
            sim, idx, clusters, universe
        )

        if out[0] is None:
            pc_clean = str(postal_input).strip().zfill(6)
            st.error(
                f"Postal code **{pc_clean}** not found in the similarity index "
                f"({len(idx):,} properties indexed)."
            )
        else:
            results_df, query_label, query_cluster, query_info = out

            # Query property info + coverage warning
            pc_clean = str(postal_input).strip().zfill(6)
            query_meta = universe[universe["postal_code"] == pc_clean]
            if len(query_meta):
                q = query_meta.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Building", q.get("building_name") or "—")
                col2.metric("Zoning", q.get("zoning") or "—")
                col3.metric("GFA (sqm)", f"{q['gfa_sqm']:,.0f}" if pd.notna(q.get("gfa_sqm")) else "—")
                col4.metric("Planning Area", q.get("planning_area") or "—")

                query_mc = int(q.get("matrix_count", 0) or 0)
                if query_mc < 2:
                    st.error(
                        f"**Low data coverage for this property** (only {query_mc}/3 feature blocks available). "
                        "Properties with sparse data collapse to similar zero-vectors in the model — "
                        "similarity scores will be unreliably high. Results should be treated with caution. "
                        "Consider using 'Medium+' or 'High only' result coverage filter to surface better-quality matches."
                    )

            if query_cluster >= 0:
                st.info(f"**Cluster {query_cluster}** — {query_label}")
            else:
                st.warning("This property is a unique outlier — results use raw cosine similarity without cluster context.")

            # Degenerate result detection
            if len(results_df) > 0:
                top_scores = results_df["similarity"].dropna()
                if len(top_scores) >= 3 and (top_scores >= 0.999).sum() >= 3:
                    st.warning(
                        f"{(top_scores >= 0.999).sum()} results have similarity = 1.000. "
                        "This usually means these properties share the same missing/sparse feature profile "
                        "rather than being genuinely similar. Switch the coverage filter to 'Medium+' for more reliable results."
                    )

            st.divider()
            st.subheader(f"Top {top_n} Similar Properties")

            display = results_df.copy()
            display["similarity"] = display["similarity"].map("{:.3f}".format)
            display["gfa_sqm"] = display["gfa_sqm"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
            )
            display["same_cluster"] = display["same_cluster"].map({True: "✓", False: ""})
            display = display.rename(columns={
                "rank": "#",
                "postal_code": "Postal",
                "building_name": "Building",
                "street_address": "Address",
                "planning_area": "Planning Area",
                "zoning": "Zone",
                "gfa_sqm": "GFA (sqm)",
                "cluster_label": "Cluster",
                "similarity": "Sim Score",
                "same_cluster": "Same Cluster",
                "data_coverage": "Coverage",
            })
            st.dataframe(display.set_index("#"), use_container_width=True)

            # Downloads
            raw = results_df.drop(columns=["rank"]) if "rank" in results_df.columns else results_df
            csv_bytes = raw.to_csv(index=False).encode("utf-8")

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                raw.to_excel(writer, index=False, sheet_name="Similar Properties")
            buf.seek(0)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download CSV", data=csv_bytes,
                file_name=f"similar_{postal_input.strip()}.csv",
                mime="text/csv"
            )
            dl2.download_button(
                "Download Excel", data=buf,
                file_name=f"similar_{postal_input.strip()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
