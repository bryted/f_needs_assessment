from __future__ import annotations

from io import BytesIO
from itertools import product
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ACTIONS_FILE = OUTPUT_DIR / "child_visits_actions_01_resolved_other_v3.csv"

GENDER_ORDER = ["Male", "Female", "Unknown"]
GENDER_COLORS = {
    "Male": "#1f77b4",
    "Female": "#e45756",
    "Unknown": "#7f7f7f",
}


def _format_recommendation(col_name: str, group: str) -> str:
    prefix = f"{group}__"
    code = col_name[len(prefix) :] if col_name.startswith(prefix) else col_name
    return code.replace("_", " ").strip().title()


def _recommendation_columns(df: pd.DataFrame, group: str) -> list[str]:
    prefix = f"{group}__"
    return [c for c in df.columns if c.startswith(prefix) and not c.endswith("_flag")]


def _format_token(token: str) -> str:
    return str(token).replace("_", " ").strip().title()


def _shorten_sankey_label(label: str, max_len: int = 30) -> str:
    text = str(label).strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3].rstrip()}..."


def _extract_group_tokens(row: pd.Series, columns: list[str], group: str) -> list[str]:
    prefix = f"{group}__"
    tokens: list[str] = []
    for col in columns:
        if row[col] <= 0:
            continue
        token = col[len(prefix) :] if col.startswith(prefix) else col
        tokens.append(token)
    return sorted(set(tokens))


def _normalize_gender(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.replace({"male": "Male", "female": "Female", "": "Unknown"}).mask(
        ~text.isin({"male", "female", ""}), "Unknown"
    )


def _compute_unique_children_metric(df: pd.DataFrame) -> tuple[str, str]:
    if "visit_id" in df.columns:
        return "Unique children (visit_id)", f"{df['visit_id'].nunique():,}"
    if {"child_code", "household_id"}.issubset(df.columns):
        pairs = df[["child_code", "household_id"]].dropna().drop_duplicates()
        return "Unique children (child+household)", f"{len(pairs):,}"
    if "child_code" in df.columns:
        return "Unique children", f"{df['child_code'].nunique():,}"
    if "household_id" in df.columns:
        return "Unique children (household proxy)", f"{df['household_id'].nunique():,}"
    return "Unique children", "N/A"


@st.cache_data(show_spinner=False)
def load_actions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def summarize_recommendations(
    df: pd.DataFrame,
    group: str,
    dimensions: list[str] | None = None,
    add_share: bool = False,
) -> pd.DataFrame:
    dimensions = dimensions or []
    base_cols = dimensions + ["Recommendation"] + GENDER_ORDER + ["Total"]
    if add_share:
        base_cols += ["% of filtered children"]

    if df.empty:
        return pd.DataFrame(columns=base_cols)

    rec_cols = _recommendation_columns(df, group)
    if not rec_cols:
        return pd.DataFrame(columns=base_cols)

    missing_dims = [d for d in dimensions if d not in df.columns]
    if missing_dims:
        return pd.DataFrame(columns=base_cols)

    work = df.copy()
    if "child_gender" in work.columns:
        work["Gender"] = _normalize_gender(work["child_gender"])
    else:
        work["Gender"] = "Unknown"

    for dim in dimensions:
        work[dim] = work[dim].fillna("").astype(str).str.strip().replace({"": "Unknown"})

    indicators = work[rec_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    event_parts: list[pd.DataFrame] = []
    keep_cols = dimensions + ["Gender"]

    for rec_col in rec_cols:
        mask = indicators[rec_col] > 0
        if not mask.any():
            continue
        part = work.loc[mask, keep_cols].copy()
        part["Recommendation"] = _format_recommendation(rec_col, group)
        event_parts.append(part)

    if not event_parts:
        return pd.DataFrame(columns=base_cols)

    events = pd.concat(event_parts, ignore_index=True)
    grouped = (
        events.groupby(dimensions + ["Recommendation", "Gender"], dropna=False)
        .size()
        .reset_index(name="Count")
    )
    pivot = (
        grouped.pivot_table(
            index=dimensions + ["Recommendation"],
            columns="Gender",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for gender in GENDER_ORDER:
        if gender not in pivot.columns:
            pivot[gender] = 0

    pivot[GENDER_ORDER] = pivot[GENDER_ORDER].astype(int)
    pivot["Total"] = pivot[GENDER_ORDER].sum(axis=1).astype(int)
    pivot = pivot[pivot["Total"] > 0].copy()

    if dimensions:
        pivot = pivot.sort_values(
            dimensions + ["Total", "Recommendation"],
            ascending=[True] * len(dimensions) + [False, True],
        )
    else:
        pivot = pivot.sort_values(["Total", "Recommendation"], ascending=[False, True])

    if add_share:
        if "visit_id" in work.columns:
            denom = max(int(work["visit_id"].nunique()), 1)
        else:
            denom = max(len(work), 1)
        pivot["% of filtered children"] = (pivot["Total"] / denom * 100).round(1)

    return pivot[base_cols].reset_index(drop=True)


def render_gender_stacked_chart(summary: pd.DataFrame, title: str) -> None:
    if summary.empty:
        st.info("No recommendation signals found for current filters.")
        return

    chart_base = summary[["Recommendation"] + GENDER_ORDER + ["Total"]].copy()
    chart_base = chart_base.sort_values(["Total", "Recommendation"], ascending=[False, True]).reset_index(drop=True)
    recommendation_order = chart_base["Recommendation"].tolist()

    long_df = chart_base.melt(
        id_vars=["Recommendation", "Total"],
        value_vars=GENDER_ORDER,
        var_name="Gender",
        value_name="Count",
    )
    long_df = long_df[long_df["Count"] > 0].copy()
    if long_df.empty:
        st.info("No recommendation signals found for current filters.")
        return

    fig = px.bar(
        long_df,
        x="Count",
        y="Recommendation",
        color="Gender",
        orientation="h",
        title=title,
        color_discrete_map=GENDER_COLORS,
        category_orders={"Gender": GENDER_ORDER},
    )
    for trace in fig.data:
        if trace.name in {"Male", "Female"}:
            trace.text = trace.x
            trace.texttemplate = "%{text:.0f}"
            trace.textposition = "auto"
            trace.textfont = dict(size=11, color="white")
            trace.insidetextfont = dict(size=11, color="white")
            trace.outsidetextfont = dict(size=11, color="#111827")
        else:
            trace.text = None
    fig.update_layout(
        barmode="stack",
        height=max(420, 26 * len(recommendation_order) + 140),
        margin=dict(l=20, r=20, t=55, b=30),
        legend_title_text="Gender",
        uniformtext=dict(minsize=10, mode="hide"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title="Count", rangemode="tozero")
    fig.update_yaxes(
        title="",
        categoryorder="array",
        categoryarray=list(reversed(recommendation_order)),
    )
    st.plotly_chart(fig, width="stretch")


def summarize_combinations(df: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "Child Recommendations",
        "Household Recommendations",
        "Community Recommendations",
        "Count",
    ]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    child_cols = _recommendation_columns(df, "child")
    household_cols = _recommendation_columns(df, "household")
    community_cols = _recommendation_columns(df, "community")
    if not child_cols or not household_cols or not community_cols:
        return pd.DataFrame(columns=out_cols)

    cols = child_cols + household_cols + community_cols
    work = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    records: list[tuple[str, str, str]] = []
    for _, row in work.iterrows():
        child_tokens = _extract_group_tokens(row, child_cols, "child")
        household_tokens = _extract_group_tokens(row, household_cols, "household")
        community_tokens = _extract_group_tokens(row, community_cols, "community")
        if not child_tokens or not household_tokens or not community_tokens:
            continue
        records.extend(product(child_tokens, household_tokens, community_tokens))

    if not records:
        return pd.DataFrame(columns=out_cols)

    out = (
        pd.DataFrame(records, columns=["child_token", "household_token", "community_token"])
        .groupby(["child_token", "household_token", "community_token"], dropna=False)
        .size()
        .reset_index(name="Count")
    )
    out["Child Recommendations"] = out["child_token"].map(_format_token)
    out["Household Recommendations"] = out["household_token"].map(_format_token)
    out["Community Recommendations"] = out["community_token"].map(_format_token)
    out["Count"] = out["Count"].astype(int)
    out = out.sort_values(
        [
            "Count",
            "Child Recommendations",
            "Household Recommendations",
            "Community Recommendations",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return out[out_cols]


def build_sankey_from_combinations(combinations: pd.DataFrame, top_n: int) -> go.Figure | None:
    if combinations.empty:
        return None

    combos_top = combinations.head(top_n).copy()
    if combos_top.empty:
        return None

    stage_specs = [
        ("child", "Child Recommendations", "Child", 0.02, "#dbeafe"),
        ("household", "Household Recommendations", "Household", 0.50, "#ffedd5"),
        ("community", "Community Recommendations", "Community", 0.98, "#dcfce7"),
    ]

    for stage_key, rec_col, _, _, _ in stage_specs:
        cleaned = combos_top[rec_col].fillna("").astype(str).str.strip()
        cleaned = cleaned.where(cleaned != "", "Unknown")
        combos_top[f"{stage_key}_full_label"] = cleaned
        combos_top[f"{stage_key}_id"] = f"{stage_key}::" + cleaned

    child_to_household = (
        combos_top.groupby(["child_id", "household_id"], dropna=False)["Count"]
        .sum()
        .reset_index()
        .rename(columns={"child_id": "source", "household_id": "target"})
    )
    household_to_community = (
        combos_top.groupby(["household_id", "community_id"], dropna=False)["Count"]
        .sum()
        .reset_index()
        .rename(columns={"household_id": "source", "community_id": "target"})
    )
    links = (
        pd.concat([child_to_household, household_to_community], ignore_index=True)
        .groupby(["source", "target"], dropna=False)["Count"]
        .sum()
        .reset_index()
    )
    if links.empty:
        return None

    node_meta: dict[str, dict[str, str]] = {}
    stage_nodes: dict[str, list[str]] = {}
    stage_x: dict[str, float] = {}
    stage_color: dict[str, str] = {}
    for stage_key, _, stage_name, x_pos, color in stage_specs:
        id_col = f"{stage_key}_id"
        label_col = f"{stage_key}_full_label"
        seen = (
            combos_top[[id_col, label_col]]
            .drop_duplicates()
            .rename(columns={id_col: "node_id", label_col: "full_label"})
        )
        stage_counts = (
            combos_top.groupby(id_col, dropna=False)["Count"]
            .sum()
            .reset_index(name="stage_total")
            .rename(columns={id_col: "node_id"})
        )
        ordered = seen.merge(stage_counts, on="node_id", how="left").fillna({"stage_total": 0})
        ordered = ordered.sort_values(["stage_total", "full_label"], ascending=[False, True]).reset_index(drop=True)
        node_ids = ordered["node_id"].tolist()
        stage_nodes[stage_name] = node_ids
        stage_x[stage_name] = x_pos
        stage_color[stage_name] = color
        for _, item in ordered.iterrows():
            node_meta[item["node_id"]] = {
                "stage": stage_name,
                "full_label": str(item["full_label"]),
                "display_label": _shorten_sankey_label(item["full_label"], max_len=30),
            }

    node_ids_ordered = (
        stage_nodes["Child"] + stage_nodes["Household"] + stage_nodes["Community"]
    )
    if not node_ids_ordered:
        return None
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids_ordered)}

    node_x: list[float] = []
    node_y: list[float] = []
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_customdata: list[list[str]] = []

    for stage_name in ("Child", "Household", "Community"):
        ids = stage_nodes.get(stage_name, [])
        count = len(ids)
        if count == 1:
            y_positions = [0.5]
        elif count > 1:
            top = 0.05
            bottom = 0.95
            step = (bottom - top) / (count - 1)
            y_positions = [top + i * step for i in range(count)]
        else:
            y_positions = []

        for node_id, y_pos in zip(ids, y_positions):
            meta = node_meta[node_id]
            node_x.append(stage_x[stage_name])
            node_y.append(y_pos)
            node_labels.append(meta["display_label"])
            node_colors.append(stage_color[stage_name])
            node_customdata.append([meta["stage"], meta["full_label"]])

    links = links[links["source"].isin(node_index) & links["target"].isin(node_index)].copy()
    if links.empty:
        return None

    links["source_stage"] = links["source"].map(lambda x: node_meta[x]["stage"])
    links["source_label"] = links["source"].map(lambda x: node_meta[x]["full_label"])
    links["target_stage"] = links["target"].map(lambda x: node_meta[x]["stage"])
    links["target_label"] = links["target"].map(lambda x: node_meta[x]["full_label"])
    link_customdata = links[["source_stage", "source_label", "target_stage", "target_label"]].values

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                textfont=dict(
                    family="Segoe UI, Arial, sans-serif",
                    size=15,
                    color="#0f172a",
                ),
                node=dict(
                    pad=16,
                    thickness=20,
                    line=dict(color="rgba(17,24,39,0.35)", width=0.8),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    customdata=node_customdata,
                    hovertemplate=(
                        "Stage: %{customdata[0]}<br>"
                        "Recommendation: %{customdata[1]}<extra></extra>"
                    ),
                ),
                link=dict(
                    source=links["source"].map(node_index).tolist(),
                    target=links["target"].map(node_index).tolist(),
                    value=links["Count"].astype(float).tolist(),
                    color="rgba(71,85,105,0.35)",
                    customdata=link_customdata,
                    hovertemplate=(
                        "From: %{customdata[0]} - %{customdata[1]}<br>"
                        "To: %{customdata[2]} - %{customdata[3]}<br>"
                        "Count: %{value:,.0f}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Top {min(top_n, len(combos_top))} Recommendation Flow (Child -> Household -> Community)",
        font=dict(family="Segoe UI, Arial, sans-serif", size=14, color="#0f172a"),
        title_font=dict(family="Segoe UI Semibold, Segoe UI, Arial, sans-serif", size=24, color="#0f172a"),
        title_x=0.5,
        title_y=0.98,
        title_xanchor="center",
        title_yanchor="top",
        margin=dict(l=20, r=20, t=120, b=20),
        height=max(650, 30 * len(node_ids_ordered)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#94a3b8",
            font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#0f172a"),
        ),
        annotations=[
            dict(
                x=0.02,
                y=1.03,
                xref="paper",
                yref="paper",
                text="<b>Child</b>",
                showarrow=False,
                xanchor="center",
                font=dict(
                    family="Segoe UI Semibold, Segoe UI, Arial, sans-serif",
                    size=17,
                    color="#0f172a",
                ),
            ),
            dict(
                x=0.50,
                y=1.03,
                xref="paper",
                yref="paper",
                text="<b>Household</b>",
                showarrow=False,
                xanchor="center",
                font=dict(
                    family="Segoe UI Semibold, Segoe UI, Arial, sans-serif",
                    size=17,
                    color="#0f172a",
                ),
            ),
            dict(
                x=0.98,
                y=1.03,
                xref="paper",
                yref="paper",
                text="<b>Community</b>",
                showarrow=False,
                xanchor="center",
                font=dict(
                    family="Segoe UI Semibold, Segoe UI, Arial, sans-serif",
                    size=17,
                    color="#0f172a",
                ),
            ),
        ],
    )
    return fig


def build_decision_pack_excel(df: pd.DataFrame) -> bytes:
    general_child = summarize_recommendations(df, "child", dimensions=[], add_share=True)
    general_household = summarize_recommendations(df, "household", dimensions=[], add_share=True)
    general_community = summarize_recommendations(df, "community", dimensions=[], add_share=True)

    coop_child = summarize_recommendations(df, "child", dimensions=["cooperative"], add_share=False)
    coop_household = summarize_recommendations(df, "household", dimensions=["cooperative"], add_share=False)
    coop_community = summarize_recommendations(df, "community", dimensions=["cooperative"], add_share=False)

    community_child = summarize_recommendations(df, "child", dimensions=["community"], add_share=False)
    community_household = summarize_recommendations(df, "household", dimensions=["community"], add_share=False)
    community_community = summarize_recommendations(df, "community", dimensions=["community"], add_share=False)

    rename_map = {"cooperative": "Cooperative", "community": "Community"}
    sheets = {
        "General_Child": general_child,
        "General_Household": general_household,
        "General_Community": general_community,
        "Coop_Child": coop_child.rename(columns=rename_map),
        "Coop_Household": coop_household.rename(columns=rename_map),
        "Coop_Community": coop_community.rename(columns=rename_map),
        "Community_Child": community_child.rename(columns=rename_map),
        "Community_Household": community_household.rename(columns=rename_map),
        "Community_Community": community_community.rename(columns=rename_map),
    }

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.getvalue()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    cooperative_options = (
        sorted(df["cooperative"].dropna().astype(str).unique().tolist())
        if "cooperative" in df.columns
        else []
    )
    community_options = (
        sorted(df["community"].dropna().astype(str).unique().tolist())
        if "community" in df.columns
        else []
    )

    gender_options: list[str] = []
    if "child_gender" in df.columns:
        gender_series = _normalize_gender(df["child_gender"])
        gender_options = [g for g in GENDER_ORDER if g in set(gender_series.tolist())]

    selected_coops = st.sidebar.multiselect("Cooperative", cooperative_options, default=cooperative_options)
    selected_communities = st.sidebar.multiselect("Community", community_options, default=community_options)
    if gender_options:
        selected_genders = st.sidebar.multiselect("Child gender", gender_options, default=gender_options)
    else:
        selected_genders = None

    out = df.copy()
    if selected_coops and "cooperative" in out.columns:
        out = out[out["cooperative"].astype(str).isin(selected_coops)]
    if selected_communities and "community" in out.columns:
        out = out[out["community"].astype(str).isin(selected_communities)]
    if selected_genders is not None:
        if selected_genders:
            out = out[_normalize_gender(out["child_gender"]).isin(selected_genders)]
        else:
            out = out.iloc[0:0]

    return out


def main() -> None:
    st.set_page_config(page_title="Feastables Recommendations Dashboard", layout="wide")
    st.title("Feastables Needs Assessment Dashboard")
    st.caption("Gender-split recommendation visuals for decision making")

    if not ACTIONS_FILE.exists():
        st.error(f"Missing input file: {ACTIONS_FILE}")
        st.info("Run `python feastables.py` first to generate outputs.")
        st.stop()

    df = load_actions(ACTIONS_FILE)
    filtered = apply_filters(df)
    st.sidebar.header("Display")
    top_n = st.sidebar.slider("Top recommendations / flows", min_value=5, max_value=50, value=15, step=1)

    extract_bytes = build_decision_pack_excel(filtered)
    st.download_button(
        label="Download summarized Excel extract",
        data=extract_bytes,
        file_name="dashboard_summarized_decision_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    child_metric_label, child_metric_value = _compute_unique_children_metric(filtered)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric(child_metric_label, child_metric_value)
    metric2.metric(
        "Unique households",
        f"{filtered['household_id'].nunique():,}" if "household_id" in filtered.columns else "N/A",
    )
    metric3.metric(
        "Unique communities",
        f"{filtered['community'].nunique():,}" if "community" in filtered.columns else "N/A",
    )

    if filtered.empty:
        st.warning("No records match the selected filters. The Excel extract contains empty structured sheets.")
        st.stop()

    child_summary = summarize_recommendations(filtered, "child", dimensions=[], add_share=False).head(top_n)
    household_summary = summarize_recommendations(filtered, "household", dimensions=[], add_share=False).head(top_n)
    community_summary = summarize_recommendations(filtered, "community", dimensions=[], add_share=False).head(top_n)

    st.subheader("Child Recommendations by Gender")
    render_gender_stacked_chart(child_summary, "Child Recommendations (Descending)")

    st.subheader("Household Recommendations by Gender")
    render_gender_stacked_chart(household_summary, "Household Recommendations (Descending)")

    st.subheader("Community Recommendations by Gender")
    render_gender_stacked_chart(community_summary, "Community Recommendations (Descending)")

    combinations = summarize_combinations(filtered)
    st.subheader("Recommendation Flow (Child -> Household -> Community)")
    sankey_fig = build_sankey_from_combinations(combinations, top_n=top_n)
    if sankey_fig is None:
        st.info("No recommendation flow available for current filters.")
    else:
        st.plotly_chart(sankey_fig, width="stretch")


if __name__ == "__main__":
    main()
