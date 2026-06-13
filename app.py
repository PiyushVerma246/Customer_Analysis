"""
Smart Retail Customer Behavior Analytics Platform
Flask + Bootstrap 5 Application
=================================================
Routes:
  GET  /                   → Dashboard
  GET/POST /insights        → Customer Insights (POST: look up a customer)
  GET/POST /recommendations → Product Recommendations (POST: filter by product)
  GET  /analytics           → Analytics (8 Plotly charts)
"""

import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request

from nba_engine import NBAEngine

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Design constants ───────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "VIP":     "#a855f7",
    "Regular": "#10b981",
    "At Risk": "#ef4444",
}

# Base Plotly dark-theme layout (no xaxis/yaxis/legend — added per chart)
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#64748b", size=12),
    margin=dict(l=10, r=10, t=52, b=10),
    title_font=dict(color="#e2e8f0", size=14, family="Inter, sans-serif"),
    hoverlabel=dict(
        bgcolor="rgba(8,14,26,0.96)",
        bordercolor="rgba(99,102,241,0.45)",
        font=dict(family="Inter, sans-serif", color="#f8fafc", size=12),
    ),
)

_AXIS = dict(
    gridcolor="rgba(255,255,255,0.04)",
    linecolor="rgba(255,255,255,0.06)",
    zerolinecolor="rgba(255,255,255,0.04)",
    tickfont=dict(color="#64748b"),
)

_LEGEND = dict(
    bgcolor="rgba(8,14,26,0.88)",
    bordercolor="rgba(255,255,255,0.07)",
    borderwidth=1,
    font=dict(color="#94a3b8", size=11),
)


def _layout(**extra):
    """Merge base layout with per-chart overrides."""
    base = dict(**_PLOTLY_BASE)
    base["xaxis"] = dict(**_AXIS)
    base["yaxis"] = dict(**_AXIS)
    base["legend"] = dict(**_LEGEND)
    base.update(extra)
    return base


def _layout_no_xy(**extra):
    """Layout for pie/donut/polar charts (no axes)."""
    base = dict(**_PLOTLY_BASE)
    base["legend"] = dict(**_LEGEND)
    base.update(extra)
    return base


# ── Data loading ───────────────────────────────────────────────────────────

def _clean_frozenset(val) -> str:
    if pd.isna(val):
        return ""
    val = str(val)
    val = re.sub(r"frozenset\(\{?", "", val)
    val = re.sub(r"\}?\)", "", val)
    val = val.replace('"', "").replace("'", "").strip(", ").strip()
    return val


def _load_data():
    try:
        df_c = pd.read_csv("customer_segments.csv")
        df_c.columns = [c.strip() for c in df_c.columns]
        df_c["CustomerID"] = pd.to_numeric(df_c["CustomerID"], errors="coerce")
        df_c.dropna(subset=["CustomerID"], inplace=True)
        df_c["CustomerID"] = df_c["CustomerID"].astype(int)
    except FileNotFoundError:
        df_c = pd.DataFrame()

    try:
        df_r = pd.read_csv("recommendation_rules.csv")
        df_r.columns = [c.strip() for c in df_r.columns]
        df_r["antecedents_clean"] = df_r["antecedents"].apply(_clean_frozenset)
        df_r["consequents_clean"] = df_r["consequents"].apply(_clean_frozenset)
        df_r = df_r[df_r["antecedents_clean"].str.len() > 0].reset_index(drop=True)
    except FileNotFoundError:
        df_r = pd.DataFrame()

    try:
        model = joblib.load("kmeans_model.pkl")
    except Exception:
        model = None

    return df_c, df_r, model


# Load once at startup
DF_CUSTOMERS, DF_RULES, KMEANS_MODEL = _load_data()
NBA_ENGINE = NBAEngine(DF_CUSTOMERS, DF_RULES)


# ── Utilities ──────────────────────────────────────────────────────────────

def to_json(fig) -> str:
    """Serialise a Plotly figure to JSON for the template."""
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def fmt_gbp(v: float) -> str:
    return f"£{v:,.2f}"


_INTELLIGENCE = {
    "VIP": {
        "icon": "bi-gem",
        "icon_color": "#c084fc",
        "color": "#c084fc",
        "bg": "rgba(192,132,252,0.1)",
        "border": "rgba(192,132,252,0.3)",
        "title": "VIP Customer — High Value",
        "body": (
            "This customer ranks among your top performers with high purchase frequency "
            "and significant lifetime value. Prioritise them with exclusive loyalty tiers, "
            "early-access launches, VIP events, and personalised account management. "
            "Small improvements in VIP retention yield outsized revenue returns."
        ),
    },
    "Regular": {
        "icon": "bi-cart3",
        "icon_color": "#34d399",
        "color": "#34d399",
        "bg": "rgba(52,211,153,0.1)",
        "border": "rgba(52,211,153,0.3)",
        "title": "Regular Customer — Steady Engagement",
        "body": (
            "An active, reliable buyer who consistently engages with your brand. "
            "Cross-sell complementary products, run frequency-based incentive programs, "
            "and send personalised promotions to motivate this customer toward VIP status. "
            "Targeted communications can significantly increase basket size."
        ),
    },
    "At Risk": {
        "icon": "bi-exclamation-triangle",
        "icon_color": "#fb7185",
        "color": "#fb7185",
        "bg": "rgba(251,113,133,0.1)",
        "border": "rgba(251,113,133,0.3)",
        "title": "At-Risk Customer — Needs Intervention",
        "body": (
            "This customer's purchase recency is high — they may be disengaging or churning. "
            "Deploy a win-back campaign immediately: time-limited discount, free shipping, "
            "or a personalised product recommendation. Early intervention dramatically "
            "increases re-activation probability before churn becomes permanent."
        ),
    },
}


# ── Context processor — sidebar stats available on every page ──────────────
@app.context_processor
def inject_sidebar_stats():
    if DF_CUSTOMERS.empty:
        return dict(sb_total="—", sb_vip="—", sb_regular="—",
                    sb_atrisk="—", sb_rules="—")
    df = DF_CUSTOMERS
    return dict(
        sb_total=f"{len(df):,}",
        sb_vip=f"{len(df[df['Segment'] == 'VIP']):,}",
        sb_regular=f"{len(df[df['Segment'] == 'Regular']):,}",
        sb_atrisk=f"{len(df[df['Segment'] == 'At Risk']):,}",
        sb_rules=f"{len(DF_RULES):,}" if not DF_RULES.empty else "0",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  0 — LANDING HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("home.html")


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/dashboard")
def dashboard():
    df = DF_CUSTOMERS
    if df.empty:
        return render_template("dashboard.html", active_page="dashboard", error=True)

    total      = len(df)
    vip_count  = len(df[df["Segment"] == "VIP"])
    reg_count  = len(df[df["Segment"] == "Regular"])
    risk_count = len(df[df["Segment"] == "At Risk"])

    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    # ── Donut chart ────────────────────────────────────────────────────────
    fig_donut = go.Figure(go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["Count"],
        hole=0.62,
        marker=dict(
            colors=[SEGMENT_COLORS.get(s, "#6366f1") for s in seg_counts["Segment"]],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textinfo="label+percent",
        textfont=dict(color="#f8fafc", size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,} customers · %{percent}<extra></extra>",
        pull=[0.05 if s == "VIP" else 0 for s in seg_counts["Segment"]],
    ))
    fig_donut.add_annotation(
        text=f"<b>{total:,}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=18, color="#f8fafc", family="Inter"),
        showarrow=False,
    )
    fig_donut.update_layout(
        **_layout_no_xy(
            height=360,
            showlegend=True,
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                        **{k: v for k, v in _LEGEND.items() if k != "orientation"}),
        )
    )

    # ── Revenue bar chart ──────────────────────────────────────────────────
    rev_df = df.groupby("Segment")["Monetary"].sum().reset_index()
    rev_df.columns = ["Segment", "Revenue"]
    rev_df = rev_df.sort_values("Revenue", ascending=False)
    fig_rev = px.bar(
        rev_df, x="Segment", y="Revenue",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title="Total Revenue by Customer Segment",
        text=rev_df["Revenue"].map(lambda v: f"£{v:,.0f}"),
    )
    fig_rev.update_traces(textposition="outside", marker_line_width=0, width=0.45)
    fig_rev.update_layout(**_layout(height=300, showlegend=False, yaxis_title="Revenue (£)"))

    # ── Summary statistics ─────────────────────────────────────────────────
    summary = (
        df.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(2)
        .reset_index()
    )

    # Segment breakdown data for progress bars
    seg_breakdown = [
        {
            "segment": row["Segment"],
            "count": int(row["Count"]),
            "pct": round(row["Count"] / total * 100, 1),
            "color": SEGMENT_COLORS.get(row["Segment"], "#6366f1"),
        }
        for _, row in seg_counts.iterrows()
    ]

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        total=f"{total:,}",
        vip_count=f"{vip_count:,}",
        reg_count=f"{reg_count:,}",
        risk_count=f"{risk_count:,}",
        vip_pct=f"{vip_count/total*100:.1f}",
        reg_pct=f"{reg_count/total*100:.1f}",
        risk_pct=f"{risk_count/total*100:.1f}",
        avg_ltv=fmt_gbp(df["Monetary"].mean()),
        total_rev=fmt_gbp(df["Monetary"].sum()),
        seg_breakdown=seg_breakdown,
        chart_donut=to_json(fig_donut),
        chart_rev=to_json(fig_rev),
        summary=summary.to_dict("records"),
        is_high_risk=risk_count / total > 0.30,
        risk_pct_str=f"{risk_count/total:.0%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  2 — CUSTOMER INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/insights", methods=["GET", "POST"])
def insights():
    df = DF_CUSTOMERS
    customer = None
    form_error = None
    form_warning = None
    cid_value = ""

    sample = (
        df.sample(min(12, len(df)), random_state=99)
        [["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]]
        .sort_values("Segment")
        .to_dict("records")
        if not df.empty else []
    )
    id_range = {
        "min": int(df["CustomerID"].min()) if not df.empty else 0,
        "max": int(df["CustomerID"].max()) if not df.empty else 0,
        "total": len(df),
    }

    if request.method == "POST":
        raw = request.form.get("customer_id", "").strip()
        cid_value = raw
        if not raw:
            form_error = "Please enter a CustomerID."
        else:
            try:
                cid = int(float(raw))
                row_df = df[df["CustomerID"] == cid]
                if row_df.empty:
                    form_warning = f"Customer ID {cid:,} was not found. Try a different ID."
                else:
                    r   = row_df.iloc[0]
                    seg = r["Segment"]
                    clr = SEGMENT_COLORS.get(seg, "#6366f1")

                    r_pct = 1.0 - (r["Recency"]  / df["Recency"].max())
                    f_pct =        r["Frequency"] / df["Frequency"].max()
                    m_pct =        r["Monetary"]  / df["Monetary"].max()
                    pct_rank = (df["Monetary"] < r["Monetary"]).mean()

                    # Radar chart
                    cats = ["Recency", "Frequency", "Monetary", "Recency"]
                    vals = [round(r_pct, 3), round(f_pct, 3), round(m_pct, 3), round(r_pct, 3)]
                    # Convert hex color to rgba for fillcolor (8-digit hex not supported by Plotly)
                    def _hex_to_rgba(hex_color, alpha=0.12):
                        h = hex_color.lstrip('#')
                        r_c, g_c, b_c = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        return f"rgba({r_c},{g_c},{b_c},{alpha})"

                    fig_radar = go.Figure(go.Scatterpolar(
                        r=vals, theta=cats, fill="toself",
                        fillcolor=_hex_to_rgba(clr, 0.12),
                        line=dict(color=clr, width=2.5),
                        marker=dict(size=8, color=clr),
                        hovertemplate="%{theta}: <b>%{r:.0%}</b><extra></extra>",
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(
                                visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.05)",
                                linecolor="rgba(255,255,255,0.05)",
                                tickfont=dict(color="#475569", size=9),
                                tickformat=".0%",
                            ),
                            angularaxis=dict(
                                gridcolor="rgba(255,255,255,0.05)",
                                linecolor="rgba(255,255,255,0.05)",
                                tickfont=dict(color="#94a3b8", size=12, family="Inter"),
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#94a3b8"),
                        height=310, showlegend=False,
                        margin=dict(l=40, r=40, t=20, b=20),
                    )

                    customer = {
                        "id":         cid,
                        "segment":    seg,
                        "color":      clr,
                        "icon":       {"VIP": "bi-gem", "Regular": "bi-cart3",
                                       "At Risk": "bi-exclamation-triangle"}.get(seg, "bi-person"),
                        "seg_class":  {"VIP": "seg-vip", "Regular": "seg-reg",
                                       "At Risk": "seg-risk"}.get(seg, "seg-reg"),
                        "recency":    int(r["Recency"]),
                        "frequency":  int(r["Frequency"]),
                        "monetary":   fmt_gbp(r["Monetary"]),
                        "r_pct":      round(r_pct * 100, 1),
                        "f_pct":      round(f_pct * 100, 1),
                        "m_pct":      round(m_pct * 100, 1),
                        "top_pct":    f"{100 - pct_rank*100:.0f}",
                        "avg_recency":  f"{df['Recency'].mean():.0f}",
                        "avg_frequency": f"{df['Frequency'].mean():.1f}",
                        "avg_monetary":  fmt_gbp(df["Monetary"].mean()),
                        "chart_radar":   to_json(fig_radar),
                        "intel":         _INTELLIGENCE.get(seg, _INTELLIGENCE["Regular"]),
                    }
            except (ValueError, TypeError):
                form_error = "Please enter a valid numeric CustomerID."

    return render_template(
        "insights.html",
        active_page="insights",
        customer=customer,
        form_error=form_error,
        form_warning=form_warning,
        cid_value=cid_value,
        sample=sample,
        id_range=id_range,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  3 — PRODUCT RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    df = DF_RULES
    all_products = sorted(
        [p for p in df["antecedents_clean"].dropna().unique() if p]
    ) if not df.empty else []

    selected      = None
    rec_cards     = []
    chart_lift    = None
    stats         = {}
    rec_warning   = None

    overview = {}
    if not df.empty:
        overview = {
            "total": len(df),
            "max_lift": f"{df['lift'].max():.2f}",
            "avg_conf": f"{df['confidence'].mean():.1%}",
        }

    if request.method == "POST":
        selected = request.form.get("product", "").strip()
        if selected:
            matching = (
                df[df["antecedents_clean"] == selected]
                .copy()
                .sort_values("lift", ascending=False)
                .reset_index(drop=True)
            )
            if matching.empty:
                rec_warning = f"No recommendations found for '{selected}'. Try a different product."
            else:
                stats = {
                    "count":    len(matching),
                    "max_lift": f"{matching['lift'].max():.2f}",
                    "avg_conf": f"{matching['confidence'].mean():.1%}",
                }
                rec_cards = [
                    {
                        "rank":     i + 1,
                        "product":  row["consequents_clean"],
                        "lift":     f"{row['lift']:.2f}",
                        "conf":     f"{row['confidence']:.1%}",
                        "conf_pct": round(row["confidence"] * 100, 1),
                        "support":  f"{row['support']:.4f}",
                    }
                    for i, (_, row) in enumerate(matching.iterrows())
                ]

                cd = matching.head(15).copy()
                cd["label"] = cd["consequents_clean"].apply(
                    lambda x: (x[:44] + "…") if len(x) > 44 else x
                )
                fig_lift = px.bar(
                    cd, x="lift", y="label", orientation="h",
                    color="confidence",
                    color_continuous_scale=["#3b82f6", "#6366f1", "#a855f7"],
                    title=f"Lift Scores — top recs for: {selected[:40]}",
                    labels={"lift": "Lift Score", "label": "Product", "confidence": "Confidence"},
                    text=cd["lift"].map("{:.2f}×".format),
                )
                fig_lift.update_traces(textposition="outside", marker_line_width=0)
                fig_lift.update_layout(
                    **_layout(
                        height=max(300, 50 * len(cd)),
                        yaxis=dict(**_AXIS, autorange="reversed"),
                        coloraxis_colorbar=dict(title="Conf.", tickformat=".0%", len=0.55,
                                                tickfont=dict(color="#94a3b8"),
                                                title_font=dict(color="#94a3b8")),
                    )
                )
                chart_lift = to_json(fig_lift)

    return render_template(
        "recommendations.html",
        active_page="recommendations",
        all_products=all_products,
        selected=selected,
        rec_cards=rec_cards,
        chart_lift=chart_lift,
        stats=stats,
        overview=overview,
        rec_warning=rec_warning,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  4 — NEXT BEST ACTION
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/nba", methods=["GET", "POST"])
def nba_route():
    customer_id = None
    customer_details = None
    actions = []
    error = None

    if request.method == "POST":
        raw_id = request.form.get("customer_id", "").strip()
        if not raw_id:
            error = "Please enter a Customer ID."
        else:
            try:
                customer_id = int(float(raw_id))
                row_df = DF_CUSTOMERS[DF_CUSTOMERS["CustomerID"] == customer_id]
                
                if row_df.empty:
                    error = f"Customer ID {customer_id} not found."
                else:
                    cust = row_df.iloc[0]
                    customer_details = {
                        "id": customer_id,
                        "segment": cust["Segment"],
                        "recency": int(cust["Recency"]),
                        "frequency": int(cust["Frequency"]),
                        "monetary": fmt_gbp(cust["Monetary"])
                    }
                    actions = NBA_ENGINE.get_actions(customer_id)
            except ValueError:
                error = "Invalid Customer ID format."

    return render_template(
        "nba.html",
        active_page="nba",
        customer_id=customer_id,
        customer=customer_details,
        actions=actions,
        error=error
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTE  5 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/analytics")
def analytics():
    df = DF_CUSTOMERS
    if df.empty:
        return render_template("analytics.html", active_page="analytics", error=True)

    # 1. Segment pie
    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_pie = px.pie(
        seg_counts, values="Count", names="Segment",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title="Customer Segment Distribution", hole=0.48,
    )
    fig_pie.update_traces(
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
        pull=[0.04 if s == "At Risk" else 0 for s in seg_counts["Segment"]],
    )
    fig_pie.update_layout(**_layout_no_xy(height=400))

    # 2. Top 10 customers
    top10 = df.nlargest(10, "Monetary")[["CustomerID", "Monetary", "Segment"]].copy()
    top10["CustomerID"] = top10["CustomerID"].astype(str)
    top10 = top10.sort_values("Monetary", ascending=True)
    fig_top = px.bar(
        top10, x="Monetary", y="CustomerID", orientation="h",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title="Top 10 Customers by Lifetime Value",
        text=top10["Monetary"].map("£{:,.0f}".format),
    )
    fig_top.update_traces(textposition="outside", marker_line_width=0)
    fig_top.update_layout(**_layout(height=400))

    # 3. Recency histogram
    fig_rec = px.histogram(
        df, x="Recency", color="Segment", color_discrete_map=SEGMENT_COLORS,
        nbins=40, title="Recency Distribution (days since last purchase)",
        barmode="overlay", opacity=0.78,
    )
    fig_rec.update_layout(**_layout(height=360, bargap=0.03))

    # 4. Frequency histogram
    freq_cap = df["Frequency"].quantile(0.99)
    fig_frq = px.histogram(
        df[df["Frequency"] <= freq_cap],
        x="Frequency", color="Segment", color_discrete_map=SEGMENT_COLORS,
        nbins=35, title="Frequency Distribution (capped at 99th percentile)",
        barmode="overlay", opacity=0.78,
    )
    fig_frq.update_layout(**_layout(height=360, bargap=0.03))

    # 5. Monetary box
    mon_cap = df["Monetary"].quantile(0.97)
    fig_box = px.box(
        df[df["Monetary"] <= mon_cap],
        x="Segment", y="Monetary",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title="Monetary Value Distribution by Segment", points="outliers",
    )
    fig_box.update_traces(marker_size=4, line_width=1.8)
    fig_box.update_layout(**_layout(height=400, showlegend=False))

    # 6. Scatter
    df_samp = df.sample(min(1800, len(df)), random_state=42)
    fig_scat = px.scatter(
        df_samp, x="Recency", y="Monetary",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        size="Frequency", size_max=20, opacity=0.70,
        title="Recency vs Monetary Value (bubble = Frequency)",
    )
    fig_scat.update_layout(**_layout(height=460))

    # 7. Treemap
    top20 = df.nlargest(20, "Monetary").copy()
    top20["CID"] = top20["CustomerID"].astype(str)
    fig_tree = px.treemap(
        top20, path=["Segment", "CID"], values="Monetary",
        color="Monetary",
        color_continuous_scale=["#1e1b4b", "#6366f1", "#a855f7"],
        title="Top-20 Customers — Revenue Contribution",
    )
    fig_tree.update_layout(
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc"),
        margin=dict(l=10, r=10, t=52, b=10),
        title_font=dict(color="#e2e8f0", size=14),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
    )

    # 8. Freq bucket bar
    df_bkt = df.copy()
    df_bkt["Freq Bucket"] = pd.cut(
        df_bkt["Frequency"],
        bins=[0, 2, 5, 10, 20, df_bkt["Frequency"].max() + 1],
        labels=["1–2", "3–5", "6–10", "11–20", "20+"],
    )
    avg_mon = (
        df_bkt.groupby("Freq Bucket", observed=True)["Monetary"]
        .mean().reset_index()
    )
    avg_mon.columns = ["Frequency Bucket", "Avg Monetary (£)"]
    fig_bkt = px.bar(
        avg_mon, x="Frequency Bucket", y="Avg Monetary (£)",
        color="Avg Monetary (£)",
        color_continuous_scale=["#3b82f6", "#6366f1", "#a855f7"],
        title="Avg Lifetime Value by Purchase Frequency Bucket",
        text=avg_mon["Avg Monetary (£)"].map("£{:,.0f}".format),
    )
    fig_bkt.update_traces(textposition="outside", marker_line_width=0, width=0.52)
    fig_bkt.update_layout(**_layout(height=360, showlegend=False, coloraxis_showscale=False))

    return render_template(
        "analytics.html",
        active_page="analytics",
        chart_pie=to_json(fig_pie),
        chart_top=to_json(fig_top),
        chart_rec=to_json(fig_rec),
        chart_frq=to_json(fig_frq),
        chart_box=to_json(fig_box),
        chart_scat=to_json(fig_scat),
        chart_tree=to_json(fig_tree),
        chart_bkt=to_json(fig_bkt),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")


@app.route("/update-db", methods=["GET", "POST"])
def update_db():
    return render_template("update_db.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
