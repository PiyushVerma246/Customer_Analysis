"""
app.py — RetailIQ Customer Management & Analytics Platform
==========================================================
All routes use retailer-friendly language. ML terms are hidden from the UI.

Routes
------
GET  /                          → Landing page
GET  /dashboard                 → Business KPI dashboard
GET  /segmentation              → Customer Categories
GET  /churn                     → Customer Retention
GET/POST /recommendations       → Frequently Bought Together
GET  /forecasting               → Expected Sales forecast
GET/POST /customer-entry        → Add customer / record purchase
GET  /upload-dataset            → Batch CSV import
GET  /history                   → Customer Activity Log
GET  /customers                 → Customer Directory
GET  /customers/top             → Top Customers
GET  /customer/<customer_id>    → Customer Profile (mini-CRM)
GET/POST /customer-search       → Customer Search
GET  /api/next-customer-id      → Next auto CUST ID (JSON)
GET  /api/search-customers      → Autocomplete search (JSON)
POST /api/predict               → JSON prediction endpoint
GET  /api/forecast              → JSON forecast data
GET  /api/dashboard-stats       → JSON KPI stats
"""

import io
import json
import warnings
from datetime import date

import pandas as pd
from flask import (Flask, Response, jsonify, redirect,
                   render_template, request, url_for, session, flash)
from functools import wraps

from utils.database import (
    init_db,
    generate_next_customer_id,
    customer_id_exists,
    save_customer_record,
    save_transaction,
    save_prediction,
    get_customer_by_id,
    get_customer_stats,
    search_customers_autocomplete,
    get_customer_transactions,
    compute_customer_rfm,
    get_customer_predictions,
    get_latest_prediction,
    get_all_predictions,
    get_all_customers_with_stats,
    get_top_customers,
    create_user,
    verify_user,
    count_users,
)
from utils.preprocessing import (
    load_customer_segments,
    load_churn_data,
    load_recommendation_rules,
    load_cluster_summary,
    load_dashboard_summary,
    compute_rfm_from_upload,
)
from utils.segmentation import (
    predict_segment, models_loaded as seg_loaded,
    SEGMENT_COLORS, SEGMENT_LABEL_MAP, SEGMENT_ACTIONS,
)
from utils.churn import predict_churn, model_loaded as churn_loaded, RISK_ACTIONS
from utils.recommendation import get_all_products, get_recommendations, get_overview
from utils.forecasting import get_forecast_series, get_forecast_summary

warnings.filterwarnings("ignore")

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "retailiq-super-secret-key-change-in-production"
init_db()

# ── Auth Decorator ─────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ── Global dataset data (loaded once at startup) ───────────────────────────
DF_SEGMENTS = load_customer_segments()
DF_CHURN    = load_churn_data()
DF_RULES    = load_recommendation_rules()
DF_CLUSTER  = load_cluster_summary()

# ── Friendly label map for templates ──────────────────────────────────────
ALL_CATEGORIES = list(SEGMENT_LABEL_MAP.values())
ALL_RISKS      = ["High Retention Risk", "Medium Retention Risk", "Low Retention Risk"]


# ── Context processor ─────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    df = DF_SEGMENTS
    if df.empty:
        return dict(
            sb_total="—", sb_best="—", sb_repeat="—",
            sb_standard="—", sb_losing="—", sb_pairs="—",
            seg_loaded=False, churn_loaded=False,
        )
    return dict(
        sb_total    = f"{len(df):,}",
        sb_best     = f"{len(df[df['Segment']=='VIP']):,}",
        sb_repeat   = f"{len(df[df['Segment']=='Loyal']):,}",
        sb_standard = f"{len(df[df['Segment']=='Regular']):,}",
        sb_losing   = f"{len(df[df['Segment']=='At Risk']):,}",
        sb_pairs    = f"{len(DF_RULES):,}" if not DF_RULES.empty else "0",
        seg_loaded  = seg_loaded(),
        churn_loaded= churn_loaded(),
    )


# ══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
        
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if verify_user(username, password):
            session["user_id"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password"
            
    return render_template("login.html", error=error, has_users=(count_users() > 0))

@app.route("/register", methods=["GET", "POST"])
def register():
    # Optional: only allow registration if no users exist, or allow anytime.
    # For this demo, we allow creating users anytime.
    if "user_id" in session:
        return redirect(url_for("dashboard"))
        
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            error = "Username and password are required."
        elif create_user(username, password):
            session["user_id"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Username already exists."
            
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    summary = load_dashboard_summary()
    return render_template("index.html", active_page="home", summary=summary)

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    summary  = load_dashboard_summary()
    df       = DF_SEGMENTS
    churn_df = DF_CHURN

    if df.empty:
        return render_template("dashboard.html", active_page="dashboard",
                               error=True, summary=summary)

    # Category breakdown (friendly labels)
    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    total = len(df)

    seg_breakdown = []
    for _, row in seg_counts.iterrows():
        friendly = SEGMENT_LABEL_MAP.get(row["Segment"], row["Segment"])
        seg_breakdown.append({
            "segment":  row["Segment"],
            "category": friendly,
            "count":    int(row["Count"]),
            "pct":      round(row["Count"] / total * 100, 1),
            "color":    SEGMENT_COLORS.get(friendly, "#64748b"),
            "actions":  SEGMENT_ACTIONS.get(friendly, []),
        })

    # Retention stats
    churn_total = len(churn_df) if not churn_df.empty else 0
    at_risk_count = int(churn_df["Churn"].sum()) if not churn_df.empty and "Churn" in churn_df.columns else 0
    retention_rate = round((churn_total - at_risk_count) / churn_total * 100, 1) if churn_total else 0

    # Revenue by category
    rev_by_segment = []
    for _, row in df.groupby("Segment")["Monetary"].sum().reset_index().iterrows():
        friendly = SEGMENT_LABEL_MAP.get(row["Segment"], row["Segment"])
        rev_by_segment.append({
            "category": friendly,
            "Revenue":  round(float(row["Monetary"]), 2),
            "color":    SEGMENT_COLORS.get(friendly, "#64748b"),
        })

    cluster_rows = DF_CLUSTER.to_dict("records") if not DF_CLUSTER.empty else []
    forecast_preview = get_forecast_series(days=30)

    return render_template(
        "dashboard.html",
        active_page        = "dashboard",
        summary            = summary,
        seg_breakdown      = seg_breakdown,
        rev_by_segment     = rev_by_segment,
        at_risk_count      = at_risk_count,
        retention_rate     = retention_rate,
        cluster_rows       = cluster_rows,
        forecast_preview   = json.dumps(forecast_preview),
        seg_breakdown_json = json.dumps(seg_breakdown),
        rev_json           = json.dumps(rev_by_segment),
        risk_actions       = RISK_ACTIONS.get("High Retention Risk", []),
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER CATEGORIES (was: segmentation)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/segmentation")
@login_required
def segmentation():
    df = DF_SEGMENTS
    if df.empty:
        return render_template("segmentation.html", active_page="segmentation", error=True)

    total = len(df)
    seg_counts = df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    seg_data = []
    for _, row in seg_counts.iterrows():
        friendly = SEGMENT_LABEL_MAP.get(row["Segment"], row["Segment"])
        sub = df[df["Segment"] == row["Segment"]]
        seg_data.append({
            "segment":       row["Segment"],
            "category":      friendly,
            "count":         int(row["Count"]),
            "pct":           round(row["Count"] / total * 100, 1),
            "color":         SEGMENT_COLORS.get(friendly, "#64748b"),
            "avg_recency":   round(sub["Recency"].mean(), 1),
            "avg_frequency": round(sub["Frequency"].mean(), 1),
            "avg_monetary":  round(sub["Monetary"].mean(), 2),
            "total_revenue": round(sub["Monetary"].sum(), 2),
            "actions":       SEGMENT_ACTIONS.get(friendly, []),
        })

    cluster_rows = DF_CLUSTER.to_dict("records") if not DF_CLUSTER.empty else []

    sample = (
        df.sample(min(20, total), random_state=42)
        [["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]]
        .sort_values("Segment")
        .to_dict("records")
    )
    # Translate segment in sample
    for r in sample:
        r["category"] = SEGMENT_LABEL_MAP.get(r["Segment"], r["Segment"])

    return render_template(
        "segmentation.html",
        active_page    = "segmentation",
        seg_data       = seg_data,
        seg_data_json  = json.dumps(seg_data),
        cluster_rows   = cluster_rows,
        sample         = sample,
        total          = total,
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER RETENTION (was: churn)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/churn")
@login_required
def churn():
    df = DF_CHURN
    if df.empty:
        return render_template("churn.html", active_page="churn", error=True)

    total    = len(df)
    churned  = int(df["Churn"].sum()) if "Churn" in df.columns else 0
    retained = total - churned
    churn_rate = round(churned / total * 100, 1) if total else 0

    churn_seg_data = []
    if "Segment" in df.columns and "Churn" in df.columns:
        gdf = (
            df.groupby("Segment")["Churn"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "Churned", "count": "Total"})
        )
        for _, row in gdf.iterrows():
            friendly = SEGMENT_LABEL_MAP.get(row["Segment"], row["Segment"])
            churn_seg_data.append({
                "segment":  row["Segment"],
                "category": friendly,
                "Churned":  int(row["Churned"]),
                "Total":    int(row["Total"]),
                "ChurnRate":round(row["Churned"] / row["Total"] * 100, 1),
                "color":    SEGMENT_COLORS.get(friendly, "#64748b"),
            })

    at_risk_table = []
    if "ChurnRisk" in df.columns:
        at_risk = df[df["ChurnRisk"] >= 0.70].nlargest(20, "ChurnRisk")
        for _, row in at_risk.iterrows():
            friendly = SEGMENT_LABEL_MAP.get(row.get("Segment", ""), row.get("Segment", ""))
            at_risk_table.append({
                "CustomerID": int(row["CustomerID"]),
                "category":   friendly,
                "Recency":    row.get("Recency", 0),
                "Frequency":  row.get("Frequency", 0),
                "Monetary":   row.get("Monetary", 0),
                "ChurnRisk":  row.get("ChurnRisk", 0),
            })

    return render_template(
        "churn.html",
        active_page    = "churn",
        total          = total,
        churned        = churned,
        retained       = retained,
        churn_rate     = churn_rate,
        churn_seg_data = churn_seg_data,
        churn_seg_json = json.dumps(churn_seg_data),
        at_risk_table  = at_risk_table,
        risk_actions   = RISK_ACTIONS.get("High Retention Risk", []),
    )


# ══════════════════════════════════════════════════════════════════════════
# FREQUENTLY BOUGHT TOGETHER (was: recommendations)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/recommendations", methods=["GET", "POST"])
@login_required
def recommendations():
    df           = DF_RULES
    all_products = get_all_products(df)
    overview     = get_overview(df)
    selected     = None
    rec_cards    = []
    rec_warning  = None
    stats        = {}

    if request.method == "POST":
        selected = request.form.get("product", "").strip()
        if selected:
            rec_cards = get_recommendations(df, selected)
            if not rec_cards:
                rec_warning = f"No product pairings found for '{selected}'."
            else:
                stats = {
                    "count":    len(rec_cards),
                    "max_lift": max(r["lift_float"] for r in rec_cards),
                    "avg_conf": f"{sum(r['conf_pct'] for r in rec_cards)/len(rec_cards):.1f}%",
                }

    return render_template(
        "recommendations.html",
        active_page  = "recommendations",
        all_products = all_products,
        overview     = overview,
        selected     = selected,
        rec_cards    = rec_cards,
        rec_warning  = rec_warning,
        stats        = stats,
    )


# ══════════════════════════════════════════════════════════════════════════
# EXPECTED SALES (was: forecasting)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/forecasting")
@login_required
def forecasting():
    summary      = get_forecast_summary()
    forecast_all = get_forecast_series()
    return render_template(
        "forecasting.html",
        active_page   = "forecasting",
        summary       = summary,
        forecast_json = json.dumps(forecast_all),
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER ENTRY — Add/Select Customer + Record Purchase
# ══════════════════════════════════════════════════════════════════════════
@app.route("/customer-entry", methods=["GET", "POST"])
@login_required
def customer_entry():
    result   = None
    form_err = None
    next_id  = generate_next_customer_id()
    cust_stats = get_customer_stats()

    if request.method == "POST":
        try:
            mode        = request.form.get("mode", "new")   # "new" or "existing"
            name        = request.form.get("name", "").strip()
            phone       = request.form.get("phone", "").strip()
            amount_str  = request.form.get("purchase_amount", "").strip()
            pdate       = request.form.get("purchase_date", "").strip()
            product     = request.form.get("product_purchased", "").strip()

            if not amount_str:
                raise ValueError("Purchase amount is required.")
            purchase_amount = float(amount_str)
            if purchase_amount <= 0:
                raise ValueError("Purchase amount must be greater than £0.")
            if not pdate:
                raise ValueError("Purchase date is required.")

            # Determine Customer ID
            if mode == "existing":
                cid = request.form.get("existing_customer_id", "").strip()
                if not cid:
                    raise ValueError("Please select an existing customer.")
                if not customer_id_exists(cid):
                    raise ValueError(f"Customer '{cid}' not found in the system.")
            else:
                # New customer
                if not name:
                    raise ValueError("Customer name is required for new customers.")
                cid = next_id
                save_customer_record({
                    "customer_id": cid,
                    "name":  name,
                    "phone": phone,
                })

            # Record the purchase
            save_transaction({
                "customer_id":       cid,
                "purchase_amount":   purchase_amount,
                "purchase_date":     pdate,
                "product_purchased": product,
            })

            # Compute RFM from full transaction history
            rfm = compute_customer_rfm(cid)

            # Run ML predictions
            seg = predict_segment(rfm["recency"], rfm["frequency"], rfm["monetary"])
            ch  = predict_churn(
                frequency       = rfm["frequency"],
                monetary        = rfm["monetary"],
                total_revenue   = rfm["total_revenue"],
                total_quantity  = rfm["total_quantity"],
                unique_products = rfm["unique_products"],
            )

            # Save prediction
            save_prediction({
                "customer_id":       cid,
                "customer_category": seg["customer_category"],
                "retention_risk":    ch["retention_risk"],
                "churn_probability": ch["churn_probability"],
            })

            # Refresh stats after new customer
            cust_stats = get_customer_stats()
            next_id    = generate_next_customer_id()

            result = {
                "customer_id":    cid,
                "name":           name if mode == "new" else (get_customer_by_id(cid) or {}).get("name", ""),
                "purchase_amount":purchase_amount,
                "purchase_date":  pdate,
                "product":        product,
                **rfm, **seg, **ch,
            }

        except (ValueError, TypeError) as e:
            form_err = str(e)
        except RuntimeError as e:
            form_err = f"Model error: {e}"
        except Exception as e:
            form_err = f"Unexpected error: {e}"

    recent = get_all_predictions()[:8]

    return render_template(
        "customer_entry.html",
        active_page  = "customer_entry",
        result       = result,
        form_err     = form_err,
        next_id      = next_id,
        cust_stats   = cust_stats,
        recent       = recent,
        seg_loaded   = seg_loaded(),
        churn_loaded = churn_loaded(),
        today        = date.today().isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════
# UPLOAD DATASET (batch import — keeps original IDs)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/upload-dataset", methods=["GET", "POST"])
@login_required
def upload_dataset():
    results   = []
    form_err  = None
    form_info = None

    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            form_err = "Please select a CSV file to upload."
        elif not file.filename.endswith(".csv"):
            form_err = "Only CSV files are accepted."
        else:
            try:
                df_raw = pd.read_csv(io.StringIO(file.read().decode("utf-8", errors="replace")))
                df_raw.columns = [c.strip() for c in df_raw.columns]

                required_cols = {"CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
                missing = required_cols - set(df_raw.columns)
                if missing:
                    raise ValueError(
                        f"Missing columns: {', '.join(sorted(missing))}. "
                        "Required: CustomerID, InvoiceDate, Quantity, UnitPrice"
                    )

                rfm_df = compute_rfm_from_upload(df_raw)

                for _, row in rfm_df.iterrows():
                    try:
                        cid = str(int(row["CustomerID"]))
                        # Keep original CSV ID — save customer record
                        save_customer_record({
                            "customer_id": cid,
                            "name":  f"Imported #{cid}",
                            "phone": "",
                        })
                        # Save one aggregate transaction representing total spend
                        save_transaction({
                            "customer_id":       cid,
                            "purchase_amount":   float(row["Monetary"]),
                            "purchase_date":     date.today().isoformat(),
                            "product_purchased": "Dataset Import",
                        })
                        seg = predict_segment(row["Recency"], row["Frequency"], row["Monetary"])
                        ch  = predict_churn(
                            frequency       = row["Frequency"],
                            monetary        = row["Monetary"],
                            total_revenue   = row["TotalRevenue"],
                            total_quantity  = row["TotalQuantity"],
                            unique_products = row["UniqueProducts"],
                        )
                        save_prediction({
                            "customer_id":       cid,
                            "customer_category": seg["customer_category"],
                            "retention_risk":    ch["retention_risk"],
                            "churn_probability": ch["churn_probability"],
                        })
                        results.append({
                            "customer_id":    cid,
                            "monetary":       round(row["Monetary"], 2),
                            "frequency":      int(row["Frequency"]),
                            "recency":        int(row["Recency"]),
                            "customer_category": seg["customer_category"],
                            "retention_risk":    ch["retention_risk"],
                            "churn_pct":         ch["churn_pct"],
                        })
                    except Exception:
                        continue

                form_info = f"Successfully imported {len(results)} customers from '{file.filename}'."

            except ValueError as e:
                form_err = str(e)
            except Exception as e:
                form_err = f"Error processing file: {e}"

    return render_template(
        "upload_dataset.html",
        active_page  = "upload_dataset",
        results      = results,
        form_err     = form_err,
        form_info    = form_info,
        seg_loaded   = seg_loaded(),
        churn_loaded = churn_loaded(),
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER ACTIVITY LOG (was: history)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/history")
@login_required
def history():
    predictions = get_all_predictions()
    return render_template(
        "history.html",
        active_page = "history",
        predictions = predictions,
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER DIRECTORY
# ══════════════════════════════════════════════════════════════════════════
@app.route("/customers")
@login_required
def customers():
    search   = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip()
    risk     = request.args.get("risk", "").strip()
    sort_by  = request.args.get("sort", "total_spending")
    sort_dir = request.args.get("dir", "desc")
    page     = max(1, int(request.args.get("page", 1)))

    data = get_all_customers_with_stats(
        search=search, category=category, risk=risk,
        sort_by=sort_by, sort_dir=sort_dir,
        page=page, per_page=25,
    )
    
    cust_stats = get_customer_stats()

    return render_template(
        "customers.html",
        active_page    = "customers",
        data           = data,
        cust_stats     = cust_stats,
        search         = search,
        sel_category   = category,
        sel_risk       = risk,
        sort_by        = sort_by,
        sort_dir       = sort_dir,
        all_categories = ALL_CATEGORIES,
        all_risks      = ALL_RISKS,
    )


# ── Export CSV ─────────────────────────────────────────────────────────────
@app.route("/customers/export")
@login_required
def customers_export():
    data = get_all_customers_with_stats(per_page=10000, page=1)
    rows = data["rows"]
    if not rows:
        return redirect(url_for("customers"))

    lines = ["Customer ID,Name,Customer Category,Retention Risk,Total Spending,Orders,CLV,Avg Order,Last Purchase"]
    for r in rows:
        lines.append(
            f"{r['customer_id']},{r.get('name','')},{r.get('customer_category','')},{r.get('retention_risk','')},"
            f"{r.get('total_spending',0):.2f},{r.get('order_count',0)},"
            f"{r.get('clv',0):.2f},{r.get('avg_order_value',0):.2f},{r.get('last_purchase','')}"
        )
    csv_content = "\n".join(lines)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=customers.csv"},
    )


# ══════════════════════════════════════════════════════════════════════════
# TOP CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════
@app.route("/customers/top")
@login_required
def customers_top():
    sort_by = request.args.get("sort", "total_spending")
    rows    = get_top_customers(n=25, sort_by=sort_by)
    return render_template(
        "customers_top.html",
        active_page = "customers_top",
        rows        = rows,
        sort_by     = sort_by,
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER PROFILE (mini-CRM)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/customer/<customer_id>")
@login_required
def customer_detail(customer_id):
    customer = get_customer_by_id(customer_id)
    if not customer:
        return render_template("404.html"), 404

    transactions = get_customer_transactions(customer_id)
    predictions  = get_customer_predictions(customer_id)
    latest_pred  = predictions[0] if predictions else None
    rfm          = compute_customer_rfm(customer_id)

    # Recommendations based on last purchased product
    rec_product = None
    rec_cards   = []
    if transactions:
        last_product = transactions[0].get("product_purchased", "")
        if last_product and last_product not in ("Dataset Import", ""):
            rec_product = last_product
            rec_cards   = get_recommendations(DF_RULES, last_product)[:6]

    # Spending trend for chart (last 12 transactions)
    spend_trend = [
        {"date": t["purchase_date"], "amount": t["purchase_amount"]}
        for t in reversed(transactions[-12:])
    ]

    return render_template(
        "customer_detail.html",
        active_page  = "customers",
        customer     = customer,
        transactions = transactions,
        predictions  = predictions,
        latest_pred  = latest_pred,
        rfm          = rfm,
        rec_product  = rec_product,
        rec_cards    = rec_cards,
        spend_trend  = json.dumps(spend_trend),
    )


# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER SEARCH
# ══════════════════════════════════════════════════════════════════════════
@app.route("/customer-search", methods=["GET", "POST"])
@login_required
def customer_search():
    query    = request.args.get("q", "").strip() or request.form.get("q", "").strip()
    category = request.args.get("category", "").strip()
    risk     = request.args.get("risk", "").strip()
    results  = []

    if query or category or risk:
        data = get_all_customers_with_stats(
            search=query, category=category, risk=risk,
            sort_by="customer_id", sort_dir="asc",
            page=1, per_page=50,
        )
        results = data["rows"]

    return render_template(
        "customer_search.html",
        active_page    = "customer_search",
        query          = query,
        sel_category   = category,
        sel_risk       = risk,
        results        = results,
        all_categories = ALL_CATEGORIES,
        all_risks      = ALL_RISKS,
    )


# ══════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/next-customer-id")
@login_required
def api_next_customer_id():
    return jsonify({"next_id": generate_next_customer_id()})


@app.route("/api/search-customers")
@login_required
def api_search_customers():
    q       = request.args.get("q", "").strip()
    results = search_customers_autocomplete(q) if q else []
    return jsonify(results)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        body = request.get_json(force=True) or {}
        cid  = str(body.get("customer_id", ""))

        rfm_data = compute_customer_rfm(cid) if customer_id_exists(cid) else {
            "recency":        float(body.get("recency", 30)),
            "frequency":      float(body.get("frequency", 1)),
            "monetary":       float(body.get("monetary", 0)),
            "total_revenue":  float(body.get("monetary", 0)),
            "total_quantity": float(body.get("frequency", 1)),
            "unique_products":1.0,
        }

        seg = predict_segment(rfm_data["recency"], rfm_data["frequency"], rfm_data["monetary"])
        ch  = predict_churn(
            rfm_data["frequency"], rfm_data["monetary"],
            rfm_data["total_revenue"], rfm_data["total_quantity"], rfm_data["unique_products"],
        )

        if cid:
            save_prediction({
                "customer_id":       cid,
                "customer_category": seg["customer_category"],
                "retention_risk":    ch["retention_risk"],
                "churn_probability": ch["churn_probability"],
            })

        return jsonify({
            "status":           "success",
            "customer_id":      cid,
            "customer_category":seg["customer_category"],
            "retention_risk":   ch["retention_risk"],
            "churn_probability":ch["churn_probability"],
            "churn_pct":        ch["churn_pct"],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/forecast")
@login_required
def api_forecast():
    days_param = request.args.get("days", "all")
    days   = None if days_param == "all" else int(days_param)
    series = get_forecast_series(days=days)
    summ   = get_forecast_summary(days=days)
    return jsonify({"status": "success", "data": series, "summary": summ})


@app.route("/api/dashboard-stats")
@login_required
def api_dashboard_stats():
    summary = load_dashboard_summary()
    df = DF_SEGMENTS
    seg_dist = {}
    if not df.empty:
        for seg, friendly in SEGMENT_LABEL_MAP.items():
            seg_dist[friendly] = int((df["Segment"] == seg).sum())
    return jsonify({"status": "success", "kpis": summary, "seg_distribution": seg_dist})

@app.route("/api/generate-demo-data", methods=["POST"])
@login_required
def api_generate_demo_data():
    try:
        from utils.database import get_connection
        import pandas as pd
        from datetime import date
        
        conn = get_connection()
        c = conn.cursor()
        
        segments = pd.read_csv('data/customer_segments.csv')
        churn = pd.read_csv('data/customer_churn.csv')
        merged = pd.merge(segments, churn, on='CustomerID', how='outer')
        
        # Insert a subset to keep it fast
        for _, row in merged.head(150).iterrows():
            cid = str(int(row['CustomerID']))
            c.execute('INSERT OR IGNORE INTO customers (customer_id, name, phone, created_at) VALUES (?, ?, ?, ?)',
                     (cid, f'Customer #{cid}', '', date.today().isoformat()))
            
            monetary = float(row.get('Monetary_x', row.get('Monetary_y', 0)))
            if not pd.isna(monetary):
                c.execute('INSERT INTO transactions (customer_id, purchase_amount, purchase_date, product_purchased, created_at) VALUES (?, ?, ?, ?, ?)',
                         (cid, monetary, date.today().isoformat(), 'Demo Import', date.today().isoformat()))
            
            seg = row.get('Segment_x', row.get('Segment_y', 'Regular'))
            risk = row.get('ChurnRisk', 0)
            risk_label = 'High Retention Risk' if risk >= 0.7 else 'Medium Retention Risk' if risk >= 0.4 else 'Low Retention Risk'
            
            c.execute('INSERT INTO predictions (customer_id, customer_category, retention_risk, churn_probability, prediction_time) VALUES (?, ?, ?, ?, ?)',
                     (cid, seg, risk_label, risk, date.today().isoformat()))
                     
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Demo data generated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── 404 handler ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
