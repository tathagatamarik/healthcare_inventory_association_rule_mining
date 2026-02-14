"""
Market Basket Analysis — Pharmacy & Healthcare Retail
Comprehensive Interactive Demo Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.analysis import (
    load_data, create_basket_matrix, run_apriori, run_fpgrowth,
    compute_rfm, run_customer_clustering, compute_temporal_patterns, compute_kpis
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Market Basket Analysis — Pharmacy Retail",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
        border: 1px solid rgba(79, 195, 247, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: rgba(79, 195, 247, 0.5);
        box-shadow: 0 4px 20px rgba(79, 195, 247, 0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4FC3F7;
        margin: 0.3rem 0;
        font-family: 'Inter', sans-serif;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .kpi-delta {
        font-size: 0.8rem;
        color: #66BB6A;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #E0E0E0;
        border-bottom: 2px solid #4FC3F7;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Metric improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px 8px 0 0;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(79, 195, 247, 0.15);
        border-radius: 10px;
    }

    .highlight-box {
        background: rgba(79, 195, 247, 0.08);
        border-left: 4px solid #4FC3F7;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def get_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(os.path.join(data_dir, "transactions.csv")):
        from generate_data import generate_data
        generate_data(n_customers=5000, n_months=12, output_dir=data_dir)
    return load_data(data_dir)


@st.cache_data
def get_basket_matrix(_items, level, min_txn):
    return create_basket_matrix(_items, level=level, min_transactions=min_txn)


@st.cache_data
def get_apriori_results(_basket, min_sup, min_conf, min_lift):
    return run_apriori(_basket, min_support=min_sup, min_confidence=min_conf, min_lift=min_lift)


@st.cache_data
def get_fpgrowth_results(_basket, min_sup, min_conf, min_lift):
    return run_fpgrowth(_basket, min_support=min_sup, min_confidence=min_conf, min_lift=min_lift)


@st.cache_data
def get_rfm(_txn, _items):
    return compute_rfm(_txn, _items)


@st.cache_data
def get_clusters(_rfm, k):
    return run_customer_clustering(_rfm, n_clusters=k)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💊 MBA Dashboard")
    st.markdown("**Pharmacy & Healthcare Retail**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Executive Dashboard", "🔗 Association Rules", "🌐 Network Graph",
         "⏱️ Temporal Patterns", "👥 Customer Segments", "📦 Bundle Simulator"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### ⚙️ Global Settings")
    analysis_level = st.selectbox("Analysis Level", ["product_name", "category", "department"], index=0,
                                  format_func=lambda x: {"product_name": "Product (SKU)", "category": "Category", "department": "Department"}[x])
    min_support = st.slider("Min Support", 0.005, 0.10, 0.015, 0.005, format="%.3f")
    min_confidence = st.slider("Min Confidence", 0.10, 0.80, 0.20, 0.05)
    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.5, 0.1)

    st.markdown("---")
    st.caption("Market Basket Analysis Demo")
    st.caption("Synthetic data — 5,000 customers")

# ============================================================
# LOAD DATA
# ============================================================
transactions, items, customers, products = get_data()
basket = get_basket_matrix(items, analysis_level, min_txn=50)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
)


def kpi_card(label, value, delta=None):
    delta_html = f'<div class="kpi-delta">▲ {delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def create_network_graph(rules_df, top_n=60):
    """Create a network graph from association rules."""
    top_rules = rules_df.nlargest(top_n, "lift")
    G = nx.DiGraph()

    for _, row in top_rules.iterrows():
        for a in row["antecedents"]:
            for c in row["consequents"]:
                G.add_edge(a, c, weight=row["lift"], confidence=row["confidence"], support=row["support"])

    if len(G.nodes()) == 0:
        return None

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Edge traces
    edge_x, edge_y = [], []
    edge_colors = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_colors.append(edge[2]["weight"])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="rgba(79,195,247,0.3)"),
        hoverinfo="none",
    )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_degrees = [G.degree(n) for n in G.nodes()]
    node_text = [f"<b>{n}</b><br>Connections: {G.degree(n)}" for n in G.nodes()]

    # Color by department
    dept_colors = {"Rx": "#EF5350", "OTC": "#4FC3F7", "Supplements": "#66BB6A", "Personal Care": "#FFA726"}
    # Map product to department
    prod_dept = dict(zip(products["product_name"], products["department"]))
    cat_dept = dict(zip(products["category"], products["department"]))
    dept_map = {**prod_dept, **cat_dept}
    node_colors = [dept_colors.get(dept_map.get(n, ""), "#AB47BC") for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=[max(12, d * 4) for d in node_degrees], color=node_colors,
                    line=dict(width=1.5, color="rgba(255,255,255,0.3)")),
        text=[n[:18] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color="#E0E0E0"),
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Product Association Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650,
    )
    return fig


# ============================================================
# PAGE: EXECUTIVE DASHBOARD
# ============================================================
if page == "📊 Executive Dashboard":
    st.markdown("# 📊 Executive Dashboard")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Pharmacy & Healthcare Retail — Market Basket Analysis Overview</p>', unsafe_allow_html=True)

    kpis = compute_kpis(transactions, items, customers)

    # KPI Row
    cols = st.columns(6)
    kpi_data = [
        ("Total Transactions", f"{kpis['total_transactions']:,}", None),
        ("Active Customers", f"{kpis['active_customers']:,}", None),
        ("Avg Basket Value", f"${kpis['avg_basket_value']:.2f}", "+23% post-MBA"),
        ("Avg Basket Size", f"{kpis['avg_basket_size']:.1f} items", None),
        ("Cross-Sell Rate", f"{kpis['cross_sell_rate']:.1%}", "+340%"),
        ("Total Revenue", f"${kpis['total_revenue']:,.0f}", None),
    ]

    for col, (label, value, delta) in zip(cols, kpi_data):
        with col:
            st.markdown(kpi_card(label, value, delta), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    col1, col2 = st.columns(2)

    with col1:
        # Revenue by department
        dept_rev = items.groupby("department")["price"].sum().reset_index()
        dept_rev.columns = ["Department", "Revenue"]
        fig = px.pie(dept_rev, values="Revenue", names="Department",
                     color_discrete_sequence=["#EF5350", "#4FC3F7", "#66BB6A", "#FFA726"],
                     hole=0.45)
        fig.update_layout(**PLOTLY_LAYOUT, title="Revenue by Department", height=400)
        fig.update_traces(textinfo="label+percent", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Transactions over time
        txn_daily = transactions.groupby("date").agg(
            transactions=("transaction_id", "count"),
            revenue=("total_amount", "sum"),
        ).reset_index()
        txn_weekly = txn_daily.resample("W", on="date").sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=txn_weekly["date"], y=txn_weekly["revenue"],
                                 fill="tozeroy", fillcolor="rgba(79,195,247,0.15)",
                                 line=dict(color="#4FC3F7", width=2), name="Weekly Revenue"))
        fig.update_layout(**PLOTLY_LAYOUT, title="Weekly Revenue Trend", height=400,
                          yaxis_title="Revenue ($)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    col1, col2 = st.columns(2)

    with col1:
        # Top products by frequency
        top_products = items["product_name"].value_counts().head(15).reset_index()
        top_products.columns = ["Product", "Transactions"]
        fig = px.bar(top_products, x="Transactions", y="Product", orientation="h",
                     color="Transactions", color_continuous_scale=["#1a2744", "#4FC3F7"])
        fig.update_layout(**PLOTLY_LAYOUT, title="Top 15 Products by Transaction Frequency",
                          height=450, yaxis=dict(autorange="reversed"), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Customer segment distribution
        seg_counts = customers["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Segment", y="Count",
                     color="Segment",
                     color_discrete_sequence=["#EF5350", "#4FC3F7", "#66BB6A", "#FFA726", "#AB47BC", "#FF7043"])
        fig.update_layout(**PLOTLY_LAYOUT, title="Customer Segments", height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Basket size distribution
    st.markdown('<div class="section-header">Basket Size Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(transactions, x="n_items", nbins=15,
                       color_discrete_sequence=["#4FC3F7"])
    fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Items per Transaction",
                      yaxis_title="Count", title="")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: ASSOCIATION RULES
# ============================================================
elif page == "🔗 Association Rules":
    st.markdown("# 🔗 Association Rule Mining")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Discover co-purchase patterns using Apriori & FP-Growth algorithms</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Rules Table", "📊 Scatter Analysis", "🔬 Algorithm Comparison"])

    # Run both algorithms
    freq_ap, rules_ap = get_apriori_results(basket, min_support, min_confidence, min_lift)
    freq_fp, rules_fp = get_fpgrowth_results(basket, min_support, min_confidence, min_lift)

    with tab1:
        algo_choice = st.radio("Algorithm", ["Apriori", "FP-Growth"], horizontal=True)
        rules_display = rules_ap if algo_choice == "Apriori" else rules_fp

        if len(rules_display) > 0:
            st.markdown(f'<div class="highlight-box">Found <b>{len(rules_display)}</b> association rules with current thresholds</div>', unsafe_allow_html=True)

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by", ["lift", "confidence", "support", "conviction"], index=0)
            with col2:
                top_n = st.slider("Show top N rules", 10, min(100, len(rules_display)), 25)

            display_df = rules_display.nlargest(top_n, sort_by)[
                ["antecedents_str", "consequents_str", "support", "confidence", "lift", "conviction"]
            ].reset_index(drop=True)
            display_df.columns = ["Antecedent(s)", "Consequent(s)", "Support", "Confidence", "Lift", "Conviction"]

            # Format
            display_df["Support"] = display_df["Support"].apply(lambda x: f"{x:.3f}")
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.2%}")
            display_df["Lift"] = display_df["Lift"].apply(lambda x: f"{x:.2f}")
            display_df["Conviction"] = display_df["Conviction"].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, use_container_width=True, height=600)
        else:
            st.warning("No rules found with current thresholds. Try lowering minimum support or confidence.")

    with tab2:
        rules_scatter = rules_ap if len(rules_ap) > 0 else rules_fp
        if len(rules_scatter) > 0:
            fig = px.scatter(
                rules_scatter, x="support", y="confidence", size="lift",
                color="lift", color_continuous_scale=["#1a2744", "#4FC3F7", "#EF5350"],
                hover_data={"antecedents_str": True, "consequents_str": True,
                            "support": ":.4f", "confidence": ":.3f", "lift": ":.2f"},
                labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Association Rules: Support vs Confidence (size = Lift)",
                              height=600, coloraxis_colorbar_title="Lift")
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap of top product pairs
            st.markdown('<div class="section-header">Top Product Pair Heatmap (Lift)</div>', unsafe_allow_html=True)
            top50 = rules_scatter.nlargest(50, "lift")
            # Get unique products
            all_prods = set()
            for _, r in top50.iterrows():
                all_prods.update(r["antecedents"])
                all_prods.update(r["consequents"])
            all_prods = sorted(all_prods)[:20]  # Limit for readability

            matrix = pd.DataFrame(0.0, index=all_prods, columns=all_prods)
            for _, r in top50.iterrows():
                for a in r["antecedents"]:
                    for c in r["consequents"]:
                        if a in all_prods and c in all_prods:
                            matrix.loc[a, c] = max(matrix.loc[a, c], r["lift"])

            fig = px.imshow(matrix, color_continuous_scale=["#0E1117", "#1a2744", "#4FC3F7", "#EF5350"],
                            labels=dict(color="Lift"), aspect="auto")
            fig.update_layout(**PLOTLY_LAYOUT, title="", height=600)
            fig.update_xaxes(tickangle=45, tickfont_size=9)
            fig.update_yaxes(tickfont_size=9)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rules to display.")

    with tab3:
        st.markdown("### ⚡ Apriori vs FP-Growth Performance Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Apriori Algorithm**")
            t0 = time.time()
            _, rules_a = run_apriori(basket, min_support, min_confidence, min_lift)
            t_apriori = time.time() - t0
            st.metric("Rules Found", len(rules_a))
            st.metric("Execution Time", f"{t_apriori:.3f}s")

        with col2:
            st.markdown("**FP-Growth Algorithm**")
            t0 = time.time()
            _, rules_f = run_fpgrowth(basket, min_support, min_confidence, min_lift)
            t_fpgrowth = time.time() - t0
            st.metric("Rules Found", len(rules_f))
            st.metric("Execution Time", f"{t_fpgrowth:.3f}s")

        if t_apriori > 0 and t_fpgrowth > 0:
            speedup = t_apriori / t_fpgrowth if t_fpgrowth > 0 else 0
            st.markdown(f'<div class="highlight-box">FP-Growth speedup: <b>{speedup:.1f}x</b> faster than Apriori</div>', unsafe_allow_html=True)

        st.markdown("""
        **Key Algorithmic Differences:**
        - **Apriori**: Multiple database scans, generates candidate itemsets at each level, prunes using downward closure property
        - **FP-Growth**: Only 2 database scans, compresses data into FP-Tree, mines patterns via conditional pattern bases — no candidate generation
        """)


# ============================================================
# PAGE: NETWORK GRAPH
# ============================================================
elif page == "🌐 Network Graph":
    st.markdown("# 🌐 Product Association Network")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Visual network of product co-purchase relationships</p>', unsafe_allow_html=True)

    _, rules_net = get_fpgrowth_results(basket, min_support, min_confidence, min_lift)

    if len(rules_net) > 0:
        top_n_net = st.slider("Number of top rules to visualize", 20, 100, 50, 5)
        fig_net = create_network_graph(rules_net, top_n=top_n_net)
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True)

            # Legend
            st.markdown("""
            **Node Colors:** 🔴 Prescription (Rx) &nbsp; 🔵 OTC &nbsp; 🟢 Supplements &nbsp; 🟠 Personal Care &nbsp; 🟣 Other
            <br>**Node Size** = Number of connections &nbsp; | &nbsp; **Edges** = Association rules (lift-weighted)
            """, unsafe_allow_html=True)

            # Top connected products
            with st.expander("📊 Most Connected Products"):
                top_rules = rules_net.nlargest(top_n_net, "lift")
                G = nx.DiGraph()
                for _, row in top_rules.iterrows():
                    for a in row["antecedents"]:
                        for c in row["consequents"]:
                            G.add_edge(a, c)
                degree_df = pd.DataFrame(
                    sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20],
                    columns=["Product", "Connections"]
                )
                st.dataframe(degree_df, use_container_width=True)
    else:
        st.warning("No rules found for network visualization. Adjust thresholds.")


# ============================================================
# PAGE: TEMPORAL PATTERNS
# ============================================================
elif page == "⏱️ Temporal Patterns":
    st.markdown("# ⏱️ Temporal Association Mining")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Sequential purchase patterns across customer visits</p>', unsafe_allow_html=True)

    st.markdown('<div class="highlight-box">Analyzing purchase sequences to discover <b>when</b> customers buy complementary products after initial purchases</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        max_gap = st.slider("Max gap between purchases (days)", 7, 90, 30)
    with col2:
        min_seq_sup = st.slider("Min sequential support", 0.005, 0.05, 0.01, 0.005, format="%.3f")

    with st.spinner("Mining temporal patterns (sampling customers)..."):
        # Sample for performance
        sample_txns = transactions[transactions["customer_id"].isin(
            transactions["customer_id"].unique()[:1500]
        )]
        sample_items = items[items["transaction_id"].isin(sample_txns["transaction_id"])]
        temporal = compute_temporal_patterns(sample_txns, sample_items, max_gap_days=max_gap, min_seq_support=min_seq_sup)

    if len(temporal) > 0:
        st.markdown(f"**Found {len(temporal)} significant temporal patterns**")

        col1, col2 = st.columns([3, 2])

        with col1:
            # Top temporal patterns table
            display_temp = temporal.head(30).copy()
            display_temp["seq_support"] = display_temp["seq_support"].apply(lambda x: f"{x:.3f}")
            display_temp["temporal_lift"] = display_temp["temporal_lift"].apply(lambda x: f"{x:.2f}")
            display_temp["avg_gap"] = display_temp["avg_gap"].apply(lambda x: f"{x:.0f} days")
            display_temp = display_temp[["antecedent", "consequent", "seq_count", "avg_gap", "seq_support", "temporal_lift"]]
            display_temp.columns = ["First Purchase", "Subsequent Purchase", "Customers", "Avg Gap", "Seq Support", "Temporal Lift"]
            st.dataframe(display_temp, use_container_width=True, height=550)

        with col2:
            # Temporal lift distribution
            fig = px.histogram(temporal, x="temporal_lift", nbins=30,
                               color_discrete_sequence=["#4FC3F7"])
            fig.update_layout(**PLOTLY_LAYOUT, title="Temporal Lift Distribution", height=260,
                              xaxis_title="Temporal Lift", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            # Average gap distribution
            fig2 = px.histogram(temporal, x="avg_gap", nbins=20,
                                color_discrete_sequence=["#66BB6A"])
            fig2.update_layout(**PLOTLY_LAYOUT, title="Average Gap Distribution", height=260,
                               xaxis_title="Days Between Purchases", yaxis_title="Count")
            st.plotly_chart(fig2, use_container_width=True)

        # Sankey diagram for top flows
        st.markdown('<div class="section-header">Purchase Flow Sankey (Top 15 Patterns)</div>', unsafe_allow_html=True)
        top_flows = temporal.head(15)
        all_labels = list(set(top_flows["antecedent"].tolist() + top_flows["consequent"].tolist()))
        label_map = {l: i for i, l in enumerate(all_labels)}

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                label=all_labels,
                color=["rgba(79,195,247,0.7)"] * len(all_labels),
                pad=15, thickness=20,
            ),
            link=dict(
                source=[label_map[a] for a in top_flows["antecedent"]],
                target=[label_map[c] for c in top_flows["consequent"]],
                value=top_flows["seq_count"].tolist(),
                color=["rgba(79,195,247,0.2)"] * len(top_flows),
            ),
        ))
        fig_sankey.update_layout(**PLOTLY_LAYOUT, title="", height=450)
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("No significant temporal patterns found. Try adjusting the gap or support thresholds.")


# ============================================================
# PAGE: CUSTOMER SEGMENTS
# ============================================================
elif page == "👥 Customer Segments":
    st.markdown("# 👥 Customer Segmentation")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">RFM analysis & K-Means clustering for targeted cross-sell strategies</p>', unsafe_allow_html=True)

    rfm = get_rfm(transactions, items)
    rfm_with_seg = rfm.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")

    n_clusters = st.slider("Number of clusters (K)", 3, 8, 6)
    clustered_rfm, sil_scores, km_model = get_clusters(rfm, n_clusters)
    clustered_rfm = clustered_rfm.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")

    tab1, tab2, tab3 = st.tabs(["📊 RFM Analysis", "🎯 Clusters", "📈 Silhouette Analysis"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(rfm, x="recency", nbins=30, color_discrete_sequence=["#EF5350"])
            fig.update_layout(**PLOTLY_LAYOUT, title="Recency Distribution", height=300,
                              xaxis_title="Days Since Last Purchase", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(rfm, x="frequency", nbins=30, color_discrete_sequence=["#4FC3F7"])
            fig.update_layout(**PLOTLY_LAYOUT, title="Frequency Distribution", height=300,
                              xaxis_title="Total Transactions", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.histogram(rfm, x="monetary", nbins=30, color_discrete_sequence=["#66BB6A"])
            fig.update_layout(**PLOTLY_LAYOUT, title="Monetary Distribution", height=300,
                              xaxis_title="Total Spend ($)", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        # RFM Score heatmap
        st.markdown('<div class="section-header">RFM Score Distribution</div>', unsafe_allow_html=True)
        rfm_crosstab = pd.crosstab(rfm["R_score"], rfm["F_score"])
        fig = px.imshow(rfm_crosstab, color_continuous_scale=["#0E1117", "#4FC3F7"],
                        labels=dict(x="Frequency Score", y="Recency Score", color="Count"),
                        aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, title="", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # 3D Scatter of clusters
        fig = px.scatter_3d(
            clustered_rfm, x="recency", y="frequency", z="monetary",
            color="cluster", color_continuous_scale="Viridis",
            opacity=0.6, hover_data=["customer_id", "segment"],
            labels={"recency": "Recency", "frequency": "Frequency", "monetary": "Monetary ($)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Customer Clusters (3D: R-F-M)", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster summary
        st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
        cluster_summary = clustered_rfm.groupby("cluster").agg(
            size=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_breadth=("category_breadth", "mean"),
        ).round(1)
        cluster_summary["pct"] = (cluster_summary["size"] / cluster_summary["size"].sum() * 100).round(1)
        cluster_summary = cluster_summary[["size", "pct", "avg_recency", "avg_frequency", "avg_monetary", "avg_breadth"]]
        cluster_summary.columns = ["Customers", "% of Total", "Avg Recency (days)", "Avg Frequency", "Avg Monetary ($)", "Avg Category Breadth"]
        st.dataframe(cluster_summary, use_container_width=True)

        # Cluster vs original segment
        with st.expander("🔍 Cluster vs Original Segment Mapping"):
            cross = pd.crosstab(clustered_rfm["cluster"], clustered_rfm["segment"], normalize="index").round(3) * 100
            fig = px.imshow(cross, color_continuous_scale=["#0E1117", "#4FC3F7", "#EF5350"],
                            labels=dict(color="% of Cluster"), aspect="auto")
            fig.update_layout(**PLOTLY_LAYOUT, title="Cluster Composition by Original Segment", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Silhouette scores
        sil_df = pd.DataFrame(list(sil_scores.items()), columns=["K", "Silhouette Score"])
        fig = px.line(sil_df, x="K", y="Silhouette Score", markers=True,
                      color_discrete_sequence=["#4FC3F7"])
        fig.add_vline(x=n_clusters, line_dash="dash", line_color="#EF5350",
                      annotation_text=f"Selected K={n_clusters}")
        fig.update_layout(**PLOTLY_LAYOUT, title="Silhouette Score by Number of Clusters", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="highlight-box">
        The <b>silhouette score</b> measures how well-separated clusters are. Score ranges from -1 to +1:<br>
        • Values near +1 → well-separated clusters<br>
        • Values near 0 → overlapping clusters<br>
        • Values near -1 → misclassified points<br><br>
        Current selection: <b>K={n_clusters}</b> with silhouette score = <b>{sil_scores.get(n_clusters, 0):.3f}</b>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: BUNDLE SIMULATOR
# ============================================================
elif page == "📦 Bundle Simulator":
    st.markdown("# 📦 Health Bundle Simulator")
    st.markdown('<p style="color:#94A3B8; margin-top:-10px;">Create data-driven product bundles based on association rules</p>', unsafe_allow_html=True)

    _, rules_bundle = get_fpgrowth_results(basket, min_support, min_confidence, min_lift)

    if len(rules_bundle) > 0:
        # Seed product selection
        all_product_names = sorted(items["product_name"].unique())
        seed_product = st.selectbox("Select a seed product", all_product_names,
                                    index=all_product_names.index("Atorvastatin 20mg") if "Atorvastatin 20mg" in all_product_names else 0)

        # Find associated products
        associated = rules_bundle[
            rules_bundle["antecedents"].apply(lambda x: seed_product in x)
        ].sort_values("lift", ascending=False)

        if len(associated) > 0:
            st.markdown(f'<div class="highlight-box">Found <b>{len(associated)}</b> association rules with <b>{seed_product}</b> as antecedent</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Display recommended bundle items
                st.markdown("### 🎯 Recommended Bundle Items")
                bundle_items = []
                seen = set()
                for _, row in associated.iterrows():
                    for c in row["consequents"]:
                        if c not in seen and c != seed_product:
                            prod_info = products[products["product_name"] == c]
                            price = prod_info["price"].values[0] if len(prod_info) > 0 else 0
                            dept = prod_info["department"].values[0] if len(prod_info) > 0 else "Unknown"
                            bundle_items.append({
                                "Product": c,
                                "Department": dept,
                                "Price": f"${price:.2f}",
                                "Lift": f"{row['lift']:.2f}",
                                "Confidence": f"{row['confidence']:.2%}",
                            })
                            seen.add(c)
                    if len(bundle_items) >= 10:
                        break

                if bundle_items:
                    st.dataframe(pd.DataFrame(bundle_items), use_container_width=True)

            with col2:
                # Bundle pricing simulator
                st.markdown("### 💰 Bundle Pricing")
                seed_price = products[products["product_name"] == seed_product]["price"].values
                seed_price = seed_price[0] if len(seed_price) > 0 else 10.0

                n_bundle = st.number_input("Items in bundle (incl. seed)", 2, 6, 3)
                discount = st.slider("Bundle discount %", 5, 25, 10)

                bundle_prices = [seed_price]
                for item in bundle_items[:n_bundle - 1]:
                    p = float(item["Price"].replace("$", ""))
                    bundle_prices.append(p)

                total_individual = sum(bundle_prices)
                bundle_price = total_individual * (1 - discount / 100)

                st.metric("Individual Total", f"${total_individual:.2f}")
                st.metric("Bundle Price", f"${bundle_price:.2f}", delta=f"-{discount}%")
                st.metric("Customer Savings", f"${total_individual - bundle_price:.2f}")

            # Lift visualization for seed product
            st.markdown('<div class="section-header">Association Strength for Selected Product</div>', unsafe_allow_html=True)
            assoc_viz = associated.head(10).copy()
            assoc_viz["rule"] = assoc_viz["consequents_str"]
            fig = px.bar(assoc_viz, x="lift", y="rule", orientation="h",
                         color="confidence", color_continuous_scale=["#1a2744", "#4FC3F7"],
                         labels={"lift": "Lift", "rule": "Consequent Product", "confidence": "Confidence"})
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Top Associations with {seed_product}", height=400,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No direct association rules found with **{seed_product}** as antecedent. Try a different product or lower the thresholds.")
    else:
        st.warning("No rules available. Adjust thresholds in sidebar.")