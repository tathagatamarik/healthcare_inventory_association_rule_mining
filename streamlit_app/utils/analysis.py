"""
Utility functions for Market Basket Analysis
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def load_data(data_dir="data"):
    """Load all datasets."""
    transactions = pd.read_csv(f"{data_dir}/transactions.csv", parse_dates=["date"])
    items = pd.read_csv(f"{data_dir}/transaction_items.csv")
    customers = pd.read_csv(f"{data_dir}/customers.csv")
    products = pd.read_csv(f"{data_dir}/products.csv")
    return transactions, items, customers, products


def create_basket_matrix(items_df, level="product_name", min_transactions=1):
    """Create binary basket matrix for MBA."""
    basket = items_df.groupby(["transaction_id", level])[level].count().unstack().fillna(0)
    basket = basket.map(lambda x: 1 if x > 0 else 0)
    # Filter columns with minimum transaction count
    col_counts = basket.sum()
    basket = basket[col_counts[col_counts >= min_transactions].index]
    return basket


def run_apriori(basket_matrix, min_support=0.01, min_confidence=0.2, min_lift=1.2):
    """Run Apriori algorithm and return frequent itemsets and rules."""
    freq_itemsets = apriori(basket_matrix, min_support=min_support, use_colnames=True)
    if len(freq_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=min_lift, num_itemsets=len(freq_itemsets))
    rules = rules[rules["confidence"] >= min_confidence]
    # Format itemsets for display
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return freq_itemsets, rules


def run_fpgrowth(basket_matrix, min_support=0.01, min_confidence=0.2, min_lift=1.2):
    """Run FP-Growth algorithm and return frequent itemsets and rules."""
    freq_itemsets = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)
    if len(freq_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=min_lift, num_itemsets=len(freq_itemsets))
    rules = rules[rules["confidence"] >= min_confidence]
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return freq_itemsets, rules


def compute_rfm(transactions_df, items_df, reference_date=None):
    """Compute RFM metrics for each customer."""
    if reference_date is None:
        reference_date = transactions_df["date"].max() + pd.Timedelta(days=1)

    # Merge to get customer-level data
    merged = transactions_df.merge(items_df[["transaction_id", "price"]].groupby("transaction_id")["price"].sum().reset_index(),
                                    on="transaction_id")

    rfm = merged.groupby("customer_id").agg(
        recency=("date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("price", "sum"),
    ).reset_index()

    # Compute cross-category breadth
    cat_breadth = items_df.merge(transactions_df[["transaction_id", "customer_id"]], on="transaction_id")
    breadth = cat_breadth.groupby("customer_id")["category"].nunique().reset_index()
    breadth.columns = ["customer_id", "category_breadth"]

    rfm = rfm.merge(breadth, on="customer_id", how="left")

    # RFM Scores (1-5)
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]

    return rfm


def run_customer_clustering(rfm_df, n_clusters=6):
    """Run K-Means clustering on RFM features."""
    features = ["recency", "frequency", "monetary", "category_breadth"]
    X = rfm_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal K using silhouette
    silhouette_scores = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

    # Use specified n_clusters
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["cluster"] = km.fit_predict(X_scaled)

    return rfm_df, silhouette_scores, km


def compute_temporal_patterns(transactions_df, items_df, max_gap_days=30, min_seq_support=0.01):
    """Compute simplified temporal association patterns."""
    # Merge transaction dates with items
    merged = items_df.merge(transactions_df[["transaction_id", "customer_id", "date"]], on="transaction_id")
    merged = merged.sort_values(["customer_id", "date"])

    # For each customer, find sequential purchases
    patterns = []
    n_customers = merged["customer_id"].nunique()

    for cid, group in merged.groupby("customer_id"):
        group = group.sort_values("date")
        dates = group["date"].unique()

        for i in range(len(dates) - 1):
            for j in range(i + 1, min(i + 4, len(dates))):
                gap = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days
                if 1 <= gap <= max_gap_days:
                    items_t1 = set(group[group["date"] == dates[i]]["product_name"])
                    items_t2 = set(group[group["date"] == dates[j]]["product_name"])
                    for a in items_t1:
                        for b in items_t2:
                            if a != b:
                                patterns.append({
                                    "antecedent": a,
                                    "consequent": b,
                                    "gap_days": gap,
                                    "customer_id": cid,
                                })

    if not patterns:
        return pd.DataFrame()

    df_patterns = pd.DataFrame(patterns)

    # Calculate sequential support and lift
    pattern_counts = df_patterns.groupby(["antecedent", "consequent"]).agg(
        seq_count=("customer_id", "nunique"),
        avg_gap=("gap_days", "mean"),
    ).reset_index()

    pattern_counts["seq_support"] = pattern_counts["seq_count"] / n_customers

    # Baseline probability of purchasing consequent
    item_customers = merged.groupby("product_name")["customer_id"].nunique()
    baseline = item_customers / n_customers

    pattern_counts["baseline_prob"] = pattern_counts["consequent"].map(baseline)
    pattern_counts["temporal_lift"] = pattern_counts["seq_support"] / (
        pattern_counts["baseline_prob"] * pattern_counts["antecedent"].map(
            lambda x: item_customers.get(x, 1) / n_customers
        )
    )

    # Filter
    pattern_counts = pattern_counts[
        (pattern_counts["seq_support"] >= min_seq_support) &
        (pattern_counts["temporal_lift"] > 1.5)
    ].sort_values("temporal_lift", ascending=False)

    return pattern_counts


def compute_kpis(transactions_df, items_df, customers_df):
    """Compute key performance indicators."""
    total_transactions = len(transactions_df)
    total_revenue = transactions_df["total_amount"].sum()
    avg_basket_value = transactions_df["total_amount"].mean()
    avg_basket_size = transactions_df["n_items"].mean()

    # Items per department
    dept_revenue = items_df.groupby("department")["price"].sum()

    # Cross-sell rate (transactions with both Rx and non-Rx)
    txn_depts = items_df.groupby("transaction_id")["department"].apply(set)
    cross_sell = txn_depts.apply(lambda x: "Rx" in x and len(x) > 1).mean()

    # Unique customers
    active_customers = transactions_df["customer_id"].nunique()

    # Segment distribution
    seg_dist = customers_df["segment"].value_counts(normalize=True)

    return {
        "total_transactions": total_transactions,
        "total_revenue": total_revenue,
        "avg_basket_value": avg_basket_value,
        "avg_basket_size": avg_basket_size,
        "cross_sell_rate": cross_sell,
        "active_customers": active_customers,
        "dept_revenue": dept_revenue,
        "segment_distribution": seg_dist,
    } 