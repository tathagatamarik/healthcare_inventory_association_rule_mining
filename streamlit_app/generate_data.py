"""
Synthetic Pharmacy Transaction Data Generator
Generates realistic pharmacy retail transaction data for Market Basket Analysis demo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

np.random.seed(42)

# ============================================================
# PRODUCT CATALOG
# ============================================================

PRODUCT_CATALOG = {
    # Prescription Medications (Rx)
    "Rx": {
        "Metformin 500mg": {"category": "Diabetes Rx", "price": 15.00, "base_freq": 0.08},
        "Metformin 1000mg": {"category": "Diabetes Rx", "price": 22.00, "base_freq": 0.05},
        "Lisinopril 10mg": {"category": "Blood Pressure Rx", "price": 12.00, "base_freq": 0.07},
        "Amlodipine 5mg": {"category": "Blood Pressure Rx", "price": 14.00, "base_freq": 0.05},
        "Atorvastatin 20mg": {"category": "Statin Rx", "price": 18.00, "base_freq": 0.09},
        "Simvastatin 40mg": {"category": "Statin Rx", "price": 16.00, "base_freq": 0.04},
        "Omeprazole 20mg": {"category": "GI Rx", "price": 20.00, "base_freq": 0.06},
        "Levothyroxine 50mcg": {"category": "Thyroid Rx", "price": 11.00, "base_freq": 0.05},
        "Sertraline 50mg": {"category": "SSRI Rx", "price": 14.00, "base_freq": 0.04},
        "Fluoxetine 20mg": {"category": "SSRI Rx", "price": 13.00, "base_freq": 0.03},
        "Cetirizine Rx 10mg": {"category": "Allergy Rx", "price": 10.00, "base_freq": 0.06},
        "Montelukast 10mg": {"category": "Allergy Rx", "price": 25.00, "base_freq": 0.03},
        "Prenatal Vitamins Rx": {"category": "Prenatal Rx", "price": 18.00, "base_freq": 0.02},
        "Albuterol Inhaler": {"category": "Respiratory Rx", "price": 35.00, "base_freq": 0.04},
        "Gabapentin 300mg": {"category": "Pain Rx", "price": 16.00, "base_freq": 0.03},
    },
    # OTC Medications
    "OTC": {
        "Ibuprofen 200mg": {"category": "Pain Relief OTC", "price": 8.99, "base_freq": 0.12},
        "Acetaminophen 500mg": {"category": "Pain Relief OTC", "price": 7.99, "base_freq": 0.10},
        "Diphenhydramine 25mg": {"category": "Allergy OTC", "price": 9.49, "base_freq": 0.06},
        "Loratadine 10mg": {"category": "Allergy OTC", "price": 11.99, "base_freq": 0.07},
        "Saline Nasal Spray": {"category": "Allergy OTC", "price": 6.99, "base_freq": 0.05},
        "Antihistamine Eye Drops": {"category": "Allergy OTC", "price": 12.49, "base_freq": 0.04},
        "Cold & Flu Relief": {"category": "Cold/Flu OTC", "price": 10.99, "base_freq": 0.08},
        "Cough Syrup DM": {"category": "Cold/Flu OTC", "price": 9.99, "base_freq": 0.06},
        "Antacid Tablets": {"category": "GI OTC", "price": 7.49, "base_freq": 0.07},
        "Laxative Gentle": {"category": "GI OTC", "price": 8.99, "base_freq": 0.03},
        "Hydrocortisone Cream 1%": {"category": "Skin OTC", "price": 6.99, "base_freq": 0.04},
        "Melatonin 5mg": {"category": "Sleep OTC", "price": 8.99, "base_freq": 0.05},
        "Glucose Monitor Strips": {"category": "Diabetes OTC", "price": 29.99, "base_freq": 0.03},
        "Blood Pressure Monitor": {"category": "BP Monitor", "price": 49.99, "base_freq": 0.01},
        "Digital Thermometer": {"category": "Diagnostics", "price": 12.99, "base_freq": 0.02},
    },
    # Health Supplements
    "Supplements": {
        "Vitamin D 2000IU": {"category": "Vitamins", "price": 11.99, "base_freq": 0.09},
        "Vitamin C 1000mg": {"category": "Vitamins", "price": 9.99, "base_freq": 0.08},
        "Multivitamin Daily": {"category": "Vitamins", "price": 14.99, "base_freq": 0.07},
        "Omega-3 Fish Oil": {"category": "Supplements", "price": 16.99, "base_freq": 0.06},
        "CoQ10 100mg": {"category": "Supplements", "price": 22.99, "base_freq": 0.03},
        "Probiotic Daily": {"category": "Supplements", "price": 19.99, "base_freq": 0.05},
        "Calcium + D3": {"category": "Supplements", "price": 12.99, "base_freq": 0.04},
        "Iron Supplement": {"category": "Supplements", "price": 8.99, "base_freq": 0.03},
        "Folic Acid": {"category": "Supplements", "price": 7.99, "base_freq": 0.02},
        "Magnesium 400mg": {"category": "Supplements", "price": 10.99, "base_freq": 0.04},
        "Zinc 50mg": {"category": "Supplements", "price": 7.49, "base_freq": 0.04},
        "Turmeric Curcumin": {"category": "Supplements", "price": 18.99, "base_freq": 0.03},
        "Selenium 200mcg": {"category": "Supplements", "price": 9.49, "base_freq": 0.02},
        "B-Complex": {"category": "Vitamins", "price": 11.49, "base_freq": 0.04},
        "Elderberry Extract": {"category": "Immune", "price": 14.99, "base_freq": 0.04},
    },
    # Personal Care & Wellness
    "Personal Care": {
        "Hand Sanitizer": {"category": "Hygiene", "price": 3.99, "base_freq": 0.10},
        "Facial Tissues Box": {"category": "Tissues", "price": 2.99, "base_freq": 0.09},
        "Lip Balm SPF": {"category": "Skin Care", "price": 4.49, "base_freq": 0.06},
        "Sunscreen SPF 50": {"category": "Sun Care", "price": 12.99, "base_freq": 0.04},
        "Moisturizing Lotion": {"category": "Skin Care", "price": 8.99, "base_freq": 0.05},
        "Toothpaste Whitening": {"category": "Oral Care", "price": 5.99, "base_freq": 0.07},
        "Mouthwash Antiseptic": {"category": "Oral Care", "price": 6.99, "base_freq": 0.05},
        "Bandages Assorted": {"category": "First Aid", "price": 5.49, "base_freq": 0.04},
        "Diabetic Socks": {"category": "Diabetes Care", "price": 9.99, "base_freq": 0.02},
        "Compression Stockings": {"category": "Circulation", "price": 19.99, "base_freq": 0.01},
        "Low-Sodium Seasoning": {"category": "Heart Health Food", "price": 5.99, "base_freq": 0.02},
        "Sugar-Free Candy": {"category": "Diabetic Snacks", "price": 4.49, "base_freq": 0.03},
        "Protein Bar Variety": {"category": "Nutrition", "price": 2.99, "base_freq": 0.06},
        "Electrolyte Drink Mix": {"category": "Hydration", "price": 7.99, "base_freq": 0.04},
        "Sleep Mask & Earplugs": {"category": "Sleep Wellness", "price": 8.99, "base_freq": 0.02},
    },
}

# ============================================================
# ASSOCIATION RULES (engineered co-purchase patterns)
# ============================================================

ASSOCIATION_RULES = [
    # Diabetes cluster
    (["Metformin 500mg"], ["Glucose Monitor Strips"], 0.35),
    (["Metformin 500mg"], ["Diabetic Socks"], 0.20),
    (["Metformin 500mg"], ["Sugar-Free Candy"], 0.25),
    (["Metformin 1000mg"], ["Glucose Monitor Strips"], 0.40),
    (["Glucose Monitor Strips"], ["Diabetic Socks"], 0.30),
    # Blood pressure cluster
    (["Lisinopril 10mg"], ["Blood Pressure Monitor"], 0.15),
    (["Lisinopril 10mg"], ["Low-Sodium Seasoning"], 0.22),
    (["Amlodipine 5mg"], ["Low-Sodium Seasoning"], 0.20),
    (["Amlodipine 5mg"], ["Omega-3 Fish Oil"], 0.18),
    (["Blood Pressure Monitor"], ["Low-Sodium Seasoning"], 0.28),
    # Statin cluster
    (["Atorvastatin 20mg"], ["CoQ10 100mg"], 0.30),
    (["Atorvastatin 20mg"], ["Omega-3 Fish Oil"], 0.25),
    (["Simvastatin 40mg"], ["CoQ10 100mg"], 0.28),
    (["CoQ10 100mg"], ["Omega-3 Fish Oil"], 0.35),
    # Allergy cluster
    (["Cetirizine Rx 10mg"], ["Saline Nasal Spray"], 0.38),
    (["Cetirizine Rx 10mg"], ["Antihistamine Eye Drops"], 0.30),
    (["Loratadine 10mg"], ["Saline Nasal Spray"], 0.32),
    (["Loratadine 10mg"], ["Facial Tissues Box"], 0.40),
    (["Saline Nasal Spray"], ["Antihistamine Eye Drops"], 0.25),
    (["Diphenhydramine 25mg"], ["Melatonin 5mg"], 0.20),
    # Cold/Flu cluster
    (["Cold & Flu Relief"], ["Facial Tissues Box"], 0.50),
    (["Cold & Flu Relief"], ["Vitamin C 1000mg"], 0.35),
    (["Cold & Flu Relief"], ["Cough Syrup DM"], 0.30),
    (["Cold & Flu Relief"], ["Elderberry Extract"], 0.22),
    (["Cough Syrup DM"], ["Facial Tissues Box"], 0.38),
    (["Vitamin C 1000mg"], ["Elderberry Extract"], 0.25),
    (["Vitamin C 1000mg"], ["Zinc 50mg"], 0.28),
    # GI cluster
    (["Omeprazole 20mg"], ["Antacid Tablets"], 0.30),
    (["Omeprazole 20mg"], ["Probiotic Daily"], 0.22),
    (["Antacid Tablets"], ["Probiotic Daily"], 0.18),
    # Thyroid cluster
    (["Levothyroxine 50mcg"], ["Selenium 200mcg"], 0.20),
    (["Levothyroxine 50mcg"], ["B-Complex"], 0.18),
    # Mental health cluster
    (["Sertraline 50mg"], ["Melatonin 5mg"], 0.25),
    (["Sertraline 50mg"], ["Magnesium 400mg"], 0.18),
    (["Fluoxetine 20mg"], ["Melatonin 5mg"], 0.22),
    # Prenatal cluster
    (["Prenatal Vitamins Rx"], ["Folic Acid"], 0.45),
    (["Prenatal Vitamins Rx"], ["Iron Supplement"], 0.35),
    (["Prenatal Vitamins Rx"], ["Calcium + D3"], 0.30),
    (["Folic Acid"], ["Iron Supplement"], 0.32),
    # Wellness combos
    (["Vitamin D 2000IU"], ["Calcium + D3"], 0.30),
    (["Vitamin D 2000IU"], ["Multivitamin Daily"], 0.22),
    (["Multivitamin Daily"], ["Omega-3 Fish Oil"], 0.20),
    (["Hand Sanitizer"], ["Facial Tissues Box"], 0.25),
    (["Ibuprofen 200mg"], ["Acetaminophen 500mg"], 0.15),
    (["Ibuprofen 200mg"], ["Bandages Assorted"], 0.12),
    (["Sunscreen SPF 50"], ["Lip Balm SPF"], 0.30),
    (["Sunscreen SPF 50"], ["Moisturizing Lotion"], 0.22),
    (["Toothpaste Whitening"], ["Mouthwash Antiseptic"], 0.35),
    # Respiratory
    (["Albuterol Inhaler"], ["Vitamin C 1000mg"], 0.20),
    (["Albuterol Inhaler"], ["Elderberry Extract"], 0.15),
]

# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

CUSTOMER_SEGMENTS = {
    "Chronic Loyalists": {
        "proportion": 0.22,
        "rx_prob": 0.85, "visit_freq_range": (3, 6),
        "basket_size_range": (3, 7),
        "preferred_categories": ["Rx", "OTC", "Supplements"],
        "spend_multiplier": 1.4,
    },
    "Health Seekers": {
        "proportion": 0.18,
        "rx_prob": 0.30, "visit_freq_range": (2, 5),
        "basket_size_range": (3, 6),
        "preferred_categories": ["Supplements", "Personal Care"],
        "spend_multiplier": 1.2,
    },
    "Rx-Only": {
        "proportion": 0.25,
        "rx_prob": 0.95, "visit_freq_range": (3, 5),
        "basket_size_range": (1, 2),
        "preferred_categories": ["Rx"],
        "spend_multiplier": 0.7,
    },
    "Occasional Visitors": {
        "proportion": 0.20,
        "rx_prob": 0.15, "visit_freq_range": (1, 2),
        "basket_size_range": (2, 4),
        "preferred_categories": ["OTC", "Personal Care"],
        "spend_multiplier": 0.9,
    },
    "Price Sensitives": {
        "proportion": 0.10,
        "rx_prob": 0.40, "visit_freq_range": (2, 3),
        "basket_size_range": (2, 4),
        "preferred_categories": ["OTC", "Personal Care"],
        "spend_multiplier": 0.6,
    },
    "At-Risk Lapsing": {
        "proportion": 0.05,
        "rx_prob": 0.50, "visit_freq_range": (0, 1),
        "basket_size_range": (1, 3),
        "preferred_categories": ["Rx", "OTC"],
        "spend_multiplier": 0.8,
    },
}


def generate_data(n_customers=5000, n_months=12, output_dir="data"):
    """Generate synthetic pharmacy transaction data."""
    os.makedirs(output_dir, exist_ok=True)

    # Build flat product list
    all_products = {}
    for dept, products in PRODUCT_CATALOG.items():
        for name, info in products.items():
            all_products[name] = {**info, "department": dept}

    product_names = list(all_products.keys())

    # Generate customers
    customers = []
    segment_names = list(CUSTOMER_SEGMENTS.keys())
    segment_probs = [CUSTOMER_SEGMENTS[s]["proportion"] for s in segment_names]

    for cid in range(1, n_customers + 1):
        segment = np.random.choice(segment_names, p=segment_probs)
        seg_info = CUSTOMER_SEGMENTS[segment]
        age = np.random.randint(22, 78)
        gender = np.random.choice(["M", "F"], p=[0.45, 0.55])
        loyalty_member = np.random.choice([True, False], p=[0.75, 0.25])

        customers.append({
            "customer_id": f"C{cid:05d}",
            "segment": segment,
            "age": age,
            "gender": gender,
            "loyalty_member": loyalty_member,
            "monthly_visits": np.random.randint(*seg_info["visit_freq_range"]),
            "basket_size_mean": np.mean(seg_info["basket_size_range"]),
            "rx_prob": seg_info["rx_prob"],
            "spend_multiplier": seg_info["spend_multiplier"],
            "preferred_categories": seg_info["preferred_categories"],
        })

    # Generate transactions
    transactions = []
    transaction_items = []
    tid = 0
    start_date = datetime(2024, 3, 1)

    for cust in customers:
        n_visits = cust["monthly_visits"] * n_months
        if n_visits == 0:
            n_visits = np.random.randint(1, 4)

        for visit in range(n_visits):
            tid += 1
            # Random date within period
            day_offset = np.random.randint(0, n_months * 30)
            txn_date = start_date + timedelta(days=int(day_offset))

            # Season factor
            month = txn_date.month
            seasonal_boost = {}
            if month in [11, 12, 1, 2]:  # Cold/flu season
                seasonal_boost = {"Cold/Flu OTC": 2.5, "Immune": 2.0, "Tissues": 1.8, "Vitamins": 1.3}
            elif month in [3, 4, 5]:  # Allergy season
                seasonal_boost = {"Allergy OTC": 2.5, "Allergy Rx": 1.8}
            elif month in [6, 7, 8]:  # Summer
                seasonal_boost = {"Sun Care": 2.5, "Hydration": 2.0, "Skin Care": 1.5}

            # Determine basket size
            basket_size = max(1, int(np.random.normal(cust["basket_size_mean"], 1.2)))
            basket_size = min(basket_size, 10)

            # Select items
            basket = []

            # Add Rx item if applicable
            if np.random.random() < cust["rx_prob"]:
                rx_products = [p for p, info in all_products.items() if info["department"] == "Rx"]
                rx_weights = [all_products[p]["base_freq"] for p in rx_products]
                rx_weights = np.array(rx_weights) / sum(rx_weights)
                rx_item = np.random.choice(rx_products, p=rx_weights)
                basket.append(rx_item)

            # Fill remaining basket
            remaining = basket_size - len(basket)
            for _ in range(remaining):
                # Weighted selection based on preferences and seasonality
                weights = []
                for p in product_names:
                    if p in basket:
                        weights.append(0)
                        continue
                    w = all_products[p]["base_freq"]
                    # Category preference
                    if all_products[p]["department"] in cust["preferred_categories"]:
                        w *= 2.0
                    # Seasonal boost
                    cat = all_products[p]["category"]
                    if cat in seasonal_boost:
                        w *= seasonal_boost[cat]
                    weights.append(w)

                if sum(weights) == 0:
                    break
                weights = np.array(weights) / sum(weights)
                item = np.random.choice(product_names, p=weights)
                basket.append(item)

            # Apply association rules to add correlated items
            for antecedents, consequents, prob in ASSOCIATION_RULES:
                if all(a in basket for a in antecedents):
                    for c in consequents:
                        if c not in basket and np.random.random() < prob:
                            basket.append(c)

            # Remove duplicates
            basket = list(dict.fromkeys(basket))

            # Calculate total
            total = sum(all_products[item]["price"] * cust["spend_multiplier"] for item in basket)

            txn_record = {
                "transaction_id": f"T{tid:07d}",
                "customer_id": cust["customer_id"],
                "date": txn_date.strftime("%Y-%m-%d"),
                "hour": np.random.choice(range(8, 21), p=[0.05, 0.08, 0.10, 0.12, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]),
                "total_amount": round(total, 2),
                "n_items": len(basket),
                "store_id": f"S{np.random.randint(1, 51):03d}",
            }
            transactions.append(txn_record)

            for item in basket:
                transaction_items.append({
                    "transaction_id": f"T{tid:07d}",
                    "product_name": item,
                    "department": all_products[item]["department"],
                    "category": all_products[item]["category"],
                    "price": round(all_products[item]["price"] * cust["spend_multiplier"], 2),
                    "quantity": 1,
                })

    # Create DataFrames
    df_transactions = pd.DataFrame(transactions)
    df_items = pd.DataFrame(transaction_items)
    df_customers = pd.DataFrame([{
        "customer_id": c["customer_id"],
        "segment": c["segment"],
        "age": c["age"],
        "gender": c["gender"],
        "loyalty_member": c["loyalty_member"],
    } for c in customers])

    # Build product reference
    df_products = pd.DataFrame([
        {"product_name": name, "department": info["department"], "category": info["category"], "price": info["price"]}
        for name, info in all_products.items()
    ])

    # Save
    df_transactions.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    df_items.to_csv(os.path.join(output_dir, "transaction_items.csv"), index=False)
    df_customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    df_products.to_csv(os.path.join(output_dir, "products.csv"), index=False)

    print(f"Generated {len(df_transactions)} transactions with {len(df_items)} line items")
    print(f"Customers: {len(df_customers)} | Products: {len(df_products)}")
    print(f"Date range: {df_transactions['date'].min()} to {df_transactions['date'].max()}")

    return df_transactions, df_items, df_customers, df_products


if __name__ == "__main__":
    generate_data(n_customers=5000, n_months=12, output_dir="data")