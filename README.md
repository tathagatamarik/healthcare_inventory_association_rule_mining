# Market Basket Analysis — Pharmacy & Healthcare Retail

## Prescription & OTC Cross-Purchase Optimization

A comprehensive Market Basket Analysis (MBA) project demonstrating how advanced pattern mining and customer analytics can unlock revenue growth in pharmacy retail. This project includes both a detailed case study document and an interactive Streamlit dashboard for live demonstration.

---

##  Project Structure

```
├── case_study/
│   └── market_basket_analysis.pdf     # Comprehensive case study (22 pages)
├── streamlit_app/
│   ├── app.py                         # Main Streamlit application
│   ├── generate_data.py               # Synthetic data generator
│   ├── requirements.txt               # Python dependencies
│   ├── .streamlit/
│   │   └── config.toml                # Dark theme configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── analysis.py                # Core analysis functions
│   └── data/                          # Generated synthetic data (auto-created)
│       ├── transactions.csv
│       ├── transaction_items.csv
│       ├── customers.csv
│       └── products.csv
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (auto-runs on first launch)

```bash
python generate_data.py
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

##  Dashboard Pages

### 1. Executive Dashboard
- KPI overview (transactions, revenue, basket metrics)
- Revenue by department breakdown
- Weekly revenue trends
- Top products by frequency
- Customer segment distribution

### 2. Association Rules
- Apriori & FP-Growth algorithm comparison
- Interactive rules table with sorting and filtering
- Support vs. Confidence scatter plot (sized by Lift)
- Product pair heatmap
- Algorithm performance benchmarking

### 3. Network Graph
- Interactive product association network
- Color-coded by department (Rx, OTC, Supplements, Personal Care)
- Node size proportional to connectivity
- Most connected products analysis

### 4. Temporal Patterns
- Sequential purchase pattern mining
- Temporal lift analysis
- Gap distribution visualization
- Sankey diagram of purchase flows

### 5. Customer Segments
- RFM (Recency, Frequency, Monetary) analysis
- K-Means clustering with 3D visualization
- Silhouette score optimization
- Cluster vs. segment mapping

### 6. Bundle Simulator
- Select seed products and discover associated items
- Dynamic bundle pricing calculator
- Association strength visualization

---

##  Algorithms Implemented

| Algorithm | Purpose | Library |
|-----------|---------|---------|
| **Apriori** | Association rule mining | mlxtend |
| **FP-Growth** | Scalable frequent itemset mining | mlxtend |
| **Temporal Association Mining** | Sequential purchase patterns | Custom (PrefixSpan-inspired) |
| **K-Means + RFM** | Customer segmentation | scikit-learn |

---

##  Case Study Highlights

- **Client**: National pharmacy chain — 1,200+ locations, 45M+ annual transactions
- **Challenge**: Declining basket value, 18% Rx-to-OTC attach rate, missed cross-sell opportunities
- **Results**:
  - 23% average basket value increase
  - 340% cross-sell conversion improvement
  - 5× health bundle adoption increase
  - **$18.7M incremental annual revenue** (6-month payback)

---

##  Configuration

Adjust analysis parameters via the sidebar:
- **Analysis Level**: Product (SKU), Category, or Department
- **Min Support**: Minimum transaction frequency threshold
- **Min Confidence**: Minimum rule reliability threshold
- **Min Lift**: Minimum association strength threshold

---

##  Data Description

The synthetic dataset simulates a pharmacy retail chain with:
- **5,000 customers** across 6 behavioral segments
- **60 products** spanning Rx, OTC, Supplements, and Personal Care
- **~150,000 transactions** over 12 months
- **Engineered co-purchase patterns** reflecting real pharmacy associations (e.g., diabetes medications → glucose monitors)
- **Seasonal effects** (cold/flu season, allergy season, summer wellness)

---

##  License

This project is provided for demonstration and educational purposes.