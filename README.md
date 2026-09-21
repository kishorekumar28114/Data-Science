# 📸 Instagram Analytics & Predictive Engagement Modeling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MU30Q_FT3psXgr6fAwotEek2c_5j_ged?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data analytics and machine learning pipeline that transforms normalized social media relational data into creator engagement metrics, classifies high-activity accounts, and forecasts per-post audience interactions.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Dataset & Data Model](#-dataset--data-model)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Pipelines](#-machine-learning-pipelines)
  - [Module A: Influencer Classification](#module-a-influencer-classification)
  - [Module B: Per-Post Engagement Forecasting](#module-b-per-post-engagement-forecasting)
- [Analytics & Visualizations](#-analytics--visualizations)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Key Engineering Insights & Limitations](#-key-engineering-insights--limitations)
- [Project Documentation](#-project-documentation)

---

## 🔍 Project Overview

Social platforms generate event-stream relational data across profiles, uploads, reactions, and follows. Analyzing isolated tables fails to reveal who creates impactful content, who consumes it, or how audience size dictates interaction velocity.

This project solves this by:
1. **Relational Data Integration**: Performing multi-table left joins across 7 normalized entities to construct unified user- and creator-level feature profiles.
2. **Behavioral Segmentation (Classification)**: Identifying accounts exceeding median platform engagement thresholds using `RandomForestClassifier`.
3. **Engagement Forecasting (Regression)**: Predicting expected likes and comments per post from an account’s follower footprint and posting volume using dual `RandomForestRegressor` models.
4. **Diagnostic Analytics**: Evaluating model mechanics via confusion matrices and mapping follower-to-interaction distributions.

---

## 🏗 System Architecture & Pipeline

```mermaid
flowchart LR
    subgraph Data Sources [Raw Relational CSVs]
        U[users.csv]
        P[photos.csv]
        L[likes.csv]
        C[comments.csv]
        F[follows.csv]
    end

    subgraph Feature Engineering [Pandas Processing]
        AggA[User Actions Aggregation]
        AggB[Creator Outcomes Aggregation]
        Clean[Outlier Trimming & Zero Fill]
    end

    subgraph Modeling [Scikit-Learn ML]
        RFC[RandomForestClassifier\n(Influencer Status)]
        RFR1[RandomForestRegressor\n(Likes per Post)]
        RFR2[RandomForestRegressor\n(Comments per Post)]
    end

    subgraph Outputs [Interactive & Visualizations]
        CM[Confusion Matrix]
        Scatter[Engagement Scatter Plots]
        Predict[Inference Prompts]
    end

    U & L & C --> AggA --> Clean --> RFC --> CM
    U & P & L & C & F --> AggB --> Clean --> RFR1 & RFR2 --> Scatter
    RFC & RFR1 & RFR2 --> Predict
```

---

## 🗄 Dataset & Data Model

The project operates on an Instagram-style relational schema consisting of **7 tables**:

| Table | Records | Key Columns | Analytical Role |
| :--- | :--- | :--- | :--- |
| **`users.csv`** | 100 | `id`, `username`, `created_at` | Primary user entity and base table for left merges. |
| **`photos.csv`** | 257 | `id`, `user_id`, `image_url` | Maps published content back to the creator. |
| **`likes.csv`** | 1,000 | `user_id`, `photo_id`, `created_at` | Measures user liking actions and creator likes received. |
| **`comments.csv`** | 1,000 | `id`, `user_id`, `photo_id`, `comment_text` | Tracks commentary actions and creator discussion volume. |
| **`follows.csv`** | 1,000 | `follower_id`, `followee_id` | Derives audience reach and account popularity. |
| **`photo_tags.csv`** | 501 | `photo_id`, `tag_id` | Content categorization links (available for extension). |
| **`tags.csv`** | 21 | `id`, `tag_name` | Unique hashtags index. |

---

## ⚙️ Feature Engineering

To transform transactional rows into predictive features, the pipeline enforces strict data integrity rules:

1. **Left Merges with Zero-Imputation (`fillna(0)`)**:
   - Preserves all 100 users, including dormant accounts with 0 likes, comments, or posts.
2. **Directional Engagement Separation**:
   - *Activity Features (Outbound)*: Actions initiated by a user (liking/commenting on others) &rarr; feeds the **Classifier**.
   - *Creator Features (Inbound)*: Reactions earned on user-owned photos &rarr; feeds the **Regressors**.
3. **Division-by-Zero Guard**:
   $$\text{likes\per\post} = \frac{\text{likes}}{\text{posts} + 1}, \quad \text{comments\per\post} = \frac{\text{comments}}{\text{posts} + 1}$$
4. **Outlier Mitigation**:
   - Applies independent 99th-percentile trimming on interaction rates to prevent skew from viral anomalies.

---

## 🤖 Machine Learning Pipelines

### Module A: Influencer Classification
* **Objective**: Segment users whose combined activity (`total_likes + total_comments`) places them above the platform's median engagement threshold.
* **Input Features ($X$)**: `total_likes`, `total_comments`
* **Target ($y$)**: Binary flag ($1$ if engagement > median, else $0$)
* **Algorithm**: `RandomForestClassifier` (80/20 train-test split, `random_state=42`)
* **Evaluation**: 
  - Test Accuracy: `1.00`
  - Macro F1-Score: `1.00` (Support: Class 0: 12, Class 1: 8)
  - *Engineering Note*: As the target is derived from the threshold of the input features, this 100% score validates code pipeline correctness rather than a claim of general real-world influencer identification.

### Module B: Per-Post Engagement Forecasting
* **Objective**: Forecast the expected engagement rate a creator receives based on their profile scale.
* **Input Features ($X$)**: `followers`, `posts`
* **Models**:
  1. `likes_model`: `RandomForestRegressor` $\rightarrow$ predicts `likes_per_post`
  2. `comments_model`: `RandomForestRegressor` $\rightarrow$ predicts `comments_per_post`
* **Inference**: Accepts custom follower and post parameters and provides estimated engagement ranges.

---

## 📊 Analytics & Visualizations

The project generates diagnostic visualizations using `matplotlib`:

| Visualization | Description | Purpose |
| :--- | :--- | :--- |
| **Confusion Matrix** | Labeled True Negative, False Positive, False Negative, True Positive counts | Confirms classifier precision across test splits. |
| **Followers vs. Likes** | Scatter plot of audience count vs total likes received | Analyzes whether audience scale reliably drives higher reaction volume. |
| **Posts vs. Engagement** | Scatter plot of content quantity vs combined reactions | Identifies whether higher posting frequency yields compounding audience interaction. |

---

## 📁 Repository Structure

```text
├── Inputs/                          # Normalized relational datasets
│   ├── comments.csv
│   ├── follows.csv
│   ├── likes.csv
│   ├── photo_tags.csv
│   ├── photos.csv
│   ├── tags.csv
│   └── users.csv
├── output/
│   └── pdf/
│       └── Instagram_Analytics_Project_Guide.pdf  # Technical reference guide
├── build_project_pdf.py             # Script generating PDF technical documentation
├── instagram-analysis.ipynb         # Interactive Jupyter Notebook pipeline
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies
```

---

## 🚀 Quickstart Guide

### ⚡ 1-Click Interactive Cloud Run (No Local Setup)
You can directly run, experiment with, and execute the entire notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MU30Q_FT3psXgr6fAwotEek2c_5j_ged?usp=sharing)

---

### Local Installation & Execution

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DataScience-Instagram-Analytics.git
cd DataScience-Instagram-Analytics
```

#### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `reportlab`)*

#### 4. Run the Analysis
Launch the interactive Jupyter notebook:
```bash
jupyter notebook instagram-analysis.ipynb
```

Or recompile the PDF study guide:
```bash
python build_project_pdf.py
```

---

## 💡 Key Engineering Insights & Limitations

| Aspect | Current Implementation | Production Improvement |
| :--- | :--- | :--- |
| **Target Leakage** | Classification target is derived from input sums. | Formulate target using independent ground truth (e.g. verified badge or commercial sponsor metrics). |
| **Data Volume** | 100 user sample with 20 test records. | Scale to thousands of profiles with $k$-fold cross-validation. |
| **Regression Evaluation** | Models trained and demonstrated interactively. | Compute quantitative regression metrics ($MAE$, $RMSE$, and $R^2$). |
| **Content Features** | `tags.csv` and `photo_tags.csv` are currently unused. | Extract NLP hashtag embeddings and image sentiment/style markers. |

---

## 📄 Project Documentation

A comprehensive technical study guide and interview reference PDF is included under:
- [`output/pdf/Instagram_Analytics_Project_Guide.pdf`](output/pdf/Instagram_Analytics_Project_Guide.pdf)

---

## 👤 Author
- **Kishore Kumar**
- GitHub: [@kishorekumar28114](https://github.com/kishorekumar28114)
- LinkedIn: [Connect with me](https://www.linkedin.com)
