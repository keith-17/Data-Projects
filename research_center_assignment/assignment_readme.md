# Research Center Quality Classification — Machine Learning Assignment

## Overview

This assignment focuses on using **machine learning and data analysis** to classify research centers in the UK into quality categories based on their internal infrastructure and access to external healthcare services.

You are provided with **synthetic data** representing several research centers across UK cities. Each research center includes attributes such as internal facility count and nearby healthcare availability.  
Your goal is to perform **Exploratory Data Analysis (EDA)**, apply **feature selection**, build a **clustering model**, and create a **FastAPI endpoint** that can classify new research centers.

---

## Objectives

1. Understand the dataset and identify key indicators of research center quality.  
2. Perform **EDA** to uncover relationships and distributions.  
3. Use **feature selection** to identify the most influential variables.  
4. Apply an **unsupervised clustering algorithm (K-Means)** to categorize centers into tiers.  
5. Build a **FastAPI endpoint** that classifies a research center into one of the tiers.

---

## Dataset Description

You are provided with a single CSV file:

### 📘 File: `research_centers_simplified.csv`

Each row represents a synthetic research center and includes the following columns:

| Column | Description |
|--------|--------------|
| `researchCenterId` | Unique identifier for the research center |
| `researchCenterName` | Name of the research center |
| `city` | City name |
| `latitude`, `longitude` | Geographic coordinates |
| `internalFacilitiesCount` | Number of internal facilities (e.g., labs, testing units, workstations) |
| `hospitals_10km` | Number of hospitals within 10 km |
| `pharmacies_10km` | Number of pharmacies within 10 km |
| `facilityDiversity_10km` | Diversity index (0–1) representing how varied nearby facilities are |
| `facilityDensity_10km` | Approximate density of nearby facilities per area |

After model training, you will add:

| Column | Description |
|--------|--------------|
| `cluster` | Numeric cluster label assigned by your ML model |
| `qualityTier` | Assigned quality label — **Premium**, **Standard**, or **Basic** |

---

## Tasks

### 1. Exploratory Data Analysis (EDA)
Perform initial data exploration to understand patterns and variability.

**Key steps:**
- Check missing values and data consistency.
- Visualize facility counts, diversity, and density across cities.
- Explore correlations between numeric variables.
- Identify which variables seem most important for determining research center quality.

**Expected visualizations:**
- Histogram of internal facility counts.
- Scatter plots showing hospital and pharmacy access.
- Correlation heatmap of numeric columns.

---

### 2. Feature Selection
Use correlation analysis or variance-based filtering to determine which features are most relevant to research center quality.

**Suggested important features:**
- `internalFacilitiesCount`  
- `hospitals_10km`  
- `pharmacies_10km`  
- `facilityDiversity_10km`  
- `facilityDensity_10km`

Normalize these features using **StandardScaler** before applying clustering.

**Discussion points:**
- Why were these features selected?
- Which features have the highest correlation with overall facility diversity or quality?

---

### 3. Clustering Model
Build a **K-Means clustering model** to classify centers into three tiers:

- **Cluster 0 → Premium**
- **Cluster 1 → Standard**
- **Cluster 2 → Basic**

**Steps:**
1. Select the features identified in the previous step.  
2. Standardize the data.  
3. Apply **K-Means (k=3)** with proper random initialization.  
4. Calculate the **silhouette score** to evaluate clustering quality.  
5. Map cluster numbers to descriptive labels (`Premium`, `Standard`, `Basic`).

---

### 4. Model Interpretation
Analyze and explain how the clustering results reflect differences in research center quality:
- Which cluster has the highest internal facility counts and external healthcare access?
- Are high-quality centers concentrated in specific cities?
- Does diversity or density play a stronger role?

Include summary tables showing the average feature values for each cluster.

---

### 5. API Deployment (FastAPI)
Expose your trained clustering model as an API.

**Expected endpoint:**

`POST /predict`

**Example Input:**
```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
````

**Example Output:**

```json
{
  "predictedCategory": "Premium"
}
```

Your endpoint should:

* Load or train the model.
* Validate and scale input data.
* Return the predicted quality tier.

---

## Example Workflow

1. Load and clean the dataset
2. Perform EDA and visualize trends
3. Select key features
4. Standardize numeric columns
5. Train K-Means with 3 clusters
6. Evaluate using silhouette score
7. Assign readable cluster labels
8. Save clustered data and model
9. Create and test the FastAPI endpoint

---

## Deliverables

| File                              | Description                                                            |
| --------------------------------- | ---------------------------------------------------------------------- |
| `EDA_and_Model.ipynb`             | Jupyter notebook with EDA, feature selection, clustering, and analysis |
| `app.py`                          | FastAPI app exposing the trained clustering model                      |
| `research_centers.csv`            | Provided dataset                                                       |
| `requirements.txt`                | Python dependencies                                                    |
| `README.md`                       | This documentation                                                     |

---

## Evaluation Criteria

| Area                     | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| **EDA**                  | Clear insights, relevant plots, and good interpretations |
| **Feature Selection**    | Logical and justified feature choice                     |
| **Model Implementation** | Correct use of K-Means and evaluation                    |
| **Interpretability**     | Clear explanation of clusters and differences            |
| **API**                  | Functional endpoint returning correct predictions        |
| **Code Quality**         | Clean, well-structured, PEP8-compliant code              |
| **Comments**             | Justified steps and reasoning in notebook                |

---

## Topics for Discussion

* Which features had the greatest influence on clustering?
* What patterns were visible in the data during EDA?
* Why is clustering a good approach for this problem?
* How would you improve this model if real data were available?
* How can the endpoint be extended for continuous retraining?
* Bonus Point - How to commercialise and scale the solution?

---

## Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn app:app --reload
```

---

## Suggested `requirements.txt`

```
fastapi
uvicorn
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Suggested Folder Structure

Mandatory file structure, as follows.

```
📂 research-center-assignment/
│
├── research_centers.csv
├── EDA_and_Model.ipynb
├── app.py
├── requirements.txt
└── README.md
```

Bonus Point - Optional files that can be included, as follows.

```
📂 research-center-assignment/
│
├── ...
├── DockerFile
├── docker-compose.yaml
├── .env.draft
├── .gitignore
└── .dockerignore
```

---

## Test Submission

Please push all the above metioned files on your **Git Profile**, make the project `Public` and **share the link** via email.

---

This assignment evaluates your ability to:

* Analyze and interpret structured datasets.
* Select relevant features and justify your choices.
* Build and evaluate an unsupervised ML model (K-Means).
* Explain your reasoning clearly.
* Deploy the model with FastAPI for real-time classification.
* Bonus Point - If you could dockerise the solution this is definately worth extra points.

Focus on **clarity, interpretability, and reasoning**, not complexity.
Your best submissions will show strong understanding and concise, well-documented work.

---
