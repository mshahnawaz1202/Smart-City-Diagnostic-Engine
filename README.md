


# Smart City Diagnostic Engine

## Overview

The Smart City Diagnostic Engine is a modular environmental analytics system designed to detect pollution anomalies, model extreme hazard events, and analyze high-dimensional urban air quality data.

This project simulates a real-world Smart City initiative where multi-gigabyte air quality data from 100 global sensor nodes (hourly recordings for 2025) is processed to extract actionable environmental insights.

The system is built as a fully reproducible Python pipeline and deployed via an interactive Streamlit dashboard.

---

## Dataset

**Source:** OpenAQ Global Air Quality API  
(Data downloaded and stored locally as csv)

### Variables Used

- PM2.5  
- PM10  
- NO2  
- Ozone  
- Temperature  
- Humidity  
- Timestamp  
- Station ID  
- Zone (Industrial / Residential)  
- Region  
- Population Density  

### Dataset Characteristics

- 100 sensor nodes  
- Hourly observations  
- Full year (2025)  
- Multi-dimensional environmental features  

---

## Project Objectives

The diagnostic engine addresses four analytical challenges:

1. Dimensionality Reduction and Zone Clustering  
2. High-Density Temporal Violation Analysis  
3. Distribution Modeling and Tail Risk Assessment  
4. Visual Integrity and Data Representation Audit  

---

## Repository Structure

```

Smart-City-Diagnostic-Engine/
│
├── data/
│   └── air_quality_2025.xlsx
│
├── data_loader.py
├── task1_pca.py
├── task2_temporal.py
├── task3_distribution.py
├── task4_visual_audit.py
├── main.py
└── app.py

```

---

## Task 1: Dimensionality Reduction

**Method Used:** Principal Component Analysis (PCA)

### Process
- Standardized six environmental variables
- Reduced six dimensions into two principal components
- Analyzed PCA loadings
- Visualized clustering between Industrial and Residential zones

### Outcome
- Identified dominant pollution drivers
- Interpreted which variables contribute most to zone separation

---

## Task 2: High-Density Temporal Analysis

**Objective:**  
Identify simultaneous PM2.5 health threshold violations (PM2.5 > 35 µg/m³).

### Approach
- Converted violations into binary format
- Created a Station × Time pivot matrix
- Visualized using a heatmap to avoid overplotting

### Outcome
- Detected periodic pollution signatures
- Distinguished daily vs seasonal patterns

---

## Task 3: Distribution Modeling & Tail Integrity

**Objective:**  
Quantify extreme hazard events (PM2.5 > 200 µg/m³).

### Approach
- Histogram for peak detection
- Log-scale complementary CDF (CCDF) for tail preservation
- Computed 99th percentile

### Outcome
- Accurate modeling of rare hazardous pollution events
- Honest representation of long-tail risk

---

## Task 4: Visual Integrity Audit

### Proposal Evaluated
3D Bar Chart (Pollution vs Population Density vs Region)

### Decision
Rejected due to:
- Perspective distortion
- High Lie Factor
- Low data-ink ratio

### Alternative Implemented
Small multiples (2D scatter plots per region) using sequential color encoding for perceptual accuracy.

---

## Technology Stack

- Python  
- NumPy  
- Pandas  
- Scikit-Learn  
- Matplotlib  
- Streamlit  

The project is implemented entirely using modular `.py` scripts (no notebooks).

---

## Installation

Clone the repository:

```

git clone [https://github.com/mshahnawaz1202/Smart-City-Diagnostic-Engine.git](https://github.com/mshahnawaz1202/Smart-City-Diagnostic-Engine.git)
cd Smart-City-Diagnostic-Engine

```

Install dependencies:

```

pip install numpy pandas matplotlib scikit-learn streamlit openpyxl

```

---

## Running the Pipeline

Run full analysis:

```

python main.py

```

---

## Running the Interactive Dashboard

Launch the Streamlit interface:

```

streamlit run app.py

```

---

## Design Principles Followed

- High data-ink ratio  
- No 3D effects  
- No unnecessary visual clutter  
- Perceptually accurate color scales  
- Modular and reproducible architecture  

---

## Key Learning Outcomes

- High-dimensional data projection using PCA  
- Loadings interpretation for environmental drivers  
- High-density time-series visualization  
- Long-tail distribution modeling  
- Percentile-based hazard quantification  
- Ethical and principled data visualization  
- Scalable analytics pipeline design  

---

## Author

**Shahnawaz**  

---

