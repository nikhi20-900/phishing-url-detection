# Phishing URL Detection

Machine learning project that classifies URLs as phishing or legitimate using the [PhiUSIIL](https://www.sciencedirect.com/science/article/pii/S0167404824000266) phishing URL dataset.

## Dataset

File: `data/PhiUSIIL_Phishing_URL_Dataset.csv`

| Property | Value |
| --- | --- |
| Rows | 235,795 |
| Features | 56 |
| Legitimate (`label = 1`) | 134,850 (57.2%) |
| Phishing (`label = 0`) | 100,945 (42.8%) |

Features cover URL structure (length, TLD, obfuscation, HTTPS), page content signals (title, favicon, forms, iframes), and resource counts (images, CSS, JS, internal/external links).

## Project layout

```
app/        # application / inference UI (planned)
data/       # PhiUSIIL dataset
models/     # trained model artifacts
notebook/   # exploratory analysis
src/        # feature extraction and training code (planned)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Open the analysis notebook:

```bash
jupyter notebook notebook/01_dataset_analysis.ipynb
```

## Current status

Dataset analysis is in `notebook/01_dataset_analysis.ipynb`. Training, saved models, and a prediction app are the next steps.
