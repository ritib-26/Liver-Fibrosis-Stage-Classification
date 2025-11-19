# Liver Fibrosis Stage Classification

This project uses machine learning to classify liver fibrosis stages from ultrasound histopathology images.

## Dataset

The dataset contains ultrasound images of liver tissue, categorized into five fibrosis stages (F0–F4).  
**Note:** The raw images are not included in this repository due to size constraints.

### How to get the data

1. Download the dataset from https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images.  
2. Extract the downloaded files.  
3. Place the extracted folders 'F0', 'F1', F2', 'F3' into the `data/raw` directory so that the folder structure looks like this:

Liver-Fibrosis-Stage-Classification/
├── data/
│ ├── raw/
│ │ ├── F0/
│ │ │ ├── image1.jpg
│ │ │ └── ...
│ │ ├── F1/
│ │ │ └── ...
│ │ ├── F2/
│ │ │ └── ...
│ │ └── F3/
│ │ └── ...
│ └── processed/
├── src/
│ ├── dataloaders.py
│ ├── models.py
│ ├── train.py
│ └── utils.py
├── notebooks/
│ ├── EDA.ipynb
│ └── baseline_model.ipynb
├── results/
│ ├── metrics/
│ └── visualizations/
└── README.md

### Notes

- The `data/raw` folder is **ignored by Git**, so you don’t need to worry about committing large image files.  
- Only the folder structure and scripts are tracked; the actual images should be downloaded separately.


