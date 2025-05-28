# Price Elasticity Supermarket Dataset

This project explores **price elasticity** using real-world supermarket sales data to find the **optimal price point** that maximizes profit under various market conditions.

## 📌 Objective

Main Objective : `Can we know for each product the best price we can put to get the maximum profit?`

How we will do?

Build a predictive model that helps estimate optimal pricing by forecasting quantity sold under three scenarios:
- **Worst Case**
- **Likely Case**
- **Best Case**

## 🧠 Methodology

- **Model**: [`pyGAM`](https://pygam.readthedocs.io/en/latest/) with `ExpectileGAM`, which allows capturing distribution asymmetry to simulate different market responses.
- **Prediction**: We use three expectiles (0.25, 0.5, 0.75) to forecast quantities for worst, likely, and best case outcomes respectively.
- **Profit Calculation**:

$$
\text{Total Profit} = \text{Quantity} \times \left( \text{Unit Selling Price} - \frac{\text{Wholesale Price}}{1 - \text{Loss Rate}} \right)
$$

## 📊 Dataset

- **Source**: [Kaggle - Supermarket Sales Data](https://www.kaggle.com/datasets/yapwh1208/supermarket-sales-data/data)
- **Region**: China
- **Currency**: RMB
- **Records**: 878,503 rows
- **Period**: 2020 - 2023

## 🚀 Deployment

This project is deployed using **Streamlit**, allowing interactive exploration of pricing scenarios and predicted profitability.


🚀 Deployment
This project is deployed using Streamlit, allowing interactive exploration of pricing scenarios and predicted profitability.

👉 Live Demo: TBD
🛠 Code: TBD
