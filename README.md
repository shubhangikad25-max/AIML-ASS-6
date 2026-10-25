# customer-segmentation-clustering
# Customer Segmentation Using K-Means and Agglomerative Clustering

## 1. Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques. The Mall Customers dataset is used to group customers based on their annual income and spending score.

The project implements two clustering algorithms, K-Means and Agglomerative Clustering, and compares their performance using clustering evaluation metrics and visualisations.

## 2. Problem Statement

Businesses need to understand customer behaviour to improve marketing strategies and provide suitable offers. This project uses clustering techniques to identify groups of customers with similar income and spending patterns.

## 3. Dataset

- **Dataset:** Mall Customers
- **Number of records:** 200
- **Number of columns:** 5
- **Features used:** Annual Income and Spending Score
- **Type of learning:** Unsupervised Learning

The dataset contains customer information such as gender, age, annual income, and spending score. Annual income and spending score were selected for clustering.

## 4. Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 5. Algorithms Implemented

### K-Means Clustering

K-Means divides the dataset into a specified number of clusters by grouping customers based on similarity and distance from cluster centres.

### Agglomerative Clustering

Agglomerative Clustering is a hierarchical clustering algorithm that starts with individual data points and progressively combines similar groups.

## 6. Methodology

1. Load the Mall Customers dataset.
2. Inspect the dataset and check for missing values.
3. Select annual income and spending score as features.
4. Standardise the selected features using `StandardScaler`.
5. Apply the Elbow Method to examine suitable cluster counts.
6. Implement K-Means and Agglomerative Clustering for cluster counts from 2 to 10.
7. Evaluate the models using Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Score.
8. Visualise and compare the resulting customer segments.

## 7. Results

Both algorithms achieved their highest Silhouette Scores among the tested configurations at five clusters.

| Evaluation Metric | K-Means | Agglomerative |
|---|---:|---:|
| Number of clusters | 5 | 5 |
| Silhouette Score | 0.5547 | 0.5538 |
| Davies–Bouldin Index | 0.5722 | 0.5779 |
| Calinski–Harabasz Score | 248.6493 | 244.4103 |

K-Means performed marginally better across the three reported evaluation metrics. However, the difference between the two algorithms was very small.

## 8. Key Findings

- Customer groups with different income and spending patterns were identified.
- Standardisation helped make the selected numerical features comparable for distance-based clustering.
- Both algorithms produced useful customer segments.
- K-Means achieved a slightly higher Silhouette Score and a slightly lower Davies–Bouldin Index.
- The results demonstrate how clustering can support customer analysis and marketing decisions.

## 9. Repository Contents

- `Customer_Segmentation.ipynb` — Python code, model implementation, evaluation, and visualisations.
- `Mall_Customers.csv` — Dataset used for the analysis.
- `README.md` — Project description, methodology, and results.

*Note: The repository contents listed above should match the files actually uploaded.*

## 10. Conclusion

This project demonstrates the application of unsupervised machine learning for customer segmentation. K-Means and Agglomerative Clustering were implemented and evaluated on the Mall Customers dataset. Both methods identified meaningful customer groups, with K-Means showing a marginal advantage according to the evaluation metrics.

The project helped develop practical understanding of data preprocessing, feature standardisation, clustering algorithms, model evaluation, and visualisation using Python.
