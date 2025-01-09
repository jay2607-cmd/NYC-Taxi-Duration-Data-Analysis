# NYC Taxi Trip Duration - Big Data Analysis

## Conducted a big data analysis on NYC taxi trip records using a Spark cluster

This project aimed to analyze a large dataset utilizing the distributed computing framework Apache Spark and develop a machine learning model to predict trip duration based on key input features.

---

### I. Introduction
New York City (NYC) is one of the largest cities globally, with a population of approximately 18 million people. Due to its dense population, traffic congestion is a persistent issue. Taxis are a popular mode of transportation within the city. In this project, we focused on predicting taxi trip durations using the "NYC taxi trip duration dataset" to help users estimate travel times and plan trips more effectively.

The dataset, released by the NYC Taxi and Limousine Commission, contains more than 510,000 records, including pickup times, geographic coordinates, passenger counts, and other variables. Many machine learning models have been built on this dataset as part of a Kaggle challenge, but their accuracy has room for improvement. Most models use TensorFlow and convolutional neural networks. In contrast, we explored data manipulation and model development using Scikit-learn and PySpark, aiming to enhance accuracy and reliability.

We created visualizations to explore relationships between trip duration and other variables, analyzing patterns in trip count, distance, and duration across different weekdays. Additionally, we built three machine learning models for insights and trip duration predictions: linear regression, gradient-boosted trees, and K-means clustering.

---

### II. Data Preprocessing
Data preparation involved several key steps:
- Each column's data type was converted to appropriate formats such as integer, double, string, or timestamp.
- Distances between pickup and drop-off coordinates were computed for all trips.
- Trips with durations outside the range of 3 minutes to 2 hours were filtered out, assuming typical NYC taxi trips do not exceed two hours.
- Trips beyond 60 kilometers were excluded, as taxis typically avoid long-distance routes.
- Additional preprocessing steps were implemented to refine the dataset for modeling.

---

### III. Data Visualization
Visualizations provided insights into the data:
- Total trips peak on Fridays and Saturdays, while average passenger counts are highest on Sundays, likely due to family outings.
- Trip durations are generally shorter on weekends, reflecting reduced traffic, though distances tend to be longer for leisure travel.

Key findings include:
- Distance is a critical variable for trip duration predictions.
- About 70% of trips cover less than 10 kilometers and take under 30 minutes (2000 seconds).
- Outliers were observed in some trips with unusually high durations and distances.

---

### IV. Algorithms

#### Outlier Removal
Outlier removal was crucial for effective modeling. Categorical variables, like weekdays, were indexed, and features were assembled using VectorAssembler. The target column was `trip_duration`, and the data was split into 70% for training and 30% for testing.

#### I. Regression Models

- **Linear Regression**: Built using PySpark's ML library, optimized with hyperparameters such as `regParam`, `elasticNetParam`, `maxIter`, and `fitIntercept` using `ParamGridBuilder` and `CrossValidator`.

- **Gradient-Boosted Trees Regression**: A supervised learning algorithm minimizing prediction errors by iteratively adjusting residuals. Parameters like `maxDepth`, `maxIter`, and `maxBins` were tuned, yielding higher accuracy than linear regression.

#### II. K-Means Clustering
Using scikit-learn's `KMeans`, we grouped similar data points. The Elbow method determined the optimal number of clusters (k = 4). Cluster analysis provided insights into feature influences on trip duration, visualized with box plots and radar charts.

---

### V. Results

#### I. Linear Regression
- Accuracy: 50.23%
- RMSE: 500.76

#### II. Gradient-Boosted Trees Regression
- Accuracy: ~75%
- RMSE: 362.1

#### III. K-Means Clustering
Despite differentiation challenges, clustering revealed valuable patterns and visualized key characteristics using radar plots.

---

### VI. Discussion

While the gradient-boosted trees model demonstrated promising results with 75% accuracy, further improvement is possible:
- Longer training times limited hyperparameter tuning.
- Access to robust computational resources will enhance model performance.

#### Future Work
1. **Cloud-Based Resources**: Utilize AWS or GCP for extensive model tuning and scalability.
2. **Real-Time Data Integration**: Incorporate live data through APIs and continuous learning pipelines for better predictive accuracy.

---

### Conclusion
This project provides valuable insights into NYC taxi trips and highlights opportunities for predictive modeling in urban transportation.
