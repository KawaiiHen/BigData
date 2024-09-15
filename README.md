# BigData
**Massive Dataset Processing with PySpark and Hadoop**

This project demonstrates the use of PySpark and Hadoop to process massive datasets through a series of tasks that cover various machine learning and data processing techniques. The project leverages distributed computing with PySpark on top of Hadoop's HDFS for scalable, fault-tolerant data storage and processing.

## Prerequisites
- **Apache Hadoop** (HDFS, YARN)
- **Apache Spark** with PySpark
- **Python** (with required libraries: `pandas`, `numpy`, `matplotlib`)
- Dataset files (e.g., MNIST, stock prices, etc.)

## Table of Contents
- [Task 1: Clustering with KMeans](#task-1-clustering-with-kmeans)
- [Task 2: Dimensionality Reduction with SVD](#task-2-dimensionality-reduction-with-svd)
- [Task 3: Recommendation System with Collaborative Filtering](#task-3-recommendation-system-with-collaborative-filtering)
- [Task 4: Stock Price Regression](#task-4-stock-price-regression)
- [Task 5: Multi-class Classification with MLP](#task-5-multi-class-classification-with-mlp)
- [Hadoop's Role in the Project](#hadoop-role-in-the-project)
- [Setup Instructions](#setup-instructions)

---

## Task 1: Clustering with KMeans
This task applies **KMeans clustering** to the MNIST dataset, a collection of handwritten digits. The task involves:
- Loading the dataset and processing it into a form suitable for clustering.
- Applying the KMeans algorithm with different values of `k` (clusters), such as 5, 10, and 15.
- Visualizing the data and clustering results using bar charts to show how `k` impacts clustering performance.
  
KMeans is used to group similar digits based on pixel intensity, allowing for an understanding of the inherent grouping structure in the dataset.

## Task 2: Dimensionality Reduction with SVD
This task focuses on applying **Singular Value Decomposition (SVD)** to reduce the dimensionality of the MNIST dataset:
- The original 784-dimensional data (representing pixels) is reduced to 196 dimensions.
- This reduces the computational load and improves efficiency while retaining significant information.
- The reduced data is then saved for later classification tasks.

Dimensionality reduction helps simplify complex datasets while maintaining the core structure for further analysis.

## Task 3: Recommendation System with Collaborative Filtering
This task uses **Collaborative Filtering (ALS Algorithm)** to build a recommendation system:
- The input is a dataset of user-item ratings.
- The ALS model is trained to predict ratings for items not yet rated by users.
- It evaluates the performance of the recommendation system using Mean Squared Error (MSE).

This task simulates a movie or product recommendation system, where personalized recommendations are generated based on user preferences.

## Task 4: Stock Price Regression
This task applies **Linear Regression** to predict future stock prices based on past data:
- Using historical stock prices, the model is trained on data prior to July 2022.
- The model is then tested on post-June 2022 data to predict stock trends.
- Mean Squared Error (MSE) is computed for both the training and test sets, allowing evaluation of the model's accuracy.

Regression analysis helps forecast future values, making it highly useful for financial predictions.

## Task 5: Multi-class Classification with MLP
This task implements a **Multilayer Perceptron (MLP)** for multi-class classification using the MNIST dataset:
- The dataset is used to train an MLP neural network with layers `[784, 64, 10]`.
- The model is trained on the original and SVD-reduced datasets.
- Random Forest and SVM models are also applied for comparison.
- Model performance is evaluated using accuracy on both training and test sets.

This task explores deep learning techniques for recognizing handwritten digits and compares performance across different models.

## Hadoop Role in the Project
Hadoop serves as the foundation for scalable and fault-tolerant data processing:
- **HDFS (Hadoop Distributed File System)**: Provides distributed storage for large datasets like MNIST and stock price data. It allows data to be split across multiple nodes, facilitating parallel processing.
- **YARN (Yet Another Resource Negotiator)**: Enables distributed resource management for Spark jobs, allowing the system to scale and efficiently utilize computational resources.
- **Fault Tolerance**: Hadoop ensures that in the event of node failure, the data and tasks are replicated, avoiding interruptions in processing large datasets.

While PySpark is the primary processing tool, Hadoop’s underlying architecture provides the necessary infrastructure to efficiently handle the large-scale computations and data storage required by this project.

## Setup Instructions
1. **Install Hadoop**: Follow instructions to set up a Hadoop cluster with HDFS and YARN.
2. **Install Spark**: Ensure PySpark is installed and properly configured to work with your Hadoop setup.
3. **Run PySpark Jobs**:
    - For each task, navigate to the relevant script and run the code using the PySpark environment.
    - Example: `spark-submit task1_kmeans.py`
4. **Evaluate Results**:
    - View the bar charts and printed metrics (MSE, accuracy) to analyze the performance of each task.

## Conclusion
This project demonstrates the use of PySpark for performing various machine learning tasks on massive datasets, facilitated by Hadoop’s distributed computing and storage capabilities. The combination of clustering, dimensionality reduction, recommendation systems, regression, and classification techniques illustrates the power of big data processing with Spark and Hadoop.
