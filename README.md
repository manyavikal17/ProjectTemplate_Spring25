![](UTA-DataScience-Logo.png)

# Mushroom Classification Project

This repository holds an attempt to apply machine learning models to classify mushrooms as edible or poisonous using data from Kaggle. [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)

## Overview

This project classifies mushrooms as edible or poisonous using their physical attributes. Each entry describes a mushroom through 22 categorical features such as odor, cap shape, etc, with the target variable indicating edibility. The project frames this as a binary classification task. All categorical features were one-hot encoded, and the target was label-encoded. The dataset was split into training (60%), validation (20%), and test (20%) subsets. 

I trained both a Random Forest and Logistic Regression model, with Random Forest achieving 100% accuracy on the validation set. Through visualizations and model insights, I noticed that a few features,especially odor and gill-size, clearly separate edible from poisonous mushrooms. This highlighted how the classification problem is surprisingly trivial for machine learning due to the presence of a few highly informative features.

## Summary of Workdone


### Data

* Data:
  * Type: For example
    * Input: CSV file with 22 categorical features describing physical characteristics of mushrooms (e.g., cap shape, odor, gill color).
    * Output/Target: Categorical variable class indicating whether the mushroom is edible (e) or poisonous (p).
  * Size: Total of 8,124 mushroom samples.
  * Instances (Train, Test, Validation Split):
    * Training set: 4,874 samples (60%)
    * Validation set: 1,625 samples (20%)
    * Test set: 1,625 samples (20%)


#### Preprocessing / Clean up

* Dropped the veil-type column due to having only one unique value (no variance).
* Replaced '?' values in the stalk-root feature with a new category labeled 'missing'.
* Performed one-hot encoding on all categorical features to convert them into binary columns.
* Mapped the target variable class to binary values: 'e' - `'0'` (edible), 'p' - `'1'` (poisonous).

#### Data Visualization

* **Target Class Distribution**
  
  The bar plot of the target variable (class) shows that the dataset is fairly balanced, with 51.8% edible mushrooms and 48.2% poisonous ones. This balance is important because it means the model won’t be biased 
  toward predicting one class more often than the other, and accuracy will be a reliable evaluation metric.
  
  ![Screenshot 2025-05-01 111131](https://github.com/user-attachments/assets/7b886872-33ca-4d6a-b4ec-d05a76bb8be2)


* **Feature Distribution by Class**

  Based on the normalized stacked bar plots, we observe that the following features show clear patterns in relation to mushroom edibility:
  * Odor : `n`(none) and `l` (anise) are strongly associated with edible mushrooms, while `f` (foul) is highly indicative of poisonous mushrooms.
  * Gill-size: Mushrooms with `b` (broad) gills are mostly edible, while `n` (narrow) gills are linked to poisonous ones.
  * Bruises - The presence of bruises (`t`) is more common in edible mushrooms, while their absence (`f`) appears more in poisonous ones.
  * Spore-print-color: `k` and `n` are mostly found in edible mushrooms, whereas `w`  and `h` is seen more in poisonous mushrooms.

  These distinct patterns suggest that features like odor, gill-size, and bruises carry strong predictive signals for classifying mushrooms.
  ![top_features (2)](https://github.com/user-attachments/assets/a5cd9ba8-3462-4485-a25b-3b2b5232e5d4)


* **Feature Importance**

  Although this feature importance plot was generated after training the Random Forest model, it is included here as part of the data visualization process to highlight which features the model found most 
  predictive. It complements earlier visual patterns which confirms features like `odor_none`, `gill-size_b`, and `odor_foul` are among the strongest indicators of mushroom toxicity. This helps validate the 
  insights gained during exploratory analysis.

  ![feature_importance](https://github.com/user-attachments/assets/2216b9b3-fa46-42fd-968c-13bf20fe909c)


### Problem Formulation

* Define:
  * Input: One-hot encoded categorical features describing physical attributes of mushrooms.
  * Output: Binary classification label - `'0'` for edible and `'1'` for poisonous mushrooms.
* Models
  * Random Forest Classifier:
    I started with Random Forest because it performs well with categorical data and captures complex patterns. It also provides feature importance, which helps explain model decisions.
  * Logistic Regression:
    I used Logistic Regression as a simple and interpretable baseline model. Unlike more complex models, it is less prone to overfitting and helps evaluate whether a linear approach can effectively separate edible and poisonous mushrooms.
  * Hyperparameters:
    * Random Forest Classifier : `random_state=42`
    * Logistic Regression : `max_iter=1000` to ensure convergence, `random_state=42`

### Training

* How I Trained: I trained the models using Python and scikit-learn in a Jupyter Notebook environment on a standard laptop.
* Traning Time: Training was fast and completed within seconds for both models due to the dataset's small size and scikit-learn's efficiency.
* Stopping Criteria : Logistic Regression was configured with `max_iter=1000` to ensure convergence. Random Forest stopped automatically after building the default number of trees.
* Difficulties : Initially, Random Forest achieved perfect accuracy, which raised concerns about overfitting. To address this, I tested Logistic Regression as a simpler model and compared performance on a separate validation set.

### Performance Comparison

* Key Metric : I used accuracy as the primary performance metric, along with precision, recall, F1-score, and the confusion matrix for a deeper understanding of each model's behavior.
  
* Results Summary:
  
|  Model  | Accuracy | Precision | Recall | F1-Score |
| ------- | -------- | --------- | ------ | -------- |
| Random Forest Classifier  | 100% | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 99.75% | 1.00 | 1.00 | 1.00 |



* Visualization
  * Confusion Matrix for Random Forest
    
 ![Screenshot 2025-05-01 122741](https://github.com/user-attachments/assets/f7e8d4a5-4284-47c9-8bf9-0ea3a1173f2b)

  * Confusion Matrix for Logistic Regression
    
![Screenshot 2025-05-01 122752](https://github.com/user-attachments/assets/fd96a4da-1a63-4222-8834-288cacde0ecf)

### Conclusions

* Both models performed extremely well, with Random Forest achieving 100% accuracy and Logistic Regression closely behind at 99.75%.
* Features like `odor`, `gill-size`, and `spore-print-color` were the most influential in predicting edibility.
* This led to a key insight: Despite its real-world significance, the problem is trivial for machine learning due to the presence of a few very strong signals in the data.
* What began as a safety-focused classification task ended as a clear example of how even simple models can succeed when the data is this well-structured.

  
### Future Work

* Try tuning basic hyperparameters like the number of trees in Random Forest or regularization strength in Logistic Regression to improve model performance.
* Experiment with other decision tree-based models such as Gradient Boosting, XGBoost, or LightGBM to compare performance and generalization.
* Test the model on noisy or incomplete data to evaluate robustness in more realistic scenarios.
  

## How to reproduce results

You can reproduce the results of this project either locally or using any cloud-based environment that supports Jupyter notebooks (e.g. Google Colab)

Steps:
1. Download or clone this repository.
2. Open the notebook file (Mushroom_Classification_Project.ipynb) in your preferred environment
3. Download the dataset (mushrooms.csv) from the [Kaggle Mushroom Classification page](https://www.kaggle.com/datasets/uciml/mushroom-classification) and place it in the same directory as the notebook.
4. Run all cells to:
   * Load and clean the data
   * One-hot encode features
   * Train Random Forest and Logistic Regression models
   * Evaluate performance and generate visualizations

This project can run on most standard machines with Python installed. It does not require a GPU or TPU, as the dataset is small and the models are lightweight.

### Overview of files in repository

* `Mushroom_Classification_Project.ipynb`: Final notebook with full pipeline — data loading, preprocessing, visualization, model training, and evaluation.
* `Draft_1.ipynb`: Initial draft used for prototyping and testing early ideas.
* `submission.csv`: Prediction results formatted for Kaggle submission.


### Software Setup
* Required Packages:
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn


## Citations

* Mushroom Classification Dataset. UCI Machine Learning Repository.
  
Available on Kaggle: [https://www.kaggle.com/datasets/uciml/mushroom-classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
* [scikit-learn](https://scikit-learn.org/stable/index.html#)







