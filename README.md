# Scikit-Models-Random-Forest-SGD-Classifier

Supervised Machine Learning methods to classify short conversations extracted from Reddit
- 8 Classes based on conversation topics (Hockey, Movies, NBA, News, NFL, Politics, Soccer and WorldNews)

Data Cleaning and Feature Extraction:
- Label Encoding (Fit and Transform) & Decoding (Inverse Transform) using Scikit-Learn Proceprocessing label Encoder
- Lemmatization (using WordNetLemmatizer) - NLTK Package (Done to increase the accuracy)
- Term Frequency-Inverse Document Frequency (TF-IDF) approach - feature weighting

Reason to go with Random Forest Classifier:
- Performance in handling large datasets with higher dimensionality
- Efficiency in handling missing data issues
- Methods to balance errors in case of imbalanced classses

Reason to go with Stochastic Gradient Descent (SGD) Classifier:
- Efficiency 
- Ease of implementation (numerous opprotunities for code tuning using hyper parameters)
