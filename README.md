# Text Classification using Logistic Regression and SGDClassifier  

This project uses Python and scikit-learn to perform text classification on a dataset of reviews. The goal is to predict the sentiment of a review based on its text.  

## Dataset  
The dataset used in this project is stored in a CSV file `train.csv` in the `data` directory. The dataset contains reviews with their corresponding sentiment labels.  

## Approach  
The project uses two approaches to perform text classification:  

- **Logistic Regression**: A manual implementation of logistic regression is used to train a model on the dataset.  
- **SGDClassifier**: A pre-trained `SGDClassifier` from scikit-learn is used to train a model on the dataset.  

## Features  
- **Text preprocessing**: Reviews are converted to lowercase and tokenized using TF-IDF vectorization.  
- **Model training**: Models are trained on the preprocessed data using logistic regression and SGDClassifier.  
- **Model evaluation**: Models are evaluated using accuracy score and F1-score.  

## Requirements  
- Python 3.12+  
- scikit-learn  
- pandas  
- numpy  

## Usage  
1. Clone the repository and navigate to the project directory.  
2. Install the required dependencies using `pip install -r requirements.txt`.  
3. Run the project using `python main.py`.  

## Results  
The project prints the accuracy and F1-score of both models on the training and testing datasets.  

