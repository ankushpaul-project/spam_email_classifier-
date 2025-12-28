# Spam Email Detection Using Machine Learning

Name: Ankush Paul  
Roll Number: [125CR0036]

## Introduction
Email spam is a common problem where unwanted or irrelevant messages are sent in large numbers. These emails waste time and can sometimes contain harmful or misleading information. The purpose of this project is to build a machine learning based system that can automatically classify emails as Spam or Not Spam based on their content.

This project helps in understanding how text data can be handled using Natural Language Processing (NLP) techniques and how machine learning models can be applied to solve real-world problems.

## Objectives of the Project
The main objectives of this project are:
- To understand how text data differs from numerical data
- To learn and apply text preprocessing techniques
- To convert text data into numerical features
- To train a machine learning classifier on email data
- To evaluate the performance of the model using suitable metrics

## Dataset Description
The dataset used in this project is a publicly available spam email dataset obtained from Kaggle. The dataset contains a collection of email messages along with their corresponding labels.

Each email is labeled as:
- `spam` – unwanted or promotional emails
- `ham` – legitimate or normal emails

The dataset is first examined and cleaned by removing unnecessary columns before being used for training.

## Data Preprocessing
Since machine learning models cannot directly work with raw text, preprocessing is a crucial step. The following preprocessing steps are performed on the email text:

1. **Lowercasing**  
   All text is converted to lowercase to ensure uniformity.

2. **Removal of Punctuation and Numbers**  
   Special characters, punctuation marks, and digits are removed as they do not contribute meaningfully to email classification.

3. **Tokenization**  
   The email text is split into individual words (tokens).

4. **Stopword Removal**  
   Common English words such as “is”, “the”, “and” are removed since they occur frequently and do not add significant meaning.

These steps help reduce noise in the data and improve the performance of the classifier.

## Feature Extraction
After preprocessing, the cleaned text is converted into numerical form using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization.

TF-IDF assigns higher importance to words that appear frequently in a particular email but less frequently across all emails. This helps the model focus on important words rather than common ones.

TF-IDF is widely used in text classification problems and works well with email datasets.

## Algorithm Used
The machine learning algorithm used in this project is **Multinomial Naive Bayes**.

### Reason for Choosing Multinomial Naive Bayes
- It is simple and easy to implement
- It performs well on text classification tasks
- It is efficient for high-dimensional sparse data like TF-IDF features
- It is commonly used for spam detection problems

## Model Training
The dataset is divided into training and testing sets using an 80:20 split.  
The training data is used to train the Multinomial Naive Bayes classifier, and the testing data is used to evaluate the model on unseen emails.

## Model Evaluation
To evaluate the performance of the classifier, the following metrics are used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The **F1 Score** is considered the most important metric as it provides a balance between precision and recall, which is especially useful in classification problems where class distribution may not be perfectly balanced.

## Results
After training and evaluation, the model achieves:
- **Accuracy:** approximately 97%
- **F1 Score:** approximately 0.92

These results indicate that the model performs well in distinguishing spam emails from legitimate ones.

## Testing on New Emails
A prediction function is implemented to test the trained model on new and unseen email messages.  
The function takes an email message as input and outputs whether the email is classified as Spam or Not Spam.

This helps demonstrate the practical usability of the model.

## Project Structure
The repository contains the following files:
- `email_spam_classifier.ipynb` – Jupyter Notebook with the complete implementation
- `spam.csv` – Dataset file (if included in the repository)
- `requirements.txt` – List of Python libraries required to run the project
- `README.md` – Documentation of the project

## How to Run the Project
1. Clone the GitHub repository to your local machine.
2. Install the required dependencies using:
   pip install -r requirements.txt
3. Open the Jupyter Notebook `email_spam_classifier.ipynb`.
4. Run all the cells in sequence to train and evaluate the model.

## Learning Sources
The following resources were referred to while completing this project:
- Corey Schafer (YouTube) – Python and scikit-learn tutorials
- StatQuest (YouTube) – Understanding Naive Bayes, TF-IDF, and evaluation metrics
- Krish Naik (YouTube) – NLP and email spam classification tutorials
- Kaggle datasets and notebooks

