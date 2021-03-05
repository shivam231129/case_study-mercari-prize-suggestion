---

Suggesting the price of items for online platforms using Machine Learning

Table of Contents
Business Task
Data-set Description 
Mapping Machine Learning and Deep Learning
Evaluation Metric
Exploratory Data Analysis
Feature Engineering
Existing Solutions
My Experimentations
Final Model
Summary, results, and conclusions
Future Work
References

 "Success meant to be not about scoring but its about learning"

---

1. Business Task
The objective of this case study is to suggest an appropriate selling price to a seller who wishes to sell his/her product (usually pre-owned) on the online platform, Mercari, which connects the sellers to the buyers.The problem statement is about prediction of the price of various products may be clothes or electronic gadgets etc. Mercari is an organisation in japan which is use to sell goods online that goods can be new or used . For online sellers to decide the price of goods could be a sophisticated task.This may be useful for online sellers to automate the price deciding action and reduce the manpower . Lot of same type of product have varying price depends on their product brands,their conditions ,seasons etc and many other features we need is to come up with some feature engineering and ml model to get best output, that is precise price as possible.
Also two mobile phone with same specification have different price we need to come up with some ml model which consider all important features into consideration.This problem is regression problem we need to predict some values which are greater than zero and some real number.
This case study is based on the famous Kaggle Competition held in 2018: Mercari Price Suggestion Challenge(https://www.kaggle.com/c/mercari-price-suggestion-challenge)
The seller enters the details of the product he/she wishes to sell, like the product's name, a short description, category, brand, shipping status, and the condition of the product.When the seller enters these details, the Mercari platform returns an appropriate selling price of the given product to the user.So, the task here is to return the price based on the entered details.
