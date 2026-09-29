# Bank Customer Churn Prediction (Neural Network)

Predicts whether a bank customer will leave the bank using a small feed-forward neural network built with TensorFlow/Keras.

## Dataset
`Churn_Modelling.csv`: 10,000 bank customers, 14 columns (credit score, geography, gender, age, tenure, balance, number of products, card/active-member flags, estimated salary, and the target `Exited`).
About 20% of customers churned (2,037 of 10,000), so the classes are imbalanced.
The file is not included in this repo. Download the Kaggle "Churn Modelling" dataset and place the CSV next to the notebook.

## Approach
1. Checked shape, dtypes, duplicates and class balance
2. Dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`)
3. One-hot encoded `Geography` and `Gender`
4. 80/20 train/test split (`random_state=42`)
5. Standardised features with `StandardScaler` (fit on train only)
6. Trained a Keras `Sequential` model: Dense(3, sigmoid) -> Dense(1, sigmoid), binary cross-entropy, Adam, 10 epochs, 20% validation split

## Results
| Metric | Value |
|---|---|
| Test accuracy | 80.35% |

Accuracy is close to what you get by always predicting "no churn" (~80%), so this first model is a baseline rather than a finished result. The notebook ends with a section that adds a majority-class baseline, confusion matrix, precision/recall/F1 and ROC-AUC.

## Next steps
- Handle class imbalance (class weights or SMOTE) and tune the decision threshold
- Try a larger network, ReLU activations, early stopping
- Compare with logistic regression, random forest and gradient boosting

## Run it
```bash
git clone https://github.com/<your-username>/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt
jupyter notebook bank-churn-prediction.ipynb
```

## Tech stack
Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib

## License
MIT
