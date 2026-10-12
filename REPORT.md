# Customer Churn Prediction - Project Report

## 1. Problem
A telecom company loses about 26.5% of its customers. The goal is to predict which customers
are likely to churn, so the retention team can contact them with an offer before they leave.

## 2. Data
- Telco Customer Churn dataset: 7,043 customers, 21 columns (contract, tenure, services, charges, payment).
- Data issue found: `TotalCharges` stored as text with 11 blank values (all new customers, tenure 0) - converted and set to 0.
- 20% test set (1,409 customers) held out with stratification before any analysis and used once at the end.

## 3. Method
- EDA on training data: churn is highest for month-to-month contracts (43%), first-year customers (47%),
  fiber optic users (42%) and electronic-check payers (46%).
- 5 new features: tenure group, average monthly spend, charge increase, number of services, automatic payment.
- One Pipeline (imputation, scaling, one-hot encoding, model) so preprocessing is fitted only on training folds - no leakage.
- Compared Logistic Regression, Random Forest and XGBoost with 5-fold CV, class weights for the imbalance,
  and RandomizedSearchCV tuning (scoring = ROC-AUC). All three tied within 0.01 AUC; **Random Forest (tuned)** chosen
  (CV ROC-AUC 0.848).
- Metric: ROC-AUC, because the business ranks customers by risk; F1 also reported.
- Threshold 0.535 chosen to catch at least 75% of churners with the fewest calls.

## 4. Results (held-out test set, used once)
| ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|
| 0.844 | 0.638 | 0.547 | 0.767 |

- Caught 287 of 374 churners; missed 87. 238 calls went to customers who would have stayed.
- Calling the riskiest 37% of customers reaches 77% of churners -
  about twice as many as calling the same number of random customers.
- Test results match cross-validation, so the model is not overfitted.

## 5. Limitations
- Single historical snapshot; customer behaviour and prices may change, so the model needs monitoring and retraining.
- About 45% of flagged customers would not have churned - offers to them have a cost.
- Feature importance shows which factors matter, not proof that they cause churn.
- No data on customer complaints, service quality or competitor offers, which likely drive churn too.

## 6. Recommendations
1. Move month-to-month customers to 1- or 2-year contracts with a discount or perk (2-year churn is only 3%).
2. Strengthen onboarding and check-ins in the first 12 months, when churn risk is highest.
3. Offer free trials of online security / tech support - customers without them churn more.
4. Encourage automatic payment (manual payers churn about twice as often).
5. Run `predict_new.py` monthly on the active customer list and give the call list to the retention team;
   adjust the threshold to match the team's call capacity.
