# Pediatric Hemodynamic Shock Prediction Challenge

## Project brief explanation
This project is made to Build a Model and Predict a patient whether he will experience shock in 6 hours based on the patient's Criteria. The model is built using the LGBM Algorithm with Stratified Cross Validation

## Dataset Explanation
This data is a clinical dataset from patients in the Pediatric Intensive Care Unit (PICU) and aims to predict the probability of shock within the next 6 hours. The target variable is shock_within_6h, which is a binary label. 

| Column Name            | Short Description                                         |
|------------------------|----------------------------------------------------------|
| patient_id            | Unique ID for each patient                               |
| age_months           | Patient's age in months                                  |
| weight_kg            | Patient's weight (kg)                                   |
| heart_rate           | Heart rate per minute                                   |
| blood_pressure_sys   | Systolic blood pressure                                |
| blood_pressure_dia   | Diastolic blood pressure                               |
| respiratory_rate     | Respiratory rate per minute                            |
| spo2                | Oxygen saturation (%)                                  |
| temperature_c       | Body temperature in Celsius                           |
| lactate_level       | Lactate level (metabolic indicator, usually mmol/L)   |
| capillary_refill_sec | Capillary refill time (blood perfusion indicator)    |
| gcs_score           | Glasgow Coma Scale score (patient consciousness, 3-15) |
| base_deficit        | Base deficit (blood acidity indicator, negative = acidosis) |
| urine_output_ml_hr  | Urine output per hour (ml/hr)                         |
| crt_level           | Creatinine (kidney function indicator)                 |
| bun_level          | Blood Urea Nitrogen (kidney function indicator)       |
| wbc_count          | White blood cell count                                 |
| hemoglobin         | Hemoglobin level                                       |
| platelet_count     | Platelet count                                         |
| bilirubin          | Bilirubin level (liver function indicator)             |
| pco2              | Partial CO2 pressure (respiratory indicator)          |
| shock_within_6h   | Target: 0 = no shock, 1 = shock within 6 hours

<br><br>

## Code Explanation

```python
train_data.head(5)
```

| patient_id  | age_months | weight_kg | heart_rate | blood_pressure_sys | blood_pressure_dia | respiratory_rate | spo2  | temperature_c | lactate_level | ... | base_deficit | urine_output_ml_hr | crt_level | bun_level | wbc_count | hemoglobin | platelet_count | bilirubin | pco2 | shock_within_6h |
|------------|-----------|-----------|------------|--------------------|--------------------|-----------------|------|--------------|--------------|-----|--------------|------------------|-----------|---------|----------|-----------|---------------|----------|------|----------------|
| PICU_H0FQ7T | 90        | 34.0      | 156.16     | 71.43              | 38.86              | 39.19           | 95.65 | 36.55         | 3.26         | ... | -1.15        | 0.60              | 0.66      | 8.83     | 14.11    | 12.12      | 268.66        | 1.04     | 28.15 | 0              |
| PICU_42MG9Q | 70        | 3.0       | 132.57     | 103.79             | 62.55              | 30.17           | 96.71 | 36.92         | 3.45         | ... | -0.29        | 1.54              | 0.88      | 11.61    | 6.94     | 11.88      | 417.41        | 1.21     | 60.62 | 0              |
| PICU_LMN9UY | 41        | 9.3       | 131.92     | 92.64              | 57.65              | 48.24           | 94.60 | 37.13         | 2.21         | ... | -1.79        | 1.54              | 0.48      | 16.26    | 8.63     | 11.18      | 162.75        | 1.11     | 29.55 | 0              |
| PICU_2JSK9V | 113       | 9.8       | 158.97     | 91.61              | 72.68              | 22.59           | 94.06 | 35.94         | 0.50         | ... | -3.16        | 1.88              | 0.40      | 17.93    | 11.85    | 12.30      | 281.54        | 0.70     | 43.75 | 0              |
| PICU_UVLSUH | 32        | 9.5       | 88.63      | 94.10              | 63.07              | 47.84           | 94.31 | 37.43         | 3.99         | ... | -7.04        | 1.95              | 0.86      | 8.87     | 8.37     | 6.00       | 182.91        | 0.65     | 42.53 | 0              |
<br>

```python
test_data.head(5)
```

| patient_id  | age_months | weight_kg | heart_rate | blood_pressure_sys | blood_pressure_dia | respiratory_rate | spo2  | temperature_c | lactate_level | ... | gcs_score | base_deficit | urine_output_ml_hr | crt_level | bun_level | wbc_count | hemoglobin | platelet_count | bilirubin | pco2 |
|------------|-----------|-----------|------------|--------------------|--------------------|-----------------|------|--------------|--------------|-----|------------|--------------|------------------|-----------|---------|----------|-----------|---------------|----------|------|
| PICU_IYINNC | 138       | 36.9      | 78.71      | 77.18              | 51.15              | 47.41           | 94.61 | 38.13         | 3.11         | ... | 7          | -3.73        | 0.19              | 0.83      | 10.23    | 6.64     | 15.55      | 266.24        | 0.26     | 44.34 |
| PICU_D7NKJF | 78        | 35.9      | 141.43     | 93.01              | 44.93              | 42.82           | 100.00 | 36.12         | 0.64         | ... | 11         | -1.30        | 0.33              | 0.64      | 10.77    | 8.03     | 11.16      | 306.49        | 0.80     | 30.69 |
| PICU_DYGH15 | 132       | 30.7      | 177.19     | 76.80              | 47.61              | 27.85           | 97.61 | 37.06         | 1.21         | ... | 6          | -1.12        | 1.62              | 0.37      | 12.61    | 3.35     | 15.94      | 382.17        | 0.86     | 43.19 |
| PICU_RP5NBF | 154       | 11.5      | 161.54     | 72.39              | 46.82              | 36.95           | 97.39 | 37.61         | 0.50         | ... | 4          | -0.37        | 1.17              | 0.72      | 11.00    | 18.95    | 9.38       | 120.62        | 0.42     | 30.85 |
| PICU_QJSWFK | 140       | 14.2      | 133.39     | 105.46             | 41.37              | 37.26           | 95.27 | 38.05         | 3.43         | ... | 9          | 1.09         | 1.61              | 0.86      | 12.45    | 11.76    | 12.28      | 237.12        | 0.63     | 25.89 |
<br>

### Check Correlation

```python
# CHECK CORRELATION

numerical_cols = train_df.select_dtypes(include='number').columns

matrix_corr = train_df[numerical_cols].corr(method='pearson')

plt.figure(figsize=(15,12))
sns.heatmap(data = matrix_corr, vmin = -1, vmax = 1, cmap='coolwarm', annot = True, fmt='.2f')
```
- Output :
  ![check_multicolinearity](https://github.com/user-attachments/assets/0c143f9c-64d4-4ecb-9a90-ba3c5bf412af)

In the code, we check the correlation of each independent feature whether there is multicollinearity or not. The result is that there are no features that are correlated with each other / multicollinearity 
<br><br>

```python
# CHECK CORRELATION TO TARGET FEATURE

matrix_corr_target = train_df[numerical_cols].corr(method='spearman')[['shock_within_6h']]

plt.figure(figsize=(4,10))
sns.heatmap(data = matrix_corr_target, vmin = -1, vmax = 1, cmap='coolwarm', annot = True, fmt='.2f')
```

- Output :
  ![heatmap_correlation](https://github.com/user-attachments/assets/afb3b953-f023-426d-aed0-af7d2a1e2915)

```blood_pressure_sys``` , ```age_months``` , ```urine_output_ml_hr```, ```base_deficit```, and ```lactate_level``` are the 5 features that have the most influence on the target feature (```shock_within_6h```)
<br><br>

### Check Distribution
```python
# CHECK DISTRIBUTION


numerical_cols = numerical_cols.drop(labels = 'shock_within_6h')

fig, axes = plt.subplots(nrows = 5, ncols = 4, figsize=(20,15))

for i, feature in enumerate(numerical_cols):
    sns.histplot(data = train_df[feature], color = 'lightblue', ax = axes[i%5, i//5])
    sns.histplot(data = test_df[feature], color = 'pink', ax = axes[i%5, i//5])

    axes[i%5, i//5].set_title(feature)

plt.show()
```
The code is used to check the distribution of each feature of the dataset. The result is that almost all features are normally distributed (except ```age_months``` and ```gcs_score```)

### Visualizing Scatter Plot
```python
x = train_df.drop(columns=['patient_id', 'shock_within_6h'])
y = train_df['shock_within_6h']

zscore = StandardScaler()
x_scaled = zscore.fit_transform(x)

pca = PCA(n_components = None)
data_reduced = pca.fit_transform(x_scaled)

# VISUALIZATION
plt.figure(figsize=(12,6))
plt.scatter(x = data_reduced[:, 0], y = data_reduced[:, 1], c = y)
```
The code is used to visualize the PCA (Principal Component Analysis) plot. But before that, we do ```StandardScaler()``` first so that the range of values ​​is uniform... because PCA is very sensitive to the range of values

```python
# UMAP

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs = -1)

# FIT UMAP
x_umap = reducer.fit_transform(x_scaled)

plt.figure(figsize=(12, 6))
plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, cmap='Spectral', s=10)
plt.colorbar()
plt.title('UMAP projection of the Digits dataset')
plt.show()
```
The code is used to Perform Dimensionality Reduction using UMAP. UMAP (**Uniform Manifold Approximation and Projection**) is a dimensionality reduction technique used to visualize complex data in 2D or 3D space.The purpose of performing scatter plot visualization is to gain insight if you want to use oversampling/undersampling techniques. 
<br><br>

### Modelling
```python
# SPLIT DATA

x = train_df.drop(columns=['shock_within_6h', 'patient_id'])
y = train_df['shock_within_6h']
test_df = test_df.drop(columns=['patient_id'])
```
Before doing training, we first determine the x and y variables. <br> <br>

```python
# KFOLD

skfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2025)
#kfold = KFold(n_splits = 10, shuffle = False)

oof_train, oof_val, test_pred = [], [], []

for i, (train_index, val_index) in enumerate(skfold.split(x,y)):

    # SPLIT
    x_train , x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val  = y.iloc[train_index], y.iloc[val_index]

    # LGBM
    model = lightgbm.LGBMClassifier(
            n_estimators = 15000,
            boosting_type='gbdt',
            is_unbalance = True,
            learning_rate=0.02,
            num_leaves=120,
            max_depth=15,
            colsample_bytree=0.4,
            max_bin=1024,
            random_state=42,
            force_col_wise= True,
            n_jobs=-1,
            device='cpu'
        )

    # TRAIN LGBM
    model.fit(x_train, y_train, eval_set=(x_val, y_val), 
              callbacks=[lightgbm.early_stopping(stopping_rounds = 100), 
                         lightgbm.log_evaluation(0), 
                         lightgbm.record_evaluation({})])

    # PREDICT
    y_pred_train = model.predict(x_train)
    y_pred_val   = model.predict(x_val)

    # CALCULATE F1-SCORE
    f1_train = f1_score(y_train, y_pred_train)
    f1_val   = f1_score(y_val, y_pred_val)

    print(f'Fold {i} : Train data F1 = {f1_train} , Val data F1 = {f1_val}')

    oof_train.append(f1_train)
    oof_val.append(f1_val)
    
    # PREDICTION TEST DATA
    y_pred_test = model.predict(test_df)
    test_pred.append(y_pred_test)

print(f'Overall Train data F1 : {np.mean(oof_train)}')
print(f'Overall Val data F1 : {np.mean(oof_val)}')
```
During training, we use **Stratified K-Fold Cross Validation** with 5 Splits to train the LGBM Model. The reason for using **Stratified K-Fold Cross Validation** is because the data is unbalanced.<br>
for LGBM, we use parameters like this:
```python
lightgbm.LGBMClassifier(
            n_estimators = 15000,
            boosting_type='gbdt',
            is_unbalance = True,
            learning_rate=0.02,
            num_leaves=120,
            max_depth=15,
            colsample_bytree=0.4,
            max_bin=1024,
            random_state=42,
            force_col_wise= True,
            n_jobs=-1,
            device='cpu'
        )
```
These are the parameters we define to train a model. <br> <br>

```python
model.fit(x_train, y_train, eval_set=(x_val, y_val), 
              callbacks=[lightgbm.early_stopping(stopping_rounds = 100), 
                         lightgbm.log_evaluation(0), 
                         lightgbm.record_evaluation({})])
```
After that we build a maximum of 15,000 trees and do **early stopping** (forcefully stopping the model) if there is no change for the better in building 100 trees during the training process. <br> <br>

```python
 # PREDICT
    y_pred_train = model.predict(x_train)
    y_pred_val   = model.predict(x_val)

    # CALCULATE F1-SCORE
    f1_train = f1_score(y_train, y_pred_train)
    f1_val   = f1_score(y_val, y_pred_val)

    print(f'Fold {i} : Train data F1 = {f1_train} , Val data F1 = {f1_val}')

    oof_train.append(f1_train)
    oof_val.append(f1_val)
    
    # PREDICTION TEST DATA
    y_pred_test = model.predict(test_df)
    test_pred.append(y_pred_test)
```
1. This code predicts ```train_data``` , ```val_data``` , and ```test_data``` using the **LightGBM** model that has been built/trained.
2. After that we calculate the *F-Score* value of ```train_data``` and ```val_data``` to assess how well the model is in classifying the data. *F-Score* is used because it is a metric that considers the balance between precision and recall, so it is very suitable for imbalanced datasets.
3. Then we save the F-Score value into a list, so that we will see the average of the overall F-Score

<br> <br>

```python
print(f'Overall Train data F1 : {np.mean(oof_train)}')
print(f'Overall Val data F1 : {np.mean(oof_val)}')
```
This code is used to display the overall average of the F-Score value. So for example there are 5 split cross validations, then there will be 5 F-Scores, then finally we divide it to see the overall performance of the model.

### Feature Importances
```python
# FEATURE IMPORTANCES 

lightgbm.plot_importance(booster = model, importance_type = 'gain', figsize=(12,35))
plt.title("Feature Importance (Gain) LGBM")
plt.savefig('Feature Importance (Gain) LGBM', bbox_inches = 'tight')
plt.show()
```

- Output :
  ![Feature Importance (Gain) LGBM](https://github.com/user-attachments/assets/ab7160b6-914f-487d-8717-36e172e05709)

```python
# SPLIT IMPORTANCES

lightgbm.plot_importance(booster = model, importance_type = 'split', figsize=(12,35))
plt.title('Split Importance LGBM')
plt.savefig('Split Importance .png', bbox_inches = 'tight')
```

- Output :
  ![Split Importance ](https://github.com/user-attachments/assets/716e6243-7a31-4bd0-aeea-d66e10af63e9)

  

