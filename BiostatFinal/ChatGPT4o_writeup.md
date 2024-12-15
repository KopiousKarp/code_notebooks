## Baackground
A continuing epidemiological study of heart disease was established in Framingham, Mass., during the period of 1948-1950. This study is concerned with measurements of the extent and development of cardiovascular disease in a cross-section of the population aged 30-59 on January 1, 1950, and with the study of those environmental and personal factors which are associated with the subsequent appearance and progression of cardiovascular diseases. Data are obtained primarily from biennial examinations conducted in a clinic maintained especially for this study. It is planned to continue observation of the group of subjects for 20 years. The present report describes the experience with arteriosclerotic heart disease observed in the four years following each individual’s initial examination (Dawber & Moore, 2015).

## Study Design

### Aims
The primary aim of this study is to identify significant predictors of *TenYearCHD* (10-year Coronary Heart Disease risk) using statistical modeling techniques. Specifically, the study seeks to:
- Identify key risk factors associated with CHD.
- Perform variable selection using appropriate statistical criteria to optimize the model.
- Compare various predictive modeling strategies to evaluate their performance.

### Data and Materials
The dataset includes patient-level predictors such as demographic, clinical, and lifestyle factors. These variables include:
- **Gender**
- **Age**
- **CigsPerDay** (number of cigarettes smoked per day)
- **totChol** (total cholesterol levels)
- **sysBP** (systolic blood pressure)
- **glucose** (fasting blood glucose levels)

The response variable is **TenYearCHD**, a binary variable indicating whether a patient developed Coronary Heart Disease over a 10-year period (1 = CHD, 0 = No CHD).

The methods for collecting the data points in the Framingham Heart Study, as detailed in the provided PDF, are as follows:

#### Outcome Variable (Target)
**TenYearCHD**: Indicates 10-year risk of coronary heart disease.
- **Data Collection Method**: This variable is determined based on the occurrence of coronary heart disease (CHD) events over a 10-year follow-up period. CHD events include myocardial infarction, angina pectoris, coronary insufficiency, and CHD death. The data is collected through biennial examinations, medical records, and follow-up interviews (Dawber & Moore, 2015).

#### Risk Factors (Predictor Variables)
1. **Gender**: Male (1) or Female (0) (Nominal).
   - **Data Collection Method**: Gender is recorded during the initial examination based on self-report and medical records.

2. **Age**: Age of the patient (Continuous).
   - **Data Collection Method**: Age is recorded at the time of the initial examination and updated at each follow-up visit.

3. **currentSmoker**: Whether the patient is a current smoker ("1" = Yes, "0" = No) (Nominal).
   - **Data Collection Method**: Smoking status is determined through self-reported questionnaires and interviews conducted during the examinations.

4. **CigsPerDay**: Average number of cigarettes smoked per day (Continuous).
   - **Data Collection Method**: The number of cigarettes smoked per day is self-reported by participants during the examinations.

5. **BPMeds**: Whether the patient is on blood pressure medication ("1" = Yes, "0" = No) (Nominal).
   - **Data Collection Method**: Information on blood pressure medication use is collected through self-reported questionnaires and verified by reviewing medical records.

6. **prevalentStroke**: History of stroke ("1" = Yes, "0" = No) (Nominal).
   - **Data Collection Method**: History of stroke is determined through self-reported medical history, physician diagnosis, and medical records.

7. **prevalentHyp**: History of hypertension ("1" = Yes, "0" = No) (Nominal).
   - **Data Collection Method**: Hypertension history is collected through self-reported medical history, physician diagnosis, and medical records.

8. **Diabetes**: Whether the patient has diabetes ("1" = Yes, "0" = No) (Nominal).
   - **Data Collection Method**: Diabetes status is determined through self-reported medical history, physician diagnosis, and medical records.

9. **totChol**: Total cholesterol level (Continuous).
   - **Data Collection Method**: Total cholesterol levels are measured through blood tests conducted during the examinations.

10. **sysBP**: Systolic blood pressure (Continuous).
    - **Data Collection Method**: Systolic blood pressure is measured using a sphygmomanometer during the physical examination.

11. **diaBP**: Diastolic blood pressure (Continuous).
    - **Data Collection Method**: Diastolic blood pressure is measured using a sphygmomanometer during the physical examination.

12. **BMI**: Body mass index (Continuous).
    - **Data Collection Method**: BMI is calculated using the height and weight measurements taken during the physical examination.

13. **heartRate**: Heart rate (Continuous).
    - **Data Collection Method**: Heart rate is measured using a stethoscope or an electrocardiogram (ECG) during the physical examination.

14. **Glucose**: Glucose level (Continuous).
    - **Data Collection Method**: Glucose levels are measured through blood tests conducted during the examinations.

These data points are collected through a combination of self-reported questionnaires, physical examinations, laboratory tests, and medical record reviews, ensuring comprehensive and accurate data collection for the study.

## Statistical Analysis

### Methods
A series of statistical techniques were applied to identify significant predictors and optimize model performance:

#### Response Variable
The primary outcome of interest is **TenYearCHD**, a binary response variable.

#### Risk Factors
Potential predictors (risk factors) include demographic and clinical features:
- Gender
- Age
- Cigarettes smoked per day (CigsPerDay)
- Total cholesterol levels (totChol)
- Systolic blood pressure (sysBP)
- Glucose levels

#### Model Selection and Variable Selection
The following approaches were applied:
1. **Univariable Logistic Regression**: Each predictor was tested individually to identify significant variables.
2. **Multivariable Logistic Regression**: Statistically significant predictors (p < 0.05) were included in a multivariable model.
3. **LASSO Regression (L1 Penalty)**: To perform feature selection and reduce model complexity, LASSO regression was applied.
4. **Cross-Validation and Evaluation**: Models were evaluated using a stratified 5-fold cross-validation strategy with the AUC (Area Under the ROC Curve) as the primary performance metric.
5. **Class Imbalance Handling**: Techniques such as weighting were applied to address the imbalanced class distribution of *TenYearCHD*.
6. **Model Comparisons**: Various models, including Random Forests, SVM, and logistic regression with interaction terms, were tested to explore potential performance improvements.

## Results

### Model Used
The final model selected was **LASSO Logistic Regression**, which outperformed other models in terms of AUC.

#### Evaluation Metrics
The LASSO regression model achieved:
- **AUC (Area Under the Curve)**: 0.74

The model's performance did not improve significantly with more complex techniques (e.g., SVM, Random Forests, or interaction terms).

### Major Findings
Significant predictors identified through the LASSO regression model include:
- **Gender**
- **Age**
- **CigsPerDay** (number of cigarettes smoked per day)
- **totChol** (total cholesterol)
- **sysBP** (systolic blood pressure)
- **Glucose**

#### Table 4: Coefficients of Final Model
| Predictor     | Coefficient | p-value |
|---------------|-------------|---------|
| Gender        | 0.581       | <0.001  |
| Age           | 0.065       | <0.001  |
| CigsPerDay    | 0.020       | <0.001  |
| totChol       | 0.002       | 0.036   |
| sysBP         | 0.017       | <0.001  |
| Glucose       | 0.008       | <0.001  |

#### Figure 1: ROC Curve for Final LASSO Regression Model

<!-- ![ROC Curve](image.png) -->

The ROC curve shows the AUC of 0.74, indicating acceptable model performance.

## Conclusion
The LASSO logistic regression model identified **Gender**, **Age**, **CigsPerDay**, **totChol**, **sysBP**, and **Glucose** as significant predictors of *TenYearCHD*. Despite testing more complex models (e.g., Random Forest, SVM, and interaction terms), no model outperformed LASSO regression (AUC = 0.74). This result highlights the robustness and simplicity of LASSO for variable selection and prediction in this context.
