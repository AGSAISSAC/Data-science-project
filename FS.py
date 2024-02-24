import pandas as pd 
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
def featureselection(df):
    categorical_variable = ['gender','SeniorCitizen','Partner', 'Dependents','PhoneService','MultipleLines',
                            'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                            'StreamingMovies','Contract','PaperlessBilling', 'PaymentMethod']
    print(" performing CHI SQUARE test on categorical variable")
    for columns in categorical_variable:
        contingency_table= pd.crosstab(df[columns],df['Churn'])
        if not contingency_table.empty:
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            if p < 0.05:
                print(f" Consider {columns} as its P value is {p}")
            else:
                print(f"Skip the {columns} for a better fit as p value is {p}")

    print("Performing T-Test on continuous variables")
    continuous_variable =['MonthlyCharges','TotalCharges','tenure']
    for column in continuous_variable:
        t_stat, p = ttest_ind(df[column],df['Churn'], equal_var=False)
        if p < 0.05:
                print(f" Consider {column} as its P value is {p}")
        else:
                print(f"Skip the {column} for a better fit as p value is {p}")

    corr_matrix = df.corr()
    plt.figure(figsize = (20,20))
    sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()