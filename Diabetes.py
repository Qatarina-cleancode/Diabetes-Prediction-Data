#import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ALPHA = 0.05

# setting the theme

sns.set_theme(style="whitegrid")
SHOW_PLOTS = False
#Where to save figures
os.makedirs("figures", exist_ok=True)

def save_figure(fig, filename):
    fig.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

#load dataset
df=pd.read_csv('diabetes.csv')

#Inspecting dataset.
#patients in the dataset
print(df.shape)
#what each variable represnt
print(df.info())
#numerical variables
print(df.select_dtypes(include='number').columns)
#categorical variables
print(df.select_dtypes(include='object').columns)
#Target variable
print("Target variable: Outcome")
#patients with diabets
print((df['Outcome']==1).sum())
#patients without diabets
print((df['Outcome']==0).sum())
#Data quality assesment
#null values in percentage
print((df.isnull().sum() / len(df))*100)
#duplicate values
print(df.duplicated().sum())
#checking for phisiologically impalusive values
print(df.min())
print(df.max())
#patients that have 0 values in the following variables
columns=['Glucose','SkinThickness','Insulin','BMI','BloodPressure']
print((df[columns]==0).sum())
#Replacing Zero values
zero_cols=['Glucose','SkinThickness','Insulin','BMI','BloodPressure']
df[zero_cols]=df[zero_cols].replace(0,np.nan)
print(df.head(10))
print(df[zero_cols].isnull().sum())

#Finding outliers
#boxplot
cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
      'BMI','DiabetesPedigreeFunction','Age','Outcome']

for col in cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title('Possible outliers')
    plt.savefig(f'outlier_{col}.png')
    #plt.show()
    plt.close()
#Histogram
for col in columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'Histogram_{col}.png')
    #plt.show(block=False)
    plt.close()
    
#Quantiles
print(df[cols].quantile([0.25,0.50,0.75]))

#IQR measures the spread of 50%
cols = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]
plt.close('all')
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

print("Q1:")
print(Q1)

print("\nQ3:")
print(Q3)

print("\nIQR:")
print(IQR)

#creating a clean copy
clean_df=df.copy()
observed_df = clean_df.copy() 
# Replace nan with median.
zero_cols = [  'Glucose','SkinThickness', 'Insulin','BMI','BloodPressure']
for col in zero_cols:
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    print(clean_df.isnull().sum())
    print(clean_df.shape)
    print(clean_df.head())

#Descriptive analytics
print(clean_df.describe())
# Average
print(clean_df.groupby('Outcome').mean(numeric_only=False))

#statistics for people with diabetes
group_stats=clean_df.groupby('Outcome').agg({'Pregnancies':['mean','median','max','min','std'],
                                             'Glucose':['mean','median','max','min','std'],
                                             'BloodPressure':['mean','median','max','min','std'],
                                             'SkinThickness':['mean','median','max','min','std'],
                                             'Insulin':['mean','median','max','min','std'],
                                             'BMI':['mean','median','max','min','std'],
                                             'DiabetesPedigreeFunction':['mean','median','max','min','std'],
                                             'Age':['mean','median','max','min','std']})

print(group_stats)

#statistics for people without diabetes
No_Diabetes=clean_df[clean_df['Outcome'] == 0]
print(No_Diabetes)
#statistics for people with diabetes
Diabetes=clean_df[clean_df['Outcome']==1]
print(Diabetes)

#percentage of people with diabetes
#number of people with diabetes

counts=clean_df['Outcome'].value_counts()
perct=clean_df['Outcome'].value_counts(normalize=True)*100
print("Number of patients:\n",counts,"\n")
print("perctage of patients\n",perct.round(1),"\n")

#Bar plot
plt.figure(figsize=(5,4))
plt.bar(['No diabetes','Diabetes'],
        [counts[0],counts[1]],
        color=['Blue','yellow'])
plt.title(f"diabetes dataset({perct[1]:.1f})% have diabetes)")
plt.ylabel('No of patients')
plt.savefig('Diabetes barplot.png')

#grouping patients
clean_df['AgeGroup']=pd.cut(clean_df['Age'],
                            bins=[20,30,45,100],
                            labels=['Young(21_30)','Middle_Age(31-45)','Old_Age(46+)'])

clean_df['BMICategory']=pd.cut(clean_df['BMI'],
                               bins=[0,18.5,25,30,100],
                               labels=['Underweight','Normal','Overweight','Obese'],
                               right=False)


clean_df['GlucoseCategory'] = pd.cut( clean_df['Glucose'],
                                    bins=[0, 140, 200],
                                    labels=['Normal (<140)', 'Impaired tolerance (140-199)'],
                                    right=False)


#diabetes_prevalence by age_group
age_prevalence =(
    clean_df.groupby('AgeGroup',observed='True')['Outcome'].mean()*100
)
print(age_prevalence.round(1))
#diabetes_prevalence by BMI
BMI_prevalence= (
    clean_df.groupby('BMICategory', observed='True')['Outcome'].mean()*100
)
print(BMI_prevalence.round(1))

#Diabetes_prevalence by Glucose
Glu_prevalence=(
    clean_df.groupby('GlucoseCategory', observed=True)['Outcome'].mean()*100
)
print(Glu_prevalence.round(1))

#PREVALENCE BY GROUP
def plot_prevalence(group_col, title, filename):
    # Step 1: count patients and calculate % with diabetes in each group
    summary = clean_df.groupby(group_col, observed=True)["Outcome"].agg(
        n="count",
        prevalence=lambda s: s.mean() * 100,
    )
    print(summary.round(1))

    # Step 2: draw the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.index.astype(str), summary["prevalence"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Diabetes prevalence (%)")
    ax.set_ylim(0, 100)

    # Step 3: write the percentage and group size above each bar
    for i, (prev, n) in enumerate(zip(summary["prevalence"], summary["n"])):
        ax.text(i, prev + 2, f"{prev:.1f}%\n(n={n})", ha="center")

    # Step 4: save
    fig.savefig(f"figures/{group_col}", dpi=300, bbox_inches="tight")
plt.close()
#features to plot
plot_prevalence("AgeGroup", "Diabetes prevalence by age group", "fig04_prevalence_age.png")
plot_prevalence("BMICategory", "Diabetes prevalence by BMI category", "fig05_prevalence_bmi.png")
plot_prevalence("GlucoseCategory", "Diabetes prevalence by glucose category", "fig06_prevalence_glucose.png")


# EXPLORATORY DATA ANALYSIS
# Readable labels for charts (instead of 0 and 1)
observed_df["Diagnosis"] = observed_df["Outcome"].map({0: "No diabetes", 1: "Diabetes"})
order = ["No diabetes", "Diabetes"]


#distribution of each variable on its own 
def plot_distribution(col, filename):
#Histogram of one variable for all patients.
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(observed_df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of patients")
    save_figure(fig, filename)


plot_distribution("Age", "fig08_age_distribution.png")
plot_distribution("Glucose", "fig09_glucose_distribution.png")
plot_distribution("BMI", "fig10_bmi_distribution.png")
plot_distribution("BloodPressure", "fig11_bloodpressure_distribution.png")


# compare diabetic and non-diabetic patients
def compare_by_outcome(col, filename):
# Boxplot and histogram of one variable, split by diabetes status
#Also prints summary statistics for each group
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: boxplot (compares the middle, spread and outliers)
    sns.boxplot(data=observed_df, x="Diagnosis", y=col, order=order, ax=axes[0])
    axes[0].set_title(f"{col} by diabetes status")
    axes[0].set_xlabel("")

# Right panel: overlapping histograms (compares the whole shape)
    sns.histplot(data=observed_df, x=col, hue="Diagnosis", hue_order=order,
                 kde=True, stat="density", common_norm=False, ax=axes[1])
    axes[1].set_title(f"{col} distribution by diabetes status")

    plt.tight_layout()
    save_figure(fig, filename)

    print(f"\n=== {col} ===")
    print(observed_df.groupby("Diagnosis")[col].describe().round(2))


compare_by_outcome("Glucose", "fig12_glucose_vs_diabetes.png")
compare_by_outcome("BMI", "fig13_bmi_vs_diabetes.png")
compare_by_outcome("Age", "fig14_age_vs_diabetes.png")
compare_by_outcome("BloodPressure", "fig15_bloodpressure_vs_diabetes.png")
compare_by_outcome("Insulin", "fig16_insulin_vs_diabetes.png")
compare_by_outcome("Pregnancies", "fig17_pregnancies_vs_diabetes.png")

# CORRELATION ANALYSIS
# numeric_cols = the 8 predictors
numeric_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
corr_cols = numeric_cols + ["Outcome"]     

# correlation matrices (Pearson and Spearman) ----
corr_pearson = observed_df[corr_cols].corr(method="pearson")
corr_spearman = observed_df[corr_cols].corr(method="spearman")

#heatmap (required figure)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation matrix (Pearson, observed values only)")
plt.tight_layout()
save_figure(fig, "fig18_correlation_heatmap.png")

#which variables are most associated with Outcome? ----
outcome_corr = pd.DataFrame({
    "pearson": corr_pearson["Outcome"],
    "spearman": corr_spearman["Outcome"],
}).drop("Outcome")
outcome_corr = outcome_corr.sort_values("pearson", ascending=False).round(2)
print("Correlation with Outcome (strongest first):")
print(outcome_corr)

#Strongest and weakest relationships BETWEEN predictors 
pred_corr = corr_pearson.loc[numeric_cols, numeric_cols]

# keep each pair once (upper triangle only, excluding the diagonal)
upper = pred_corr.where(np.triu(np.ones(pred_corr.shape, dtype=bool), k=1))
pairs = upper.stack().dropna().sort_values(key=abs, ascending=False).round(2)

print("\nStrongest predictor-predictor correlations:")
print(pairs.head(5))
print("\nWeakest predictor-predictor correlations:")
print(pairs.tail(5))

#statistical analysis

def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def compare_two_groups(col):
#Welch t-test and Mann-Whitney U for one numeric variable,
#comparing diabetes (Outcome = 1) with no diabetes (Outcome = 0)."""
    diabetic = observed_df.loc[observed_df["Outcome"] == 1, col].dropna()
    healthy = observed_df.loc[observed_df["Outcome"] == 0, col].dropna()

    print(f"\n{'=' * 55}\n{col}: diabetes vs no diabetes\n{'=' * 55}")
    print(f"n:       diabetes = {len(diabetic)}, no diabetes = {len(healthy)}")
    print(f"mean:    diabetes = {diabetic.mean():.2f}, no diabetes = {healthy.mean():.2f}")
    print(f"median:  diabetes = {diabetic.median():.2f}, no diabetes = {healthy.median():.2f}")
    print(f"skewness (0 = symmetric): diabetes = {diabetic.skew():.2f}, "
          f"no diabetes = {healthy.skew():.2f}")

# Welch's t-test: compares means, does not assume equal spread
    t_stat, p_t = stats.ttest_ind(diabetic, healthy, equal_var=False)

 # Effect size for the t-test: Cohen's d (difference in standard deviations)
    pooled_sd = np.sqrt((diabetic.var(ddof=1) + healthy.var(ddof=1)) / 2)
    cohens_d = (diabetic.mean() - healthy.mean()) / pooled_sd

# Mann-Whitney U: compares the ranking of values, robust to skew
    u_stat, p_u = stats.mannwhitneyu(diabetic, healthy, alternative="two-sided")

# Effect size for Mann-Whitney: probability that a randomly chosen diabetic
# patient has a HIGHER value than a randomly chosen non-diabetic patient
    prob_superiority = u_stat / (len(diabetic) * len(healthy))

    print(f"\nWelch t-test:   t = {t_stat:.2f}, p = {format_p(p_t)}, Cohen's d = {cohens_d:.2f}")
    print(f"Mann-Whitney U: U = {u_stat:.0f}, p = {format_p(p_u)}, "
          f"P(diabetic > non-diabetic) = {prob_superiority:.2f}")


compare_two_groups("Glucose")
compare_two_groups("Insulin")


# ---- Chi-square test: AgeGroup vs Outcome ----
# Table of counts: rows = age group, columns = outcome
contingency = pd.crosstab(clean_df["AgeGroup"], clean_df["Outcome"])
contingency.columns = ["No diabetes", "Diabetes"]
print(f"\n{'=' * 55}\nAgeGroup vs Outcome (chi-square)\n{'=' * 55}")
print("Observed counts:")
print(contingency)

chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
expected_df = pd.DataFrame(expected, index=contingency.index,
                           columns=contingency.columns).round(1)
print("\nExpected counts if age group and diabetes were unrelated:")
print(expected_df)
print(f"\nSmallest expected count: {expected.min():.1f}  (should be at least 5)")

# Effect size for chi-square: Cramer's V (0 = no association, 1 = perfect)
n_total = contingency.to_numpy().sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
print(f"\nChi-square = {chi2:.2f}, df = {dof}, p = {format_p(p_chi)}, Cramer's V = {cramers_v:.2f}") 