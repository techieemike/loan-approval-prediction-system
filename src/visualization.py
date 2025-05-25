import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_loan_status_distribution(df):
    plt.figure(figsize=(8, 5))
    df['Loan_Status'].value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.title('Train Dataset Loan Status Distribution')
    plt.show()

def plot_gender_distribution(df):
    plt.figure(figsize=(8, 5))
    df['Gender'].value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Train Dataset Gender Distribution')
    plt.show()

def plot_self_employed_distribution(df):
    plt.figure(figsize=(8, 5))
    ax = df['Self_Employed'].value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.xlabel('Self Employed Status')
    plt.ylabel('Count')
    plt.title('Train Dataset Self Employed Status Distribution')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.show()

def plot_applicant_income_distribution(df):
    fig = px.histogram(df, x='ApplicantIncome', title='Applicant Income Distribution')
    fig.show()

def plot_scatter_applicant_income_vs_loan_amount(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount')
    plt.xlabel('Applicant Income')
    plt.ylabel('Loan Amount')
    plt.title('Scatter Plot: Applicant Income vs Loan Amount')
    plt.show()
