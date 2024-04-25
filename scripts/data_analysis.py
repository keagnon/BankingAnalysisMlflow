import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_distribution(df, column_name):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(data=df, x=column_name, kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {column_name}')
    sns.boxplot(x=df[column_name], ax=ax[1])
    ax[1].set_title(f'Boxplot of {column_name}')
    plt.show()

def plot_correlation_matrix(df, numeric_cols):
    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

def plot_class_distribution(df, class_column):
    class_counts = df[class_column].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='dark')
    plt.title('Frequency of Classes in Target Variable y')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()
