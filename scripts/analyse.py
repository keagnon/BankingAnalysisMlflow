import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('data/banking.csv', sep=',')


print("Vérification immédiate après chargement :")
print(data['y'].unique())
print(data['y'].value_counts())


print(data.info())


sns.set(style="whitegrid")

def plot_variable_distribution(df, column_name):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(data=df, x=column_name, kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {column_name}')
    sns.boxplot(x=df[column_name], ax=ax[1])
    ax[1].set_title(f'Boxplot of {column_name}')
    plt.show()


numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
for column in numeric_columns:
    plot_variable_distribution(data, column)



# Calcul du pourcentage de valeurs manquantes
missing_percentage = data.isnull().mean() * 100
print("Pourcentage de valeurs manquantes par variable :")
print(missing_percentage)


print(data['y'].value_counts())

numeric_cols = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_cols].corr()

# Creating a heatmap to visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


print(data['y'].value_counts())


class_counts = data['y'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='dark')
plt.title('Frequency of Classes in Target Variable y')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['No', 'Yes'])  # assuming 0 is 'No', 1 is 'Yes'
plt.show()



print(data['y'].value_counts())


# Encodage des variables catégoriques
data = pd.get_dummies(data, drop_first=True)

# Vérification et correction des erreurs de typage
data = data.convert_dtypes()


print(data['y'].value_counts())


# Affichage des premières lignes et des informations du DataFrame nettoyé
print(data.head())


print(data.info())


print("Distribution avant division :")
# print(Y.value_counts())
print(data['y'].value_counts()) 


print("Vérification de la colonne 'y' :")
print(data['y'].unique())  # Afficher toutes les valeurs uniques dans la colonne cible
print(data['y'].value_counts())  # Afficher la distribution des classes



## Séparation des caractéristiques et de la variable cible
X = data.drop(columns=['y'])  # Exclure la colonne cible du dataset des caractéristiques
Y = data['y']  # Isoler la colonne cible


print(data['y'].value_counts())


from sklearn.model_selection import train_test_split

# Division des données sans stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Affichage de la distribution des classes dans Y_train avant SMOTE
print("Distribution des classes dans Y_train avant SMOTE :")
print(Y_train.value_counts())


# Convertir X_train en float64
X_train = X_train.astype(float)

# Convertir Y_train en int, s'assurer d'utiliser le type primitif int pour compatibilité
Y_train = Y_train.astype('int')

# Affichage des types pour confirmation
print("Types dans X_train:", X_train.dtypes)
print("Type de Y_train:", Y_train.dtype)



data = pd.concat([X_train, Y_train], axis=1)  # Assurez-vous que Y_train est une colonne du DataFrame

# Séparation par classe
majority = data[data['y'] == 0]
minority = data[data['y'] == 1]

# Upsample minority class
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)  # Réplication de la classe minoritaire

# Combinaison des nouvelles classes majoritaire et minoritaire rééquilibrées
data_upsampled = pd.concat([majority, minority_upsampled])

# Séparation en X et Y
X_train_upsampled = data_upsampled.drop('y', axis=1)
Y_train_upsampled = data_upsampled['y']

print("Distribution des classes dans Y_train après rééchantillonnage manuel :")
print(Y_train_upsampled.value_counts())






