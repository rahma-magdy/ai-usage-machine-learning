import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_AI_Usage.csv')

print(df.shape)
print(df.head())


plt.hist(df['Age'], bins=10, edgecolor='white')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.show()


counts = df['AI_Tool'].value_counts()
counts.plot(kind='bar')
plt.ylabel('Number of Users')
plt.show()

counts = df['AI_Tool'].value_counts()

counts.plot(kind='bar')

plt.ylabel('Number of Users')

plt.show()

sns.countplot(data=df, x='Gender')
plt.show()

sns.boxplot(data=df, y='Productivity_Score')
plt.show()

sns.scatterplot(data=df, x='Age', y='Productivity_Score')
plt.show()


cols = [
    'Productivity_Score',
    'Accuracy_Rating',
    'Satisfaction_Score'
]

corr = df[cols].corr()

print(corr.round(2))


cols = [
    'Productivity_Score',
    'Accuracy_Rating',
    'Satisfaction_Score'
]

corr = df[cols].corr()

print(corr.round(2))

sns.heatmap(corr, annot=True, cmap='BrBG', vmin=-1, vmax=1)
plt.show()

sns.pairplot(df[['Productivity_Score', 'Accuracy_Rating', 'Satisfaction_Score']])
plt.show()

