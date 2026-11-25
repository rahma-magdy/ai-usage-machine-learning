
import pandas as pd
import numpy as np

# Loading Cleaned Dataset

df = pd.read_csv('cleaned_AI_Usage.csv')

print(df.shape)

print(df.head())


# Creating Age Groups

df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 25, 40, 60, np.inf],
    labels=['Young', 'Adult', 'Middle_Aged', 'Older'],
    right=True
)

# Checking the New Feature

print(df[['Age', 'Age_Group']].head(10))

print(df.shape)


# Saving Feature Engineered Dataset

df.to_csv(
    'AI_Usage_feature_engineered.csv',
    index=False
)


# Removing User ID

df = pd.read_csv('AI_Usage_feature_engineered.csv')

df = df.drop(
    columns=['User_ID'],
    errors='ignore'
)

print(df.shape)


# Cleaning AI Tool

df['AI_Tool'] = df['AI_Tool'].str.strip()

df['AI_Tool'] = df['AI_Tool'].replace({
    'Chat GPT': 'ChatGPT',
    'chatgpt': 'ChatGPT',
    'Chatgpt': 'ChatGPT',
    'Gemini': 'Google Gemini',
    'gemini': 'Google Gemini',
    'Copilot': 'Microsoft Copilot',
    'copilot': 'Microsoft Copilot',
    'claude': 'Claude'
})

print(df['AI_Tool'].unique())


# Cleaning Country

df['Country'] = df['Country'].str.strip()

df['Country'] = df['Country'].replace({
    'france': 'France',
    'canada': 'Canada',
    'india': 'India',
    'brazil': 'Brazil',
    'philippines': 'Philippines',
    'australia': 'Australia',
    'U.K.': 'United Kingdom',
    'United States of America': 'United States'
})

print(df['Country'].unique())


# Selecting Categorical Columns

categorical_columns = [
    'Gender',
    'Country',
    'User_Type',
    'Education_Level',
    'Profession',
    'AI_Tool',
    'AI_Purpose',
    'Age_Group'
]


# One-Hot Encoding

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)

print(df_encoded.shape)

print(df_encoded.head())


# Exporting Encoded Dataset

df_encoded.to_csv(
    'AI_Usage_encoded.csv',
    index=False
)


# Final Verification

check = pd.read_csv('AI_Usage_encoded.csv')

print("Final Dataset Shape:")
print(check.shape)

print("First Five Rows:")
print(check.head())

print("Data Types:")
print(check.dtypes)

print("Missing Values:")
print(check.isna().sum())
