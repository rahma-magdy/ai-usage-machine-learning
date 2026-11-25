
import pandas as pd

dataFrame = pd.read_csv('AI_Usage.csv')




print(dataFrame)

print(dataFrame.shape)

print(dataFrame.head(5))

print(dataFrame.info())

print(dataFrame.describe())

print(dataFrame.dtypes)

print(dataFrame.isna().sum())



# Cleaning Age


dataFrame['Age'] = dataFrame['Age'].str.replace(
    ' yrs', ''
).str.strip()

dataFrame['Age'] = pd.to_numeric(
    dataFrame['Age'], errors='coerce'
)

dataFrame['Age'] = dataFrame['Age'].fillna(
    dataFrame['Age'].median()
)



# Inspecting Columns Before Cleaning


print("Monthly Income:")
print(dataFrame['Monthly_Income'].unique())

print("AI Usage Hours:")
print(dataFrame['AI_Usage_Hours_Per_Day'].unique())

print("Tasks Performed:")
print(dataFrame['Tasks_Performed'].unique())

print("Monthly AI Cost:")
print(dataFrame['Monthly_AI_Cost'].unique())

print("Time Saved:")
print(dataFrame['Time_Saved_Hours_Per_Week'].unique())

print("Work or Study Hours:")
print(dataFrame['Work_or_Study_Hours_Per_Day'].unique())



# Cleaning Gender


print("Gender before cleaning:")
print(dataFrame['Gender'].unique())

dataFrame['Gender'] = (
    dataFrame['Gender'].str.strip().str.capitalize()
)

dataFrame['Gender'] = dataFrame['Gender'].replace({
    'M': 'Male',
    'F': 'Female'
})

print("Gender after cleaning:")
print(dataFrame['Gender'].unique())



# Cleaning Monthly Income


print("Monthly Income before cleaning:")
print(dataFrame['Monthly_Income'].unique())

dataFrame['Monthly_Income'] = (
    dataFrame['Monthly_Income'].str.strip()
)

dataFrame['Monthly_Income'] = pd.to_numeric(
    dataFrame['Monthly_Income'], errors='coerce'
)


# Cleaning AI Usage Hours Per Day


dataFrame['AI_Usage_Hours_Per_Day'] = (
    dataFrame['AI_Usage_Hours_Per_Day'].str.replace(
        ' hrs', ''
    ).str.strip()
)

dataFrame['AI_Usage_Hours_Per_Day'] = pd.to_numeric(
    dataFrame['AI_Usage_Hours_Per_Day'],
    errors='coerce'
)



# Cleaning Monthly AI Cost


dataFrame['Monthly_AI_Cost'] = (
    dataFrame['Monthly_AI_Cost'].str.replace(
        '$', '', regex=False
    ).str.strip()
)

dataFrame['Monthly_AI_Cost'] = pd.to_numeric(
    dataFrame['Monthly_AI_Cost'],
    errors='coerce'
)



# Cleaning Time Saved Per Week


dataFrame['Time_Saved_Hours_Per_Week'] = (
    dataFrame['Time_Saved_Hours_Per_Week'].str.replace(
        ' hrs', ''
    ).str.strip()
)

dataFrame['Time_Saved_Hours_Per_Week'] = pd.to_numeric(
    dataFrame['Time_Saved_Hours_Per_Week'],
    errors='coerce'
)



# Cleaning Work or Study Hours


dataFrame['Work_or_Study_Hours_Per_Day'] = (
    dataFrame['Work_or_Study_Hours_Per_Day'].str.replace(
        ' hrs', ''
    ).str.strip()
)

dataFrame['Work_or_Study_Hours_Per_Day'] = pd.to_numeric(
    dataFrame['Work_or_Study_Hours_Per_Day'],
    errors='coerce'
)


# Checking and Removing Duplicates


print("Number of duplicates:")
print(dataFrame.duplicated().sum())

clean_df = dataFrame.drop_duplicates()

print("Shape after removing duplicates:")
print(clean_df.shape)



# Checking Data Types


print("Data Types After Cleaning:")
print(clean_df.dtypes)


# Checking Missing Values


print("Missing Values After Cleaning:")
print(clean_df.isna().sum())




# Handling Missing Numerical Values


# Monthly Income

clean_df['Monthly_Income'] = clean_df['Monthly_Income'].fillna(
    clean_df['Monthly_Income'].median()
)


# AI Usage Hours Per Day

clean_df['AI_Usage_Hours_Per_Day'] = clean_df['AI_Usage_Hours_Per_Day'].fillna(
    clean_df['AI_Usage_Hours_Per_Day'].median()
)


# Monthly AI Cost

clean_df['Monthly_AI_Cost'] = clean_df['Monthly_AI_Cost'].fillna(
    clean_df['Monthly_AI_Cost'].median()
)


# Productivity Score

clean_df['Productivity_Score'] = clean_df['Productivity_Score'].fillna(
    clean_df['Productivity_Score'].median()
)


# Accuracy Rating

clean_df['Accuracy_Rating'] = clean_df['Accuracy_Rating'].fillna(
    clean_df['Accuracy_Rating'].median()
)


# Satisfaction Score

clean_df['Satisfaction_Score'] = clean_df['Satisfaction_Score'].fillna(
    clean_df['Satisfaction_Score'].median()
)


# Time Saved Hours Per Week

clean_df['Time_Saved_Hours_Per_Week'] = clean_df['Time_Saved_Hours_Per_Week'].fillna(
    clean_df['Time_Saved_Hours_Per_Week'].median()
)


# Work or Study Hours Per Day

clean_df['Work_or_Study_Hours_Per_Day'] = clean_df['Work_or_Study_Hours_Per_Day'].fillna(
    clean_df['Work_or_Study_Hours_Per_Day'].median()
)



# Checking Missing Values


print("Missing Values After Filling:")
print(clean_df.isna().sum())



# Handling Missing Categorical Values


clean_df['Gender'] = clean_df['Gender'].fillna('Unknown')

clean_df['Country'] = clean_df['Country'].fillna('Unknown')

clean_df['Education_Level'] = clean_df['Education_Level'].fillna('Unknown')

clean_df['Profession'] = clean_df['Profession'].fillna('Unknown')

clean_df['AI_Tool'] = clean_df['AI_Tool'].fillna('Unknown')

clean_df['AI_Purpose'] = clean_df['AI_Purpose'].fillna('Unknown')

clean_df['Tasks_Performed'] = clean_df['Tasks_Performed'].fillna('Unknown')



# Checking Missing Values


print("Missing Values After Cleaning:")
print(clean_df.isna().sum())


# Saving Cleaned Dataset


clean_df.to_csv('cleaned_AI_Usage.csv', index=False)


# Checking Saved Dataset


check = pd.read_csv('cleaned_AI_Usage.csv')

print(check.shape)

print(check.head(5))

print(check.info())

print(check.isna().sum())





# Saving Cleaned Dataset


clean_df.to_csv(
    'cleaned_AI_Usage.csv',
    index=False
)



# Checking Saved Dataset


check = pd.read_csv('cleaned_AI_Usage.csv')

print("Final Dataset Shape:")
print(check.shape)

print("First Five Rows:")
print(check.head(5))

print("Final Data Types:")
print(check.dtypes)

print("Final Missing Values:")
print(check.isna().sum())
