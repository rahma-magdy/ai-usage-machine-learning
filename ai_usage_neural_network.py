

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Step 2: Load the cleaned dataset

df = pd.read_csv("cleaned_AI_Usage.csv")

print("Original shape:", df.shape)


# Remove rows with missing values

df = df.dropna(
    subset=[
        "Age",
        "AI_Usage_Hours_Per_Day",
        "Productivity_Score"
    ]
)

print("Shape after removing missing values:", df.shape)



# Choose X and y

X = df[["Age", "AI_Usage_Hours_Per_Day"]]

y = (
    df["Productivity_Score"] >= 6
).astype(int)

print("X:")
print(X.head())

print("\ny:")
print(y.head())

print("\nClass distribution:")
print(y.value_counts())



# Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


#  Scale the features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



#  Create a small Neural Network

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=1000,
    random_state=42
)


#  Train the model

model.fit(X_train, y_train)


#  Make predictions

predictions = model.predict(X_test)

print("\nPredicted:", predictions)
print("Actual:", y_test.values)



#  Evaluate the model

accuracy = accuracy_score(y_test, predictions)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:\n", matrix)
