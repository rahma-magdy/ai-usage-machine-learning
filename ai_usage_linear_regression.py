
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# the cleaned dataset

df = pd.read_csv("cleaned_AI_Usage.csv")

print(df.shape)
print(df[["AI_Usage_Hours_Per_Day", "Productivity_Score"]].head())


# Remove missing values in the two columns

df = df.dropna(
    subset=["AI_Usage_Hours_Per_Day", "Productivity_Score"]
)



#  Choosing X and y

X = df[["AI_Usage_Hours_Per_Day"]]

y = df["Productivity_Score"]

print(X.head())
print(y.head())



#  Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)



#  Create Linear Regression

model = LinearRegression()



#  Train the model

model.fit(X_train, y_train)


#  Make predictions

predictions = model.predict(X_test)



#  Check the error

mae = mean_absolute_error(y_test, predictions)

print("MAE:", round(mae, 2))


#  Compare actual and predicted values

print(pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
}).head())
