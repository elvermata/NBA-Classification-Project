# NBA Classification Project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Load dataset ---
file_path = r"C:\Users\elver\Project 2 classification\NBA-Classification-Project\team_traditional.csv"
team_data = pd.read_csv(file_path)
print("Team-level dataset loaded successfully!")

# Preview columns
print("Columns in dataset:")
print(team_data.columns.tolist())

# --- 1. Create HOME indicator (1 if the team is the home team, 0 if away) ---
team_data["HOME"] = (team_data["home"] == team_data["team"]).astype(int)

# --- 2. Make sure 'win' is integer (0 = loss, 1 = win) ---
team_data["win"] = team_data["win"].astype(int)

# --- 3. Drop columns we don’t need for modeling ---
drop_cols = ["gameid", "date", "type", "teamid", "team", "home", "away"]
team_data = team_data.drop(columns=drop_cols)

# --- 4. Convert percentage columns from string/object to numeric ---
for col in ["FG%", "3P%", "FT%"]:
    team_data[col] = pd.to_numeric(team_data[col], errors="coerce")

# --- 5. Filter for seasons 2000 and onward ---
team_data = team_data[team_data["season"] >= 2000]
print(f"Filtered dataset shape (2000+ seasons): {team_data.shape}")

# --- 6. Final check ---
print("\nCleaned dataset shape:", team_data.shape)
print(team_data.head())

win_loss_summary = team_data.groupby("win").mean(numeric_only=True)


# --- Visualizations ---
# Average FG% in Wins vs Losses
plt.figure(figsize=(6,4))
win_loss_summary["FG%"].plot(kind="bar", color=["red","green"])
plt.title("Average FG%: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss", "Win"], rotation=0)
plt.ylabel("FG%")
plt.show()

# Average Rebounds and Turnovers in Wins vs Losses
plt.figure(figsize=(8,5))
win_loss_summary[["REB", "TOV"]].plot(kind="bar", color=["blue","orange"])
plt.title("Average Rebounds & Turnovers: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss", "Win"], rotation=0)
plt.ylabel("Count")
plt.show()

# Home vs Away win rate
home_win_rate = team_data.groupby("HOME")["win"].mean()
plt.figure(figsize=(6,4))
home_win_rate.plot(kind="bar", color=["grey","yellow"])
plt.title("Home vs Away Win Rate")
plt.xticks(ticks=[0,1], labels=["Away","Home"], rotation=0)
plt.ylabel("Win Rate")
plt.show()

# --- Classification Model: Decision Tree ---
features = ["PTS", "REB", "AST", "TOV", "FG%", "3P%", "FT%", "HOME"]
target = "win"

X = team_data[features]
y = team_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit decision tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # you can tune max_depth
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = clf.feature_importances_
print("\nFeature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")

# --- Visualize the decision tree ---
plt.figure(figsize=(40,20))
plot_tree(clf, feature_names=features, class_names=["Loss","Win"], filled=True, rounded=True)
plt.title("Decision Tree for NBA Win Prediction")
plt.show()
