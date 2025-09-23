# NBA Classification Project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File path
file_path = r"C:\Users\elver\OneDrive\Desktop\ITSC 3162\NBA archive (1)\team_traditional.csv"

# Load dataset
team_data = pd.read_csv(file_path)
print("Team-level dataset loaded successfully!")

# Strip spaces from column names (just in case)
team_data.columns = team_data.columns.str.strip()

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

# Save cleaned dataset to CSV
clean_path = r"C:\Users\elver\OneDrive\Desktop\ITSC 3162\NBA archive (1)\nba_cleaned_2000.csv"
team_data.to_csv(clean_path, index=False)
print(f"Cleaned dataset saved to: {clean_path}")

# Load the cleaned dataset
team_data = pd.read_csv(clean_path)

# Group by win/loss and calculate average stats
win_loss_summary = team_data.groupby("win").mean(numeric_only=True)
print("\nAverage stats for Wins (1) vs Losses (0):")
print(win_loss_summary)

# --- Visualizations ---

# 1. Average Points in Wins vs Losses
plt.figure(figsize=(6,4))
win_loss_summary["PTS"].plot(kind="bar", color=["red","green"])
plt.title("Average Points: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss", "Win"], rotation=0)
plt.ylabel("Points")
plt.show()

# 2. Average FG% in Wins vs Losses
plt.figure(figsize=(6,4))
win_loss_summary["FG%"].plot(kind="bar", color=["red","green"])
plt.title("Average FG%: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss", "Win"], rotation=0)
plt.ylabel("FG%")
plt.show()

# 3. Average Rebounds and Turnovers in Wins vs Losses
plt.figure(figsize=(8,5))
win_loss_summary[["REB", "TOV"]].plot(kind="bar", color=["blue","orange"])
plt.title("Average Rebounds & Turnovers: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss", "Win"], rotation=0)
plt.ylabel("Count")
plt.show()

# 4. Average AST, STL, BLK, PF in Wins vs Losses
plt.figure(figsize=(10,5))
win_loss_summary[["AST","STL","BLK","PF"]].plot(kind="bar", color=["blue","orange","green","purple"])
plt.title("Average AST, STL, BLK, PF: Wins vs Losses")
plt.xticks(ticks=[0,1], labels=["Loss","Win"], rotation=0)
plt.ylabel("Count")
plt.show()

# 5. Home vs Away win rate
home_win_rate = team_data.groupby("HOME")["win"].mean()
plt.figure(figsize=(6,4))
home_win_rate.plot(kind="bar", color=["grey","yellow"])
plt.title("Home vs Away Win Rate")
plt.xticks(ticks=[0,1], labels=["Away","Home"], rotation=0)
plt.ylabel("Win Rate")
plt.show()

# --- Classification Models ---

# Select features and target
features = ["PTS", "REB", "AST", "TOV", "FG%", "3P%", "FT%", "HOME"]
target = "win"

X = team_data[features]
y = team_data[target]

# ----------------------
# Model 1: Simple Decision Tree
# ----------------------
# Use median points as threshold to predict win
pts_threshold = X["PTS"].median()
y_pred_tree = (X["PTS"] > pts_threshold).astype(int)

# Calculate accuracy
accuracy_tree = (y_pred_tree == y).mean()
print(f"\nDecision Tree Accuracy (PTS split): {accuracy_tree:.2f}")

# ----------------------
# Model 2: Simple KNN (Manual, using numpy)
# ----------------------
# Sample subset for speed
sample = team_data.sample(2000, random_state=42)
X_sample = sample[features].to_numpy()
y_sample = sample[target].to_numpy()

k = 5  # number of neighbors
y_pred_knn = []

for i in range(len(X_sample)):
    # Compute distances to all other points
    distances = np.linalg.norm(X_sample - X_sample[i], axis=1)
    
    # Get indices of k nearest neighbors (exclude self)
    nearest_idx = distances.argsort()[1:k+1]
    
    # Predict by majority vote
    pred = round(y_sample[nearest_idx].mean())
    y_pred_knn.append(pred)

y_pred_knn = np.array(y_pred_knn)
accuracy_knn = (y_pred_knn == y_sample).mean()
print(f"KNN Accuracy (k={k}): {accuracy_knn:.2f}")
