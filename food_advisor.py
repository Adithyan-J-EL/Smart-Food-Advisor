# Project Title: AI-Based Smart Food Expense and Health Advisor
# Author: Adithyan J
# Tools: Python (Google Colab), Pandas, scikit-learn, Matplotlib
# Problem Statement: People lack awareness of calorie intake and food expenses,
#   making it difficult to maintain healthy and cost-effective eating habits.
# Objective: Build an AI agent that predicts food calories, tracks expenses,
#   and provides health-based recommendations.

# ---------------------------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------------------------
# Pandas stores and processes the food data.
# Matplotlib visualizes calorie consumption.
# scikit-learn builds the AI prediction model.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Dataset creation (knowledge base for the AI model)
# ---------------------------------------------------------------------------
food_data = {
    "food": ["pizza", "burger", "rice", "dosa", "idli", "apple", "banana"],
    "calories": [285, 295, 130, 168, 58, 52, 89],
    "protein": [12, 17, 2.7, 4, 2, 0.3, 1.1],
    "carbs": [36, 30, 28, 30, 12, 14, 23],
    "fat": [10, 12, 0.3, 3, 0.4, 0.2, 0.3],
    "price": [120, 100, 40, 50, 30, 20, 15]
}

df = pd.DataFrame(food_data)

# ---------------------------------------------------------------------------
# AI model training (TF-IDF + Linear Regression via MultiOutputRegressor)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["food"])
y = df[["calories", "protein", "carbs", "fat"]]
model = MultiOutputRegressor(LinearRegression())
model.fit(X, y)


# ---------------------------------------------------------------------------
# Nutrition prediction function
# ---------------------------------------------------------------------------
def predict_nutrition(food_name, quantity):
    food_vec = vectorizer.transform([food_name])
    prediction = model.predict(food_vec)[0]

    calories, protein, carbs, fat = prediction

    return (
        calories * quantity,
        protein * quantity,
        carbs * quantity,
        fat * quantity
    )

