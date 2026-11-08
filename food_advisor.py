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


# ---------------------------------------------------------------------------
# Price mapping, healthy alternatives and initialization
# ---------------------------------------------------------------------------
price_map = {"pizza": 120, "burger": 100, "rice": 40,
             "dosa": 50, "idli": 30, "apple": 20, "banana": 15
             }

healthy_alt = {"pizza": "dosa or idli",
               "burger": "rice with vegetables", "rice": "idli",
               "dosa": "idli", "idli": "apple", "apple": "banana",
               "banana": "apple"
               }
daily_limit = int(input("Enter your daily calorie limit: "))

total_calories = 0
total_cost = 0

food_list = []
calorie_list = []
cost_list = []

# ---------------------------------------------------------------------------
# User interaction & data collection
# ---------------------------------------------------------------------------
while True:
    food = input("\nEnter food item: ").lower()
    qty = int(input("Enter quantity: "))

    if food not in price_map:
        print("Food not found in database.")
        continue
    cal, prot, carb, fat = predict_nutrition(food, qty)
    cost = price_map[food] * qty
    total_calories += cal
    total_cost += cost
    food_list.append(food)
    calorie_list.append(cal)
    cost_list.append(cost)
    print("\nAdded:", food)
    print(f"Calories: {cal:.2f} kcal")
    print(f"Protein : {prot:.2f} g")
    print(f"Carbs   : {carb:.2f} g")
    print(f"Fat     : {fat:.2f} g")
    print(f"Cost    : ₹{cost}")
    more = input("Do you want to add another food item? (yes/no): ").lower()
    if more != "yes":
        break

# ---------------------------------------------------------------------------
# Report generation & AI advice
# ---------------------------------------------------------------------------
print("\nAI Food Expense & Health Report")
print("--------------------------------")
print("Foods consumed:", food_list)
print(f"Total Calories: {total_calories:.2f} kcal")
print(f"Total Expense : ₹{total_cost}")

if total_calories > daily_limit:
    print("Health Advice : You exceeded your daily calorie limit.")
else:
    print("Health Advice : Calorie intake is within your limit.")

if total_cost > 300:
    print("Expense Advice: High food spending. Try budget-friendly meals.")
else:
    print("Expense Advice: Expense is under control.")

print("\nHealthy Alternatives:")
for food in food_list:
    print(f"Instead of {food}, try {healthy_alt[food]}")
