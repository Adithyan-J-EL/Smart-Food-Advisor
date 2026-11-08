# AI-Based Smart Food Expense and Health Advisor

An AI-based decision support agent that predicts the calories and nutritional values of food items, tracks food expenses, and gives health and budget advice.

**Author:** Adithyan J
**Institution:** Mar Baselios College of Engineering and Technology
**Course:** Electrical and Computer Engineering (UG, Third Year)

## Problem Statement

People often consume food without knowing its calorie content and cost impact. Manual calculation of calories, nutrients and expenses is time-consuming and error-prone, and a lack of awareness can lead to unhealthy eating habits and unnecessary spending. This project provides an intelligent system that automatically analyzes food intake and gives meaningful health and budget recommendations.

## Features

- Predicts calories, protein, carbs and fat for a food item and quantity
- Tracks the total cost of the meal
- Compares calorie intake against the user's daily calorie limit
- Gives health advice, expense advice and healthier food alternatives
- Calculates a meal health score (0-10) and rating (Healthy / Moderate / Unhealthy)
- Visualizes calories and cost per food item with Matplotlib

## AI Approach

1. **NLP (TF-IDF):** food names are converted into numerical features.
2. **Machine Learning:** Linear Regression wrapped in `MultiOutputRegressor` predicts calories, protein, carbs and fat together.
3. **Rule-based decision logic:** evaluates calorie limits and expense thresholds to produce advice, a score and a rating.

## System Workflow

```
User input (food, quantity, daily calorie limit)
        -> TF-IDF vectorizer
        -> ML model (nutrition prediction)
        -> Decision logic (expense, calorie check, score & rating)
        -> Output (nutrition details, expense analysis, advice, graph)
```

## Tools & Libraries

Python, Google Colab, Pandas, scikit-learn, Matplotlib

## Getting Started

```bash
git clone <your-repo-url>
cd smart-food-advisor
pip install -r requirements.txt
python food_advisor.py
```

The script is interactive: enter your daily calorie limit, then add food items and quantities.

Supported foods: `pizza`, `burger`, `rice`, `dosa`, `idli`, `apple`, `banana`

## Sample Run

```
Enter your daily calorie limit: 2000

Enter food item: burger
Enter quantity: 2
...
Total Calories: 1211.00 kcal
Total Expense : ₹420
Health Advice : Calorie intake is within your limit.
Expense Advice: High food spending. Try budget-friendly meals.
Overall Meal Health Score: 8 / 10
Meal Rating: Healthy
```

## Future Scope

- User profile personalization for customized health recommendations
- Long-term food tracking to analyze eating habits over time
- Expanded food database for better prediction accuracy
- Mobile or web application integration
