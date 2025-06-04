# Airline Satisfaction Prediction

## About
This project uses machine learning to predict if airline passengers are satisfied or not. It looks at feedback on things like seat comfort, WiFi, and flight delays to help airlines improve their services.

## Dataset
The data (`AirlinesRatingPrediction (2).csv`) includes:
- **Passenger Info**: Gender, Age, Loyal or Disloyal Customer, Business or Personal Travel.
- **Flight Info**: Distance, Delays, Class (Eco, Eco Plus, Business).
- **Ratings**: WiFi, Food, Entertainment, etc. (scored 0-5).
- **Output**: Satisfied or Neutral/Dissatisfied.

## How It Works
- **Data Prep**: Filled missing delay times, converted text to numbers, scaled data.
- **Model**: Used XGBoost with 70% training, 30% testing data.
- **Results**: 
  - Accuracy: [Run `model.py` to get, e.g., ~95%].
  - ROC-AUC: [Run `model.py` to get, e.g., ~0.94].
  - Key Factors: Entertainment, comfort, delays matter most.

## Files
- `app.py`: Web app to enter ratings and predict satisfaction.
- `model.py`: Trains model, saves it as `model.pkl`.
- `AirlinesRatingPrediction (2).csv`: Data for training.
- `model.pkl`, `preprocessor.pkl`: Saved model and data prep files.

## How to Use
1. **Run Model**: Use `model.py` to train and see results.
2. **Web App**: Run `app.py`, go to `http://localhost:5000` to test predictions.
3. **Install**: Need Python, pandas, numpy, scikit-learn, xgboost, flask, seaborn, matplotlib.

## Setup
1. Clone repo: `git clone https://github.com/KerulKidecha234/AirlineSatisfactionPrediction.git`
2. Install packages: `pip install pandas numpy scikit-learn xgboost flask seaborn matplotlib`
3. Run `model.py` or `app.py`.

## Contact
- **Author**: Kidecha Kerul
- **GitHub**: [KerulKidecha234](https://github.com/KerulKidecha234)