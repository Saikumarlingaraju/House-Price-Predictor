# Quick Start Guide 🚀

Welcome to the House Price Predictor! This guide will help you get started quickly.

## Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/Saikumarlingaraju/House-Price-Predictor.git
cd House-Price-Predictor
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

## Training Your First Model

### Quick Training (Hyderabad Example)
```powershell
python train_india.py --csv merged_hyderabad.csv --state Hyderabad --n-estimators 100
```

### Training for Multiple Cities
```powershell
# Train for Bangalore
python train_india.py --csv merged_files.csv --state Bangalore --n-estimators 150

# Train for Mumbai
python train_india.py --csv merged_files.csv --state Mumbai --n-estimators 150
```

## Running the Web App

### Start the App
```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the App

### 1. Predict Price Tab 🏠
- Enter property details (area, bedrooms, location)
- Select amenities
- Click "Predict Price" for instant valuation

### 2. Analytics Dashboard 📊
- View market trends
- Analyze city-wise statistics
- Explore price distributions

### 3. Price Comparison 📈
- Compare multiple predictions
- View prediction history
- Export data to CSV

### 4. Help & Guide ℹ️
- Learn about features
- Understand metrics
- Get tips for better predictions

## Tips for Best Results

1. **Accurate Data**: Provide precise property measurements
2. **Location**: Select the correct city and locality
3. **Amenities**: Include all available features
4. **Recent Models**: Use recently trained models for current market trends

## Common Issues

### Port Already in Use
```powershell
# Check what's using port 8501
netstat -a -n -o | findstr 8501

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Restart the app
streamlit run app.py
```

### Missing Dependencies
```powershell
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Model Not Found
Make sure you've trained a model first:
```powershell
python train_india.py --csv your_data.csv --state YourCity
```

## Advanced Features

### Compare Models
```powershell
python utils.py compare
```

### Export Model Summary
```powershell
python utils.py summary --model model_Hyderabad.pkl
```

### Custom Training Parameters
```powershell
python train_india.py --csv merged_files.csv --state Custom --n-estimators 200
```

## Next Steps

- Check out the [full README](README.md) for detailed documentation
- Visit the [GitHub repository](https://github.com/Saikumarlingaraju/House-Price-Predictor) for updates
- Report issues or request features on GitHub

## Need Help?

- 📧 Email: Check GitHub profile
- 💬 Issues: [GitHub Issues](https://github.com/Saikumarlingaraju/House-Price-Predictor/issues)
- 📖 Docs: [Full Documentation](README.md)

---

Happy Predicting! 🏡✨
