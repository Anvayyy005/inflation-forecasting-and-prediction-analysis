# Global Inflation Forecasting Application

A modern web application for analyzing and predicting global inflation trends with interactive visualizations and category-specific analysis.

![Inflation Dashboard](screenshots/dashboard.png)

## Features

- **Inflation Forecasting**: Predict future inflation rates with statistical modeling
- **Event Impact Analysis**: Examine how major global events affected inflation trends
- **Category Analysis**: Break down inflation by specific economic sectors
- **Interactive Dashboards**: User-friendly interface with modern design
- **Data Visualization**: Clear graphical representation of historical data and predictions

## Technology Stack

- **Backend**: Python, Flask, statsmodels (SARIMAX)
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
   ```
   git clone https://github.com/yourusername/inflation-forecasting-prediction-and-analysis.git
   cd inflation-forecasting-prediction-and-analysis
   ```

2. Create and activate a virtual environment (optional but recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the application
   ```
   python application.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

### Inflation Dashboard
The main dashboard provides an overview of current inflation rates, historical trends, and forecasting tools. Use the form to:
- Select a date range for analysis
- Choose a global event to see its impact on inflation
- Generate forecasts with confidence intervals

### Category Analysis
The category analysis tool allows you to:
- Compare inflation in specific sectors against overall inflation
- Analyze how different categories of goods and services are affected by economic trends
- Generate category-specific reports and visualizations

## Methodology

The forecasting uses Seasonal AutoRegressive Integrated Moving Average with eXogenous factors (SARIMAX) models to capture complex patterns in inflation data. The models are periodically retrained with the latest data to maintain accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inflation data sourced from trusted statistical agencies
- Built with open-source technologies

## Contributors

- Data Science Team
- UI/UX Design Team
- Economic Analysis Team
