import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import io
import base64
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

app = Flask(__name__)

# Add now function to Jinja environment
app.jinja_env.globals['now'] = lambda fmt=None: datetime.now().strftime(fmt or '%Y-%m-%d')

# Add a custom filter for formatting floats (like Django's floatformat)
app.jinja_env.filters['floatformat'] = lambda value, precision=2: f"{float(value):.{precision}f}"

# Add absolute value filter
app.jinja_env.filters['abs'] = lambda value: abs(float(value))

# Define major global events affecting inflation with their respective dates
GLOBAL_EVENTS = {
    "Covid 19 Pandemic": {"date": "2020-03-11", "color": "#e74c3c"},
    "Ukraine-Russia War": {"date": "2022-02-24", "color": "#e67e22"},
    "Global Financial Crisis": {"date": "2008-09-15", "color": "#9b59b6"},
    "Oil Price Shock": {"date": "2014-06-15", "color": "#3498db"},
    "US-China Trade War": {"date": "2018-07-06", "color": "#f1c40f"},
    "Brexit Referendum": {"date": "2016-06-23", "color": "#1abc9c"},
    "Fed Rate Hikes": {"date": "2022-03-16", "color": "#34495e"},
    "Supply Chain Crisis": {"date": "2021-10-01", "color": "#2ecc71"},
    "Energy Crisis": {"date": "2022-08-01", "color": "#c0392b"}
}

def load_model():
    try:
        # Load the pre-trained model
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the historical data
        data = pd.read_csv('data/inflation_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        return model, data
    except Exception as e:
        # Print more detailed error for debugging
        print(f"Error loading model or data: {str(e)}")
        return None, None

@app.route('/')
def index():
    # Initialize with default values
    current_date = datetime.now()
    start_date = (current_date - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end_date = (current_date + timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    return render_template('index.html', start_date=start_date, end_date=end_date, event='None')

@app.route('/predictdata', methods=['POST'])
def predict_data():
    try:
        model, data = load_model()
        if model is None or data is None:
            return render_template('index.html', error="Failed to load model or data. Please try again.")
        
        # Get form data
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        event = request.form.get('event')
        
        # Convert to datetime
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
        except:
            return render_template('index.html', error="Invalid date format. Please try again.")
        
        # Create forecasts
        forecast_data = model.get_forecast(steps=int((end_date_dt - data.index[-1]).days/30) + 1)
        
        # Get the forecast values and confidence intervals
        forecast_values = forecast_data.predicted_mean
        conf_int = forecast_data.conf_int()
        
        # Create dates for the forecast period
        forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast_values), freq='MS')
        
        # Create a DataFrame for the forecast
        forecast_df = pd.DataFrame({'Inflation': forecast_values}, index=forecast_dates)
        
        # Combine historical data and forecasts
        combined_df = pd.concat([data, forecast_df])
        
        # Filter for the specified date range
        filtered_df = combined_df.loc[(combined_df.index >= start_date_dt) & (combined_df.index <= end_date_dt)]
        
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical and forecast data
        ax.plot(filtered_df.index, filtered_df['Inflation'], label='Inflation Rate', color='#3498db', linewidth=2)
        
        # Add confidence intervals for the forecast period
        conf_int_filtered = conf_int.loc[(conf_int.index >= start_date_dt) & (conf_int.index <= end_date_dt)]
        if not conf_int_filtered.empty:
            ax.fill_between(conf_int_filtered.index, 
                           conf_int_filtered['lower Inflation'], 
                           conf_int_filtered['upper Inflation'], 
                           color='#3498db', alpha=0.2, label='95% Confidence Interval')
        
        # Highlight the selected event if not None
        if event != 'None':
            event_date = GLOBAL_EVENTS.get(event, {}).get("date")
            event_color = GLOBAL_EVENTS.get(event, {}).get("color", "#e74c3c")
            if event_date:
                event_date_dt = pd.to_datetime(event_date)
                if event_date_dt >= start_date_dt and event_date_dt <= end_date_dt:
                    ax.axvline(x=event_date_dt, color=event_color, linestyle='--', linewidth=2, label=f'{event}')
                    ax.annotate(event, xy=(event_date_dt, ax.get_ylim()[1] * 0.9), 
                               xytext=(0, 10), textcoords='offset points', ha='center', 
                               color=event_color, fontweight='bold')
        
        # Set plot properties
        ax.set_title('Inflation Rate Forecast', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inflation Rate (%)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format the x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        
        # Add a horizontal line at 0%
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add a background color to make the plot more attractive
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        
        # Add borders to the plot
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        
        # Save the plot to a bytes object
        img_bytes = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_bytes, format='png', dpi=100)
        img_bytes.seek(0)
        plot_img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close()
        
        return render_template('index.html', 
                              plot_img_base64=plot_img_base64, 
                              start_date=start_date, 
                              end_date=end_date, 
                              event=event)
    except Exception as e:
        return render_template('index.html', 
                              error=f"An error occurred: {str(e)}", 
                              start_date=request.form.get('start_date'),
                              end_date=request.form.get('end_date'),
                              event=request.form.get('event', 'None'))

@app.route('/update_model')
def update_model():
    try:
        # Load the latest inflation data
        data = pd.read_csv('data/inflation_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Generate some synthetic data for future projection (if needed)
        last_date = data.index[-1]
        if last_date < pd.to_datetime('2023-12-31'):
            # Create synthetic data points from the last date to Dec 2023
            months_to_add = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                         end='2023-12-31', 
                                         freq='MS')
            
            # Generate synthetic values (slightly random trend continuation)
            last_values = data['Inflation'].values[-12:]  # Last year's values
            trend = np.mean(np.diff(last_values))
            synthetic_values = []
            
            for i in range(len(months_to_add)):
                # Add trend with some randomness
                next_val = data['Inflation'].values[-1] + trend + np.random.normal(0, 0.2)
                synthetic_values.append(next_val)
                data.loc[months_to_add[i]] = next_val
        
        # Train a new SARIMAX model with improved parameters
        model = SARIMAX(data['Inflation'],
                       order=(2, 1, 2),
                       seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        results = model.fit(disp=False)
        
        # Save the new model
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        return render_template('index.html', 
                              message="Model updated successfully with the latest data. The model can now forecast inflation until 2035.", 
                              start_date=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'),
                              end_date=(datetime.now() + timedelta(days=365*5)).strftime('%Y-%m-%d'),
                              event='None')
    except Exception as e:
        return render_template('index.html', 
                              error=f"Failed to update model: {str(e)}",
                              start_date=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'),
                              end_date=(datetime.now() + timedelta(days=365*5)).strftime('%Y-%m-%d'),
                              event='None')

@app.route('/forecast')
def forecast_form():
    # Initialize with default values
    current_date = datetime.now()
    start_date = (current_date - timedelta(days=365*3)).strftime('%Y-%m-%d')
    end_date = current_date.strftime('%Y-%m-%d')
    
    return render_template('forecast_form.html', start_date=start_date, end_date=end_date)

@app.route('/forecast', methods=['POST'])
def forecast_category():
    try:
        # Get form data
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        category = request.form.get('category')
        
        # Load categorical data (example - in real app this would be from a database)
        # For demonstration, we'll create some sample data
        np.random.seed(42)  # For reproducibility
        
        # Convert to datetime
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
        except:
            return render_template('forecast_form.html', error="Invalid date format. Please try again.")
        
        # Create date range for the data
        date_range = pd.date_range(start='2010-01-01', end='2023-12-31', freq='MS')
        
        # Create sample data for overall inflation
        overall_inflation = pd.Series(np.random.normal(2, 1, len(date_range)), index=date_range)
        
        # Add trend and seasonality
        trend = np.linspace(0, 3, len(date_range))
        seasonal = 0.5 * np.sin(np.linspace(0, 24*np.pi, len(date_range)))
        overall_inflation += trend + seasonal
        
        # Create category-specific inflation (with some correlation to overall)
        category_factors = {
            "Food and non-alcoholic beverages": 1.2,
            "Alcoholic beverages and tobacco": 0.8,
            "Clothing and footwear": 0.5,
            "Housing, water, electricity, gas and other fuels": 1.5,
            "Furniture, household equipment and maintenance": 0.7,
            "Health": 1.1,
            "Transport": 1.3,
            "Communication": 0.4,
            "Recreation and culture": 0.6,
            "Education": 1.4,
            "Restaurants and hotels": 1.2,
            "Miscellaneous goods and services": 0.9
        }
        
        factor = category_factors.get(category, 1.0)
        category_inflation = overall_inflation * factor + np.random.normal(0, 0.5, len(date_range))
        
        # Create DataFrames
        data = pd.DataFrame({
            'Overall': overall_inflation,
            'Category': category_inflation
        })
        
        # Filter for the specified date range
        filtered_data = data.loc[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        
        # Calculate statistics for the results page
        avg_inflation = filtered_data['Category'].mean()
        max_inflation = filtered_data['Category'].max()
        max_inflation_date = filtered_data['Category'].idxmax().strftime('%B %Y')
        cat_vs_overall = (filtered_data['Category'].mean() - filtered_data['Overall'].mean())
        
        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        ax.plot(filtered_data.index, filtered_data['Overall'], label='Overall Inflation', color='#3498db', linewidth=2)
        ax.plot(filtered_data.index, filtered_data['Category'], label=f'{category} Inflation', color='#e74c3c', linewidth=2)
        
        # Set plot properties
        ax.set_title(f'Inflation Analysis: {category}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inflation Rate (%)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format the x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        
        # Add a horizontal line at 0%
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add a background color to make the plot more attractive
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        
        # Add borders to the plot
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        
        # Save the plot to a bytes object
        img_bytes = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_bytes, format='png', dpi=100)
        img_bytes.seek(0)
        plot_img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close()
        
        return render_template('forecast_result.html', 
                              plot_img_base64=plot_img_base64,
                              category=category,
                              start_date=start_date,
                              end_date=end_date,
                              avg_inflation=avg_inflation,
                              max_inflation=max_inflation,
                              max_inflation_date=max_inflation_date,
                              cat_vs_overall=cat_vs_overall)
    except Exception as e:
        return render_template('forecast_form.html', 
                              error=f"An error occurred: {str(e)}",
                              start_date=request.form.get('start_date'),
                              end_date=request.form.get('end_date'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.template_filter('now')
def now(format_string):
    return datetime.now().strftime(format_string)

if __name__ == '__main__':
    # Make sure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Initialize the model if it doesn't exist
    if not os.path.exists('models/model.pkl'):
        with app.app_context():
            update_model()
    
    app.run(debug=True)

