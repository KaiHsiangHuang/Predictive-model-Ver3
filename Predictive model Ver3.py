import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import calendar

# Try to import holidays package
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    st.warning("holidays package not installed. Install with: pip install holidays")

# Set page config
st.set_page_config(
    page_title="Integrated Demand Predictor",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
# 🔮 Integrated Demand Prediction Tool with Prebooking Analysis
### Predict Assisted Travel Demand with Real-time Prebooking Adjustments
*Combines seasonal patterns, UK holidays, and live prebooking intelligence*
""")

# ============================================================================
# PREBOOKING ANALYZER CLASS
# ============================================================================

class PrebookingAnalyzer:
    """Analyses relationship between prebookings and final demand"""
    
    def __init__(self):
        self.prebooking_models = {}  # Store models by day of week
        self.prebooking_stats = {}   # Store statistics
        self.day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def analyze_prebooking_patterns(self, historical_data):
        """Analyse prebooking patterns from historical data"""
        
        # First, create prebooking profiles
        st.write("🔍 Analysing prebooking patterns...")
        
        # For each booking, calculate days in advance
        historical_data['days_in_advance'] = (
            historical_data['scheduled_departure_date'] - 
            historical_data['booking_created_date']
        ).dt.days
        
        # Filter valid bookings
        historical_data = historical_data[
            (historical_data['days_in_advance'] >= 0) & 
            (historical_data['days_in_advance'] <= 365)
        ]
        
        # Group by departure date to get cumulative bookings
        prebooking_profiles = []
        
        for departure_date in historical_data['scheduled_departure_date'].dt.date.unique():
            date_bookings = historical_data[
                historical_data['scheduled_departure_date'].dt.date == departure_date
            ]
            
            # Skip if too few bookings
            if len(date_bookings) < 10:
                continue
            
            # Calculate cumulative bookings at each day before departure
            total_bookings = len(date_bookings)
            day_of_week = pd.to_datetime(departure_date).dayofweek
            
            for days_before in range(0, 8):  # 0-7 days before
                bookings_by_cutoff = date_bookings[
                    date_bookings['days_in_advance'] >= days_before
                ]
                count_at_cutoff = len(bookings_by_cutoff)
                percentage_at_cutoff = count_at_cutoff / total_bookings if total_bookings > 0 else 0
                
                prebooking_profiles.append({
                    'departure_date': departure_date,
                    'day_of_week': day_of_week,
                    'days_before': days_before,
                    'bookings_at_cutoff': count_at_cutoff,
                    'total_bookings': total_bookings,
                    'percentage_booked': percentage_at_cutoff
                })
        
        prebooking_df = pd.DataFrame(prebooking_profiles)
        
        # Analyze patterns by day of week
        for dow in range(7):
            dow_data = prebooking_df[prebooking_df['day_of_week'] == dow]
            
            if len(dow_data) < 10:
                continue
            
            # Calculate statistics for each days_before cutoff
            stats_by_days = {}
            
            for days_before in range(0, 8):
                cutoff_data = dow_data[dow_data['days_before'] == days_before]
                
                if len(cutoff_data) > 0:
                    stats_by_days[days_before] = {
                        'mean_percentage': cutoff_data['percentage_booked'].mean(),
                        'std_percentage': cutoff_data['percentage_booked'].std(),
                        'median_percentage': cutoff_data['percentage_booked'].median(),
                        'q25_percentage': cutoff_data['percentage_booked'].quantile(0.25),
                        'q75_percentage': cutoff_data['percentage_booked'].quantile(0.75),
                        'sample_size': len(cutoff_data)
                    }
            
            self.prebooking_stats[dow] = stats_by_days
        
        return prebooking_df
    
    def calculate_prebooking_adjustment(self, day_of_week, current_prebookings, 
                                      base_prediction, days_until_departure):
        """Calculate adjustment factor based on prebooking levels"""
        
        if day_of_week not in self.prebooking_stats:
            return 1.0, "No prebooking model available", None
        
        stats = self.prebooking_stats[day_of_week].get(days_until_departure, {})
        
        if not stats or stats.get('sample_size', 0) < 10:
            return 1.0, "Insufficient historical data", None
        
        # Expected prebookings based on base prediction and historical patterns
        expected_percentage = stats['mean_percentage']
        expected_prebookings = base_prediction * expected_percentage
        
        # Calculate predicted final demand based on current prebookings
        if expected_percentage > 0:
            predicted_final_demand = current_prebookings / expected_percentage
        else:
            predicted_final_demand = base_prediction
        
        # Calculate how much above/below the base prediction
        if base_prediction > 0:
            adjustment_ratio = predicted_final_demand / base_prediction
        else:
            adjustment_ratio = 1.0
        
        # Prepare detailed analysis
        analysis = {
            'expected_prebookings': expected_prebookings,
            'actual_prebookings': current_prebookings,
            'expected_percentage': expected_percentage,
            'predicted_final_demand': predicted_final_demand,
            'base_prediction': base_prediction,
            'adjustment_ratio': adjustment_ratio
        }
        
        # Create explanation
        if adjustment_ratio > 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Predicting {predicted_final_demand:.0f} total"
        elif adjustment_ratio < 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Lower demand signal"
        else:
            explanation = "Prebookings match expectations"
        
        return adjustment_ratio, explanation, analysis

# ============================================================================
# HOLIDAY PATTERN CLASS (from original)
# ============================================================================

class AssistedTravelHolidayPatterns:
    """Handles UK bank holiday patterns specific to assisted travel customers"""
    
    def __init__(self):
        # Get base holidays from package if available
        current_year = datetime.now().year
        if HOLIDAYS_AVAILABLE:
            self.uk_holidays = holidays.UK(years=range(current_year-2, current_year+2))
        else:
            # Manual fallback for key holidays
            self.uk_holidays = self._get_manual_holidays()
        
        # Define patterns specific to assisted travel
        self.pattern_templates = {
            'easter': {
                'good_friday': {
                    -2: 1.3,   # Thu before - outbound travel
                    -1: 1.35,  # Day before (Thu) - peak outbound
                    0: 1.3,    # Good Friday - still high
                    1: 0.85,   # Saturday - low (middle of holiday)
                    2: 0.8     # Easter Sunday - lowest
                },
                'easter_monday': {
                    -2: 0.8,   # Saturday - low
                    -1: 0.85,  # Sunday - low
                    0: 1.4,    # Easter Monday - return surge
                    1: 1.35,   # Tuesday - continued returns
                    2: 1.2     # Wednesday - normalizing
                }
            },
            'standard_monday': {  # Most bank holidays are Mondays
                -3: 1.2,   # Friday before - early outbound
                -2: 1.25,  # Weekend before - moderate
                -1: 1.0,   # Sunday - lower
                0: 0.9,    # Monday holiday - lower
                1: 1.4,    # Tuesday - return surge
                2: 1.25    # Wednesday - still elevated
            },
            'christmas': {
                -3: 1.35,  # Pre-Christmas travel
                -2: 1.4,   # Peak pre-Christmas
                -1: 1.45,  # Christmas Eve travel
                0: 0.7,    # Christmas Day - minimal
                1: 0.8,    # Boxing Day - low
                2: 1.1,    # 27th - some returns
                3: 1.2     # 28th - more returns
            },
            'new_year': {
                -2: 1.2,   # 30th Dec - outbound
                -1: 1.25,  # NYE - moderate
                0: 0.8,    # New Year's Day - low
                1: 1.1,    # 2nd Jan - returns begin
                2: 1.3     # 3rd Jan - return surge
            }
        }
    
    def _get_manual_holidays(self):
        """Manual holiday definitions as fallback"""
        current_year = datetime.now().year
        holidays_dict = {}
        
        # Add holidays for current and next year
        for year in [current_year, current_year + 1]:
            holidays_dict.update({
                pd.Timestamp(f'{year}-01-01'): "New Year's Day",
                pd.Timestamp(f'{year}-12-25'): 'Christmas Day',
                pd.Timestamp(f'{year}-12-26'): 'Boxing Day'
            })
            
            # Add Easter (approximate - would need proper calculation)
            if year == 2025:
                holidays_dict.update({
                    pd.Timestamp('2025-04-18'): 'Good Friday',
                    pd.Timestamp('2025-04-21'): 'Easter Monday',
                })
            elif year == 2026:
                holidays_dict.update({
                    pd.Timestamp('2026-04-03'): 'Good Friday',
                    pd.Timestamp('2026-04-06'): 'Easter Monday',
                })
        
        return holidays_dict
    
    def get_holiday_factor(self, check_date, base_prediction=None):
        """Get the appropriate multiplier for a given date"""
        
        # Convert to pandas timestamp if needed
        if not isinstance(check_date, pd.Timestamp):
            check_date = pd.Timestamp(check_date)
        
        # Check proximity to holidays
        best_factor = 1.0
        holiday_name = None
        
        for holiday_date, name in self.uk_holidays.items():
            if not isinstance(holiday_date, pd.Timestamp):
                holiday_date = pd.Timestamp(holiday_date)
                
            days_diff = (check_date - holiday_date).days
            
            # Skip if too far from holiday
            if abs(days_diff) > 5:
                continue
                
            # Get appropriate pattern
            pattern = self.get_pattern_for_holiday(name, holiday_date)
            
            if days_diff in pattern:
                factor = pattern[days_diff]
                
                # Adjust for day of week preferences
                if check_date.weekday() in [5, 6]:  # Weekend
                    factor *= 0.9  # Assisted travel avoids weekends
                elif check_date.weekday() in [1, 2, 3]:  # Tue-Thu
                    factor *= 1.05  # Slight boost for preferred days
                
                # Take the maximum factor if multiple holidays affect this date
                if factor != 1.0 and abs(factor - 1.0) > abs(best_factor - 1.0):
                    best_factor = factor
                    holiday_name = name
        
        return best_factor, holiday_name
    
    def get_pattern_for_holiday(self, holiday_name, holiday_date):
        """Match holiday to appropriate pattern template"""
        
        holiday_name = str(holiday_name)
        
        if "Good Friday" in holiday_name:
            return self.pattern_templates['easter']['good_friday']
        elif "Easter Monday" in holiday_name:
            return self.pattern_templates['easter']['easter_monday']
        elif "Christmas" in holiday_name:
            return self.pattern_templates['christmas']
        elif "New Year" in holiday_name:
            return self.pattern_templates['new_year']
        elif holiday_date.weekday() == 0:  # Monday holiday
            return self.pattern_templates['standard_monday']
        else:
            # Default bank holiday pattern
            return {-2: 1.2, -1: 1.25, 0: 1.1, 1: 1.25, 2: 1.2}

# ============================================================================
# SIDEBAR WITH PREBOOKING INPUTS
# ============================================================================

# Sidebar
with st.sidebar:
    st.header("📁 Historical Data")
    
    uploaded_files = st.file_uploader(
        "Upload Historical Database CSV Files",
        accept_multiple_files=True,
        type="csv",
        help="Upload 2+ years of historical data with booking_created_date column"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    st.header("📅 Prediction Period")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date(),
            help="Select the first day of your 7-day forecast"
        )
    
    with col2:
        # Automatically set end date to 6 days after start
        end_date = start_date + timedelta(days=6)
        st.date_input(
            "End Date",
            value=end_date,
            disabled=True,
            help="Automatically set to 7 days"
        )
    
    st.info(f"📊 Forecasting {start_date.strftime('%a %d %b')} to {end_date.strftime('%a %d %b %Y')}")
    
    # Check if forecast is within next 7 days
    days_until_start = (start_date - datetime.now().date()).days
    enable_prebooking = days_until_start >= 0 and days_until_start <= 7
    
    # Prebooking inputs section
    st.header("📋 Current Prebookings")
    
    if enable_prebooking:
        st.markdown("*Enter current prebooking numbers (optional)*")
        
        prebooking_inputs = {}
        for i in range(7):
            date = start_date + timedelta(days=i)
            days_until = (date - datetime.now().date()).days
            
            if days_until > 0:
                prebooking_inputs[date] = st.number_input(
                    f"{date.strftime('%a %d/%m')} ({days_until}d away)",
                    min_value=0,
                    value=0,
                    help=f"Current prebookings for {date.strftime('%A')}"
                )
        
        use_prebooking = st.checkbox(
            "Apply Prebooking Analysis",
            value=True,
            help="Use prebooking patterns to refine predictions"
        )
    else:
        st.warning("Prebooking analysis only available for predictions within next 7 days")
        prebooking_inputs = {}
        use_prebooking = False
    
    st.header("⚙️ Model Settings")
    
    confidence_level = st.slider(
        "Confidence Interval (%)",
        80, 99, 95,
        help="Confidence level for prediction intervals"
    )
    
    apply_holidays = st.checkbox(
        "Apply UK Holiday Patterns",
        value=True,
        help="Apply assisted travel holiday behavior"
    )
    
    show_components = st.checkbox(
        "Show Prediction Components",
        value=True,
        help="Break down how each prediction is calculated"
    )
    
    st.header("🎯 Operational Thresholds")
    
    low_threshold = st.number_input(
        "Low Demand Threshold",
        min_value=0,
        value=150,
        help="Below this = low staffing needed"
    )
    
    high_threshold = st.number_input(
        "High Demand Threshold", 
        min_value=0,
        value=300,
        help="Above this = full staffing needed"
    )

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_historical_data(uploaded_files):
    """Process historical data for training - handles booking_created_date if present"""
    if not uploaded_files:
        return None, None
    
    yearly_data = {}
    has_prebooking_data = False
    
    for file in uploaded_files:
        try:
            year = int(file.name.split()[0])
            df = pd.read_csv(file, low_memory=False)
            
            # Filter for Euston
            df_euston = df[df['station_code'] == "EUS"].copy()
            
            # Check if prebooking data is available
            if 'booking_created_date' in df_euston.columns:
                has_prebooking_data = True
            
            if len(df_euston) > 0:
                yearly_data[year] = df_euston
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    if not yearly_data:
        return None, None
    
    # Process each year
    processed_years = {}
    
    for year, df in yearly_data.items():
        # Convert dates
        df['scheduled_departure_date'] = pd.to_datetime(
            df['scheduled_departure_date'], 
            dayfirst=True,
            errors='coerce'
        )
        
        # Convert booking_created_date if present
        if 'booking_created_date' in df.columns:
            df['booking_created_date'] = pd.to_datetime(
                df['booking_created_date'],
                dayfirst=True,
                errors='coerce'
            )
        
        df = df.dropna(subset=['scheduled_departure_date'])
        
        # Create daily bookings
        daily_bookings = df.groupby('scheduled_departure_date').size().reset_index(name='bookings')
        
        # Full year date range
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')
        
        # Handle partial years
        if daily_bookings['scheduled_departure_date'].min() > year_start:
            year_start = daily_bookings['scheduled_departure_date'].min()
        if daily_bookings['scheduled_departure_date'].max() < year_end:
            year_end = daily_bookings['scheduled_departure_date'].max()
        
        full_dates = pd.date_range(start=year_start, end=year_end, freq='D')
        full_dates_df = pd.DataFrame({'scheduled_departure_date': full_dates})
        
        yearly_bookings = full_dates_df.merge(daily_bookings, on='scheduled_departure_date', how='left')
        yearly_bookings['bookings'] = yearly_bookings['bookings'].fillna(0)
        yearly_bookings['year'] = year
        yearly_bookings['day_of_year'] = yearly_bookings['scheduled_departure_date'].dt.dayofyear
        yearly_bookings['day_of_week'] = yearly_bookings['scheduled_departure_date'].dt.dayofweek
        yearly_bookings['month'] = yearly_bookings['scheduled_departure_date'].dt.month
        yearly_bookings['week_of_year'] = yearly_bookings['scheduled_departure_date'].dt.isocalendar().week
        
        processed_years[year] = yearly_bookings
    
    # Combine all raw data if prebooking analysis needed
    if has_prebooking_data:
        all_raw_data = pd.concat([yearly_data[year] for year in yearly_data.keys()])
        return processed_years, all_raw_data
    
    return processed_years, None

def train_prediction_model(processed_years):
    """Train the base model on all historical data"""
    
    # Combine all historical data
    all_historical = pd.concat(processed_years.values())
    
    # Calculate growth trend
    yearly_totals = []
    for year, data in processed_years.items():
        total = data['bookings'].sum()
        yearly_totals.append((year, total))
    
    years = [x[0] for x in yearly_totals]
    totals = [x[1] for x in yearly_totals]
    
    # Calculate growth factor (adaptive method)
    if len(years) >= 2:
        # Year-over-year growth rates
        growth_rates = []
        for i in range(1, len(yearly_totals)):
            prev_total = yearly_totals[i-1][1]
            curr_total = yearly_totals[i][1]
            growth = (curr_total - prev_total) / prev_total if prev_total > 0 else 0
            growth_rates.append(growth)
        
        # Weight recent years more
        if len(growth_rates) > 1:
            weights = [0.7 ** (len(growth_rates) - i - 1) for i in range(len(growth_rates))]
            weights = [w / sum(weights) for w in weights]
            avg_growth = sum(g * w for g, w in zip(growth_rates, weights))
        else:
            avg_growth = growth_rates[0]
        
        # Project to current year
        years_ahead = datetime.now().year - years[-1]
        growth_factor = (1 + avg_growth) ** years_ahead
    else:
        growth_factor = 1.1
    
    # Calculate seasonal baseline (median approach)
    seasonal_baseline = all_historical.groupby('day_of_year')['bookings'].median().to_dict()
    
    # Fill missing days with interpolation
    all_days = range(1, 367)
    for day in all_days:
        if day not in seasonal_baseline:
            # Find nearest days
            nearby_days = [d for d in seasonal_baseline.keys() if abs(d - day) <= 7]
            if nearby_days:
                seasonal_baseline[day] = np.mean([seasonal_baseline[d] for d in nearby_days])
            else:
                seasonal_baseline[day] = all_historical['bookings'].median()
    
    # Calculate day-of-week factors
    dow_medians = all_historical.groupby('day_of_week')['bookings'].median()
    overall_median = all_historical['bookings'].median()
    dow_factors = (dow_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    # Calculate monthly factors
    monthly_medians = all_historical.groupby('month')['bookings'].median()
    monthly_factors = (monthly_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    # Calculate prediction intervals based on historical variance
    daily_std = all_historical.groupby('day_of_year')['bookings'].std().to_dict()
    
    model = {
        'seasonal_baseline': seasonal_baseline,
        'dow_factors': dow_factors,
        'monthly_factors': monthly_factors,
        'growth_factor': growth_factor,
        'daily_std': daily_std,
        'overall_std': all_historical['bookings'].std(),
        'years_trained': years,
        'total_days': len(all_historical)
    }
    
    return model

def predict_future_demand_with_prebooking(model, prebooking_analyzer, start_date, 
                                         prebooking_inputs=None, apply_holidays=True, 
                                         confidence_level=95, use_prebooking=False):
    """Generate predictions with optional prebooking adjustments"""
    
    # Initialize holiday patterns
    holiday_model = AssistedTravelHolidayPatterns() if apply_holidays else None
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=7, freq='D')
    
    predictions = []
    
    for date in dates:
        # Get base components
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek
        month = date.month
        
        # Seasonal baseline
        base_value = model['seasonal_baseline'].get(day_of_year, 
                                                   np.mean(list(model['seasonal_baseline'].values())))
        
        # Apply factors
        dow_factor = model['dow_factors'].get(day_of_week, 1.0)
        month_factor = model['monthly_factors'].get(month, 1.0)
        
        # Holiday factor
        holiday_factor = 1.0
        holiday_name = None
        if holiday_model:
            holiday_factor, holiday_name = holiday_model.get_holiday_factor(date)
        
        # Combine all factors for base prediction
        base_prediction = base_value * dow_factor * month_factor * model['growth_factor'] * holiday_factor
        
        # Calculate confidence intervals
        std_dev = model['daily_std'].get(day_of_year, model['overall_std'])
        z_score = stats.norm.ppf((1 + confidence_level/100) / 2)
        margin = z_score * std_dev * 0.5  # Reduce margin for more realistic intervals
        
        lower_bound = max(0, base_prediction - margin)
        upper_bound = base_prediction + margin
        
        # Initialize prebooking analysis
        prebooking_adjustment = 1.0
        prebooking_flag = None
        prebooking_analysis = None
        final_prediction = base_prediction
        
        # Apply prebooking analysis if enabled and available
        if use_prebooking and prebooking_analyzer and prebooking_inputs:
            current_prebookings = prebooking_inputs.get(date.date(), 0)
            days_until = (date.date() - datetime.now().date()).days
            
            if current_prebookings > 0 and days_until > 0 and days_until <= 7:
                adjustment_ratio, explanation, analysis = prebooking_analyzer.calculate_prebooking_adjustment(
                    day_of_week, current_prebookings, base_prediction, days_until
                )
                
                prebooking_analysis = analysis
                predicted_from_prebooking = analysis['predicted_final_demand'] if analysis else base_prediction
                
                # Apply business rules
                if adjustment_ratio < 1.0:
                    # Lower than expected - just flag it
                    prebooking_flag = f"⚠️ Lower demand signal: {explanation}"
                    final_prediction = base_prediction  # Keep original
                else:
                    # Higher than expected
                    if predicted_from_prebooking <= upper_bound:
                        # Within confidence interval - update prediction
                        final_prediction = predicted_from_prebooking
                        prebooking_flag = f"✅ Updated based on prebookings: {explanation}"
                        prebooking_adjustment = adjustment_ratio
                    else:
                        # Outside confidence interval - keep original but flag
                        final_prediction = base_prediction
                        prebooking_flag = f"🔴 High prebooking signal (exceeds confidence): {explanation}"
        
        predictions.append({
            'date': date,
            'day_name': date.strftime('%A'),
            'base_prediction': round(base_prediction),
            'final_prediction': round(final_prediction),
            'lower_bound': round(lower_bound),
            'upper_bound': round(upper_bound),
            'base_value': round(base_value),
            'dow_factor': round(dow_factor, 3),
            'month_factor': round(month_factor, 3),
            'holiday_factor': round(holiday_factor, 3),
            'holiday_name': holiday_name,
            'growth_factor': round(model['growth_factor'], 3),
            'prebooking_adjustment': round(prebooking_adjustment, 3),
            'prebooking_flag': prebooking_flag,
            'prebooking_analysis': prebooking_analysis
        })
    
    return pd.DataFrame(predictions)

def categorize_demand(value, low_threshold, high_threshold):
    """Categorize demand level for operational planning"""
    if value < low_threshold:
        return "Low", "🟢"
    elif value < high_threshold:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    try:
        if uploaded_files:
            # Process historical data
            with st.spinner("Processing historical data..."):
                processed_years, raw_data = process_historical_data(uploaded_files)
                
                if processed_years is None or len(processed_years) < 2:
                    st.error("Need at least 2 years of historical data for reliable predictions")
                    return
            
            # Train base model
            with st.spinner("Training prediction model..."):
                model = train_prediction_model(processed_years)
            
            # Initialize prebooking analyzer if data available
            prebooking_analyzer = None
            if raw_data is not None and 'booking_created_date' in raw_data.columns:
                with st.spinner("Analyzing prebooking patterns..."):
                    prebooking_analyzer = PrebookingAnalyzer()
                    prebooking_analyzer.analyze_prebooking_patterns(raw_data)
                    st.success("✅ Prebooking patterns analyzed")
            else:
                if use_prebooking:
                    st.warning("⚠️ No booking_created_date column found - prebooking analysis unavailable")
            
            # Show model info
            st.success(f"✅ Model trained on {len(model['years_trained'])} years of data ({min(model['years_trained'])}-{max(model['years_trained'])})")
            
            # Generate predictions
            with st.spinner("Generating predictions..."):
                predictions_df = predict_future_demand_with_prebooking(
                    model, 
                    prebooking_analyzer,
                    start_date, 
                    prebooking_inputs=prebooking_inputs if use_prebooking else None,
                    apply_holidays=apply_holidays,
                    confidence_level=confidence_level,
                    use_prebooking=use_prebooking and prebooking_analyzer is not None
                )
            
            # Display predictions
            st.header("📊 7-Day Demand Forecast")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_predicted = predictions_df['final_prediction'].sum()
            avg_daily = predictions_df['final_prediction'].mean()
            peak_day = predictions_df.loc[predictions_df['final_prediction'].idxmax()]
            low_day = predictions_df.loc[predictions_df['final_prediction'].idxmin()]
            
            with col1:
                st.metric("Total Week Demand", f"{total_predicted:,.0f}")
            with col2:
                st.metric("Average Daily", f"{avg_daily:.0f}")
            with col3:
                st.metric("Peak Day", f"{peak_day['day_name'][:3]} ({peak_day['final_prediction']:,.0f})")
            with col4:
                st.metric("Lowest Day", f"{low_day['day_name'][:3]} ({low_day['final_prediction']:,.0f})")
            
            # Visualization
            fig = go.Figure()
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=predictions_df['date'],
                y=predictions_df['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions_df['date'],
                y=predictions_df['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name=f'{confidence_level}% Confidence Interval',
                fillcolor='rgba(68, 185, 255, 0.2)'
            ))
            
            # Base prediction line (if different from final)
            if use_prebooking and any(predictions_df['base_prediction'] != predictions_df['final_prediction']):
                fig.add_trace(go.Scatter(
                    x=predictions_df['date'],
                    y=predictions_df['base_prediction'],
                    mode='lines+markers',
                    name='Base Prediction',
                    line=dict(color='gray', width=2, dash='dot'),
                    marker=dict(size=8)
                ))
            
            # Final prediction line
            fig.add_trace(go.Scatter(
                x=predictions_df['date'],
                y=predictions_df['final_prediction'],
                mode='lines+markers',
                name='Final Prediction',
                line=dict(color='rgb(31, 119, 180)', width=3),
                marker=dict(size=10)
            ))
            
            # Add threshold lines
            fig.add_hline(y=low_threshold, line_dash="dash", line_color="green", 
                         annotation_text="Low Threshold")
            fig.add_hline(y=high_threshold, line_dash="dash", line_color="red",
                         annotation_text="High Threshold")
            
            # Mark holidays
            holiday_dates = predictions_df[predictions_df['holiday_name'].notna()]
            if not holiday_dates.empty:
                fig.add_trace(go.Scatter(
                    x=holiday_dates['date'],
                    y=holiday_dates['final_prediction'],
                    mode='markers',
                    name='Holiday Impact',
                    marker=dict(size=15, symbol='star', color='gold'),
                    text=holiday_dates['holiday_name'],
                    hovertemplate='%{text}<br>Demand: %{y}<extra></extra>'
                ))
            
            fig.update_layout(
                title="7-Day Demand Forecast with Prebooking Intelligence",
                xaxis_title="Date",
                yaxis_title="Predicted Bookings",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prebooking Analysis Results
            if use_prebooking and any(predictions_df['prebooking_flag'].notna()):
                st.header("📋 Prebooking Analysis Results")
                
                for _, row in predictions_df.iterrows():
                    if row['prebooking_flag']:
                        date_str = row['date'].strftime('%A %d %b')
                        st.write(f"**{date_str}**: {row['prebooking_flag']}")
                        
                        if row['prebooking_analysis']:
                            analysis = row['prebooking_analysis']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Prebookings", 
                                         f"{analysis['actual_prebookings']:.0f}")
                            with col2:
                                st.metric("Expected Prebookings", 
                                         f"{analysis['expected_prebookings']:.0f}")
                            with col3:
                                st.metric("Implied Final Demand", 
                                         f"{analysis['predicted_final_demand']:.0f}")
            
            # Detailed predictions table
            st.header("📋 Detailed Daily Predictions")
            
            # Prepare display dataframe
            display_df = predictions_df.copy()
            
            # Add demand category
            display_df['Demand Level'] = display_df['final_prediction'].apply(
                lambda x: categorize_demand(x, low_threshold, high_threshold)[1] + " " + 
                         categorize_demand(x, low_threshold, high_threshold)[0]
            )
            
            # Format columns
            display_df['Date'] = display_df['date'].dt.strftime('%a %d %b')
            display_df['Base Prediction'] = display_df['base_prediction'].apply(lambda x: f"{x:,.0f}")
            display_df['Final Prediction'] = display_df['final_prediction'].apply(lambda x: f"{x:,.0f}")
            display_df['Range'] = display_df.apply(
                lambda x: f"{x['lower_bound']:,.0f} - {x['upper_bound']:,.0f}", axis=1
            )
            display_df['Holiday'] = display_df['holiday_name'].fillna('-')
            display_df['Prebooking Adj'] = display_df.apply(
                lambda x: f"{x['prebooking_adjustment']:.2f}x" if x['prebooking_adjustment'] != 1.0 else "-", 
                axis=1
            )
            
            # Select columns to display
            display_columns = ['Date', 'Base Prediction', 'Final Prediction', 'Range', 
                              'Demand Level', 'Holiday', 'Prebooking Adj']
            
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
            
            # Component breakdown
            if show_components:
                st.header("🔧 Prediction Components Breakdown")
                
                components_df = predictions_df[[
                    'date', 'base_value', 'dow_factor', 'month_factor', 
                    'holiday_factor', 'growth_factor', 'prebooking_adjustment', 'final_prediction'
                ]].copy()
                
                components_df['Date'] = components_df['date'].dt.strftime('%a %d %b')
                components_df = components_df.drop('date', axis=1)
                
                # Rename columns
                components_df.columns = [
                    'Date', 'Seasonal Base', 'Day of Week', 'Monthly', 
                    'Holiday', 'Growth', 'Prebooking', 'Final Prediction'
                ]
                
                st.dataframe(
                    components_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Explanation
                with st.expander("📖 Understanding the Components"):
                    st.markdown("""
                    **How predictions are calculated:**
                    
                    1. **Seasonal Base**: Historical median demand for this day of year
                    2. **Day of Week**: Multiplier based on weekday patterns (Tue-Thu typically higher)
                    3. **Monthly**: Seasonal adjustment by month
                    4. **Holiday**: UK bank holiday effects (bookend pattern for assisted travel)
                    5. **Growth**: Year-over-year growth projection
                    6. **Prebooking**: Adjustment based on current booking levels
                    7. **Final = Base × DoW × Monthly × Holiday × Growth × Prebooking**
                    
                    **Prebooking Logic:**
                    - If prebookings indicate **lower** demand → Flag only (no adjustment)
                    - If prebookings indicate **higher** demand:
                      - Within confidence interval → Update prediction
                      - Outside confidence interval → Flag only
                    """)
            
            # Operational recommendations
            st.header("🎯 Operational Recommendations")
            
            high_days = predictions_df[predictions_df['final_prediction'] > high_threshold]
            low_days = predictions_df[predictions_df['final_prediction'] < low_threshold]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚠️ High Demand Days")
                if not high_days.empty:
                    for _, day in high_days.iterrows():
                        st.warning(f"**{day['date'].strftime('%A %d %b')}**: {day['final_prediction']:,.0f} bookings expected")
                    st.info("💡 Ensure full staffing and additional resources")
                else:
                    st.success("No days exceed high threshold")
            
            with col2:
                st.subheader("✅ Low Demand Days")
                if not low_days.empty:
                    for _, day in low_days.iterrows():
                        st.info(f"**{day['date'].strftime('%A %d %b')}**: {day['final_prediction']:,.0f} bookings expected")
                    st.success("💡 Opportunity for training or maintenance")
                else:
                    st.info("No days below low threshold")
            
            # Export functionality
            st.header("📥 Export Predictions")
            
            # Prepare export data
            export_df = predictions_df[[
                'date', 'base_prediction', 'final_prediction', 'lower_bound', 
                'upper_bound', 'holiday_name', 'prebooking_flag'
            ]].copy()
            export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
            export_df.columns = ['Date', 'Base_Prediction', 'Final_Prediction', 
                                'Lower_Bound', 'Upper_Bound', 'Holiday', 'Prebooking_Note']
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"integrated_forecast_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("👆 Please upload historical data files to begin forecasting")
            
            with st.expander("📚 How to Use This Tool"):
                st.markdown("""
                **Step 1: Upload Historical Data**
                - Upload at least 2 years of booking data CSV files
                - Files should include `booking_created_date` column for prebooking analysis
                - Files should be named with the year (e.g., "2023 Database.csv")
                
                **Step 2: Select Forecast Period**
                - Choose your start date (must be in the future)
                - The tool automatically forecasts 7 days ahead
                - Prebooking inputs are available only for predictions within next 7 days
                
                **Step 3: Enter Prebooking Data (Optional)**
                - If forecasting within 7 days, enter current prebooking numbers
                - The system will analyze if demand is trending higher or lower
                - Predictions are adjusted based on prebooking patterns
                
                **Step 4: Review Results**
                - Check the forecast chart with confidence intervals
                - Review prebooking flags for each day
                - Note any holiday impacts
                - Use operational recommendations for planning
                
                **Prebooking Intelligence:**
                - 🟢 **Lower demand signal**: Prebookings below expected (no adjustment)
                - ✅ **Updated prediction**: Higher prebookings within confidence (adjusted up)
                - 🔴 **High demand flag**: Very high prebookings (flagged but not adjusted)
                """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
