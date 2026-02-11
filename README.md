# Flight Price Tracker ✈️

A comprehensive "Best Time to Buy" flight price tracking and analysis tool. Track round-trip flight prices, identify cheapest days to depart, calculate optimal booking lead times, and get intelligent recommendations for your travel plans.

## Features

- **364-Day Price Tracking**: Monitor flight prices across a rolling 364-day window
- **Weekday Analysis**: Identify the cheapest days of the week to depart
- **Optimal Booking Window**: Calculate the best time to book (e.g., "45 days before departure")
- **Deal Detection**: Automatic alerts when prices drop >15% below average
- **Interactive Dashboard**: Beautiful Streamlit-based web interface
- **Multi-Route Support**: Track multiple routes simultaneously
- **Price Predictions**: Simple linear regression-based price forecasts
- **Export Functionality**: Download data as CSV for further analysis

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd FlightScraper

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The `.env` file is already configured with your Amadeus API credentials:

```env
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret
```

### 3. Initialize the Database

```bash
python main.py init
```

### 4. Add a Route to Track

```bash
python main.py add-route JFK MEX --duration 7 --month 2026-10
```

### 5. Collect Price Data

```bash
# Collect for a specific route
python main.py collect --origin JFK --dest MEX

# Or collect for all tracked routes
python main.py collect-all
```

### 6. Analyze Your Route

```bash
python main.py analyze JFK MEX --month 2026-10
```

### 7. Launch the Dashboard

```bash
python main.py dashboard
```

This will open the interactive Streamlit dashboard at `http://localhost:8501`

## CLI Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `init` | Initialize the database | `python main.py init` |
| `add-route` | Add a route to track | `python main.py add-route JFK MEX --duration 7` |
| `collect` | Fetch prices for a route | `python main.py collect --origin JFK --dest MEX` |
| `collect-all` | Fetch prices for all routes | `python main.py collect-all` |
| `analyze` | Analyze a route | `python main.py analyze JFK MEX --month 2026-10` |
| `list-routes` | Show tracking targets | `python main.py list-routes` |
| `export` | Export data to CSV | `python main.py export --route JFK-MEX --output prices.csv` |
| `schedule` | Start automated scheduler | `python main.py schedule --time 02:00` |
| `dashboard` | Launch Streamlit app | `python main.py dashboard` |
| `status` | Show system status | `python main.py status` |

## Project Structure

```
FlightScraper/
├── .env                      # API credentials (DO NOT COMMIT)
├── requirements.txt          # Python dependencies
├── flight_tracker.db         # SQLite database (auto-created)
├── main.py                   # CLI orchestrator
├── config.py                 # Configuration and constants
├── database.py               # Database initialization and utilities
├── models.py                 # Pydantic data models
├── data_fetcher.py           # Amadeus API integration
├── data_logger.py            # Database write operations
├── analyzer.py               # Trend analysis and recommendations
├── visualizer.py             # Chart generation (matplotlib/plotly)
├── app.py                    # Streamlit dashboard
├── scheduler.py              # Automated daily collection
└── utils.py                  # Validation, date helpers, etc.
```

## Dashboard Pages

### 1. Route Analysis
- Enter origin, destination, and target month
- View cheapest prices, best dates, and optimal booking windows
- Interactive charts for weekday comparison and lead time analysis
- Deal detection and recommendations

### 2. Multi-Route Tracker
- View all tracked routes in one place
- Add/remove routes
- Quick access to analysis for each route
- Batch collection controls

### 3. Historical Trends
- View price trends over time
- Volatility analysis
- Compare historical data periods

### 4. Admin Dashboard
- Database statistics
- Manual data collection with progress tracking
- API status monitoring
- Data export functionality

## Analysis Features

### Weekday Analysis
Calculates average prices for each day of the week and identifies potential savings:
- "Flying on Tuesday is 15% cheaper than Sunday"

### Lead Time Analysis
Groups prices by booking window (0-7 days, 8-14 days, etc.) to find the optimal time to book:
- "Book 45-60 days before departure for best prices"

### Deal Detection
Compares current prices to 30-day averages and flags significant drops:
- "October 15 departure is 20% below average - deal alert!"

### Price Predictions
Uses linear regression on historical lead_time vs price data:
- "Prices are predicted to increase by $50 in the next 7 days"

## Supported Airports

The tracker supports 100+ major international airports including:
- **North America**: JFK, LAX, ORD, SFO, MIA, ATL, etc.
- **Mexico**: MEX, CUN, GDL, SJD, PVR
- **Europe**: LHR, CDG, AMS, FRA, BCN, FCO, etc.
- **Asia**: NRT, HND, ICN, SIN, BKK, HKG, etc.
- **And many more...**

## Rate Limiting

The Amadeus API has rate limits. The tracker:
- Adds a 0.3-second delay between requests
- Implements exponential backoff on failures
- Provides rate limit monitoring in the dashboard

## Scheduling

To run automated daily collection:

```bash
# Start scheduler (runs at 2 AM by default)
python main.py schedule --time 02:00

# Run immediately and then continue scheduling
python main.py schedule --time 02:00 --run-now
```

Or use the scheduler directly:

```bash
python scheduler.py --time 03:00 --run-now
```

## Data Storage

All data is stored in a local SQLite database (`flight_tracker.db`). The database includes:

- **flight_snapshots**: Price data with airline, stops, duration, lead time
- **tracking_targets**: Routes being monitored
- **price_alerts**: Historical price drop alerts

## Tips for Best Results

1. **Start Early**: Begin tracking routes at least 3-6 months before travel
2. **Regular Collection**: Run daily collection to build historical data
3. **Multiple Durations**: Track the same route with different trip lengths
4. **Watch Patterns**: Weekday and lead time patterns become clearer with more data
5. **Set Alerts**: Use the notify_below_price feature for price drop alerts

## Troubleshooting

### "No data available"
Run `python main.py collect --origin XXX --dest YYY` to fetch initial data.

### API Connection Failed
Check your `.env` file has valid Amadeus credentials.

### Database Locked
Close any other processes accessing the database, or restart the application.

### Charts Not Rendering
Ensure matplotlib and plotly are installed: `pip install matplotlib plotly`

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

- [Amadeus for Developers](https://developers.amadeus.com/) for the flight search API
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
