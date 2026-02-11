"""
Database initialization and utilities for Flight Price Tracker.

Handles SQLite database operations with proper connection management.
New architecture: Comprehensive calendar search with all flight combinations.
Enhanced with: top-K offers, full timestamps, extended fields, decision engine support.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional, Generator, Any, List, Dict

from config import DATABASE_PATH

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE SCHEMA - ENHANCED ARCHITECTURE
# =============================================================================

SCHEMA_FLIGHT_COMBINATIONS = """
CREATE TABLE IF NOT EXISTS flight_combinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Timestamp (full datetime, not just date)
    scrape_timestamp DATETIME NOT NULL,
    
    -- Route
    departure_airport TEXT NOT NULL,
    arrival_airport TEXT NOT NULL,
    
    -- Exact travel dates
    departure_date DATE NOT NULL,
    return_date DATE NOT NULL,
    trip_duration_days INTEGER NOT NULL,
    
    -- Offer ranking (1=cheapest, 2=second, 3=third)
    offer_rank INTEGER DEFAULT 1,
    
    -- Pricing breakdown
    price REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    base_fare REAL,
    taxes_fees REAL,
    
    -- Baggage info
    included_bags INTEGER DEFAULT 0,
    checked_bag_price REAL,
    carry_on_allowed BOOLEAN DEFAULT TRUE,
    
    -- Availability signals
    available_seats INTEGER,
    last_ticketing_date DATE,
    
    -- Flight details
    airline_outbound TEXT,
    airline_return TEXT,
    stops_outbound INTEGER DEFAULT 0,
    stops_return INTEGER DEFAULT 0,
    duration_outbound_minutes INTEGER,
    duration_return_minutes INTEGER,
    cabin_class TEXT DEFAULT 'economy',
    
    -- Generalized cost components
    overnight_layover BOOLEAN DEFAULT FALSE,
    connection_airports TEXT,  -- JSON array of connection codes
    
    -- Analytics fields
    lead_time_days INTEGER NOT NULL,
    departure_day_of_week TEXT,
    return_day_of_week TEXT,
    is_weekend_departure BOOLEAN,
    is_weekend_return BOOLEAN,
    
    -- Booking reference
    booking_token TEXT,
    deep_link_url TEXT,
    
    -- Metadata
    search_query_id TEXT,
    other_metadata TEXT,
    
    -- Unique constraint includes offer_rank for top-K storage
    UNIQUE(scrape_timestamp, departure_airport, arrival_airport, 
           departure_date, return_date, offer_rank)
);
"""

SCHEMA_SEARCH_PROFILES = """
CREATE TABLE IF NOT EXISTS search_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    target_departure_month TEXT,
    trip_duration_days INTEGER NOT NULL,
    flexibility_days INTEGER DEFAULT 2,
    max_stops INTEGER DEFAULT 2,
    preferred_airlines TEXT,
    max_price_threshold REAL,
    notify_on_drop_percent REAL DEFAULT 10.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_date DATE DEFAULT CURRENT_DATE,
    last_searched DATETIME,
    
    -- User risk profile for decision engine
    risk_profile TEXT DEFAULT 'moderate',  -- aggressive, moderate, conservative
    
    UNIQUE(origin, destination, target_departure_month, trip_duration_days)
);
"""

SCHEMA_PRICE_ALERTS = """
CREATE TABLE IF NOT EXISTS price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_profile_id INTEGER,
    alert_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    departure_date DATE,
    return_date DATE,
    old_price REAL,
    new_price REAL,
    price_change_percent REAL,
    alert_type TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (search_profile_id) REFERENCES search_profiles(id)
);
"""

SCHEMA_ROUTE_BASELINES = """
CREATE TABLE IF NOT EXISTS route_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    month INTEGER,  -- 1-12, NULL for all-months
    day_of_week TEXT,  -- Monday-Sunday, NULL for all-days
    lead_time_bucket TEXT,  -- '0-7', '8-14', '15-30', '31-60', '60+'
    
    -- Quantile distribution
    p10 REAL,
    p25 REAL,
    p50 REAL,
    p75 REAL,
    p90 REAL,
    
    -- Statistics
    mean REAL,
    std_dev REAL,
    sample_count INTEGER,
    last_updated DATETIME,
    
    UNIQUE(origin, destination, month, day_of_week, lead_time_bucket)
);
"""

SCHEMA_BOOKING_OUTCOMES = """
CREATE TABLE IF NOT EXISTS booking_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_profile_id INTEGER,
    departure_date DATE,
    return_date DATE,
    
    -- Decision made
    decision_date DATETIME,
    decision_made TEXT,  -- 'buy', 'wait', 'watch'
    price_at_decision REAL,
    fair_value_at_decision REAL,
    regime_at_decision TEXT,
    
    -- Actual outcome
    booked_date DATETIME,
    booked_price REAL,
    final_lowest_price REAL,  -- Lowest price observed before departure
    
    -- Percentile tracking
    booked_price_percentile REAL,  -- Where booked price fell in distribution
    
    -- Regret analysis
    regret_vs_optimal REAL,  -- booked_price - final_lowest_price
    decision_was_correct BOOLEAN,
    
    FOREIGN KEY (search_profile_id) REFERENCES search_profiles(id)
);
"""

INDEXES = [
    # Flight combinations indexes
    "CREATE INDEX IF NOT EXISTS idx_scrape_route ON flight_combinations(scrape_timestamp, departure_airport, arrival_airport);",
    "CREATE INDEX IF NOT EXISTS idx_travel_dates ON flight_combinations(departure_date, return_date);",
    "CREATE INDEX IF NOT EXISTS idx_lead_time ON flight_combinations(lead_time_days);",
    "CREATE INDEX IF NOT EXISTS idx_price ON flight_combinations(price);",
    "CREATE INDEX IF NOT EXISTS idx_route ON flight_combinations(departure_airport, arrival_airport);",
    "CREATE INDEX IF NOT EXISTS idx_departure_month ON flight_combinations(departure_date);",
    "CREATE INDEX IF NOT EXISTS idx_weekday ON flight_combinations(departure_day_of_week);",
    "CREATE INDEX IF NOT EXISTS idx_offer_rank ON flight_combinations(offer_rank);",
    "CREATE INDEX IF NOT EXISTS idx_scrape_date ON flight_combinations(date(scrape_timestamp));",
    
    # Search profiles indexes
    "CREATE INDEX IF NOT EXISTS idx_profile_route ON search_profiles(origin, destination);",
    "CREATE INDEX IF NOT EXISTS idx_profile_active ON search_profiles(is_active);",
    "CREATE INDEX IF NOT EXISTS idx_profile_month ON search_profiles(target_departure_month);",
    
    # Price alerts indexes
    "CREATE INDEX IF NOT EXISTS idx_alerts_profile ON price_alerts(search_profile_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_date ON price_alerts(alert_date);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type ON price_alerts(alert_type);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_unread ON price_alerts(is_read);",
    
    # Route baselines indexes
    "CREATE INDEX IF NOT EXISTS idx_baseline_route ON route_baselines(origin, destination);",
    "CREATE INDEX IF NOT EXISTS idx_baseline_lookup ON route_baselines(origin, destination, month, lead_time_bucket);",
    
    # Booking outcomes indexes
    "CREATE INDEX IF NOT EXISTS idx_outcomes_profile ON booking_outcomes(search_profile_id);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_dates ON booking_outcomes(departure_date, return_date);",
]

PRAGMAS = [
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA cache_size = -64000;",  # 64MB cache
    "PRAGMA temp_store = MEMORY;",
]


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply configuration settings to a database connection."""
    conn.row_factory = sqlite3.Row
    for pragma in PRAGMAS:
        conn.execute(pragma)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for safe database access.
    
    Automatically handles connection cleanup and commits transactions.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        _configure_connection(conn)
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db() -> bool:
    """
    Initialize the database by creating tables and indexes if they don't exist.
    
    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        with get_db_connection() as conn:
            # Create tables
            conn.execute(SCHEMA_FLIGHT_COMBINATIONS)
            logger.info("Created/verified flight_combinations table")
            
            conn.execute(SCHEMA_SEARCH_PROFILES)
            logger.info("Created/verified search_profiles table")
            
            conn.execute(SCHEMA_PRICE_ALERTS)
            logger.info("Created/verified price_alerts table")
            
            conn.execute(SCHEMA_ROUTE_BASELINES)
            logger.info("Created/verified route_baselines table")
            
            conn.execute(SCHEMA_BOOKING_OUTCOMES)
            logger.info("Created/verified booking_outcomes table")
            
            # Create indexes
            for index_sql in INDEXES:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError as e:
                    # Some indexes may fail due to SQLite limitations
                    logger.debug(f"Index creation note: {e}")
            logger.info("Created/verified all indexes")
            
        logger.info(f"Database initialized successfully at {DATABASE_PATH}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def migrate_db() -> bool:
    """
    Migrate existing database to new schema.
    Adds new columns if they don't exist.
    """
    migrations = [
        # Add offer_rank column
        ("flight_combinations", "offer_rank", "INTEGER DEFAULT 1"),
        ("flight_combinations", "base_fare", "REAL"),
        ("flight_combinations", "taxes_fees", "REAL"),
        ("flight_combinations", "included_bags", "INTEGER DEFAULT 0"),
        ("flight_combinations", "checked_bag_price", "REAL"),
        ("flight_combinations", "carry_on_allowed", "BOOLEAN DEFAULT TRUE"),
        ("flight_combinations", "available_seats", "INTEGER"),
        ("flight_combinations", "last_ticketing_date", "DATE"),
        ("flight_combinations", "overnight_layover", "BOOLEAN DEFAULT FALSE"),
        ("flight_combinations", "connection_airports", "TEXT"),
        # Add risk_profile to search_profiles
        ("search_profiles", "risk_profile", "TEXT DEFAULT 'moderate'"),
    ]
    
    try:
        with get_db_connection() as conn:
            for table, column, col_type in migrations:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    logger.info(f"Added column {column} to {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"Column {column} already exists in {table}")
                    else:
                        logger.warning(f"Migration note for {table}.{column}: {e}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Migration failed: {e}")
        return False


def verify_db() -> dict[str, Any]:
    """
    Verify database integrity and return statistics.
    """
    stats = {
        "exists": False,
        "tables": [],
        "flight_combinations_count": 0,
        "search_profiles_count": 0,
        "price_alerts_count": 0,
        "route_baselines_count": 0,
        "booking_outcomes_count": 0,
        "earliest_scrape": None,
        "latest_scrape": None,
        "routes_tracked": 0,
        "healthy": False,
    }
    
    try:
        with get_db_connection() as conn:
            stats["exists"] = True
            
            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            stats["tables"] = [row["name"] for row in cursor.fetchall()]
            
            # Count records
            if "flight_combinations" in stats["tables"]:
                cursor = conn.execute("SELECT COUNT(*) as count FROM flight_combinations")
                stats["flight_combinations_count"] = cursor.fetchone()["count"]
                
                cursor = conn.execute(
                    "SELECT MIN(scrape_timestamp) as earliest, MAX(scrape_timestamp) as latest "
                    "FROM flight_combinations"
                )
                row = cursor.fetchone()
                stats["earliest_scrape"] = row["earliest"]
                stats["latest_scrape"] = row["latest"]
                
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT departure_airport || '-' || arrival_airport) as routes "
                    "FROM flight_combinations"
                )
                stats["routes_tracked"] = cursor.fetchone()["routes"]
            
            if "search_profiles" in stats["tables"]:
                cursor = conn.execute("SELECT COUNT(*) as count FROM search_profiles")
                stats["search_profiles_count"] = cursor.fetchone()["count"]
            
            if "price_alerts" in stats["tables"]:
                cursor = conn.execute("SELECT COUNT(*) as count FROM price_alerts")
                stats["price_alerts_count"] = cursor.fetchone()["count"]
            
            if "route_baselines" in stats["tables"]:
                cursor = conn.execute("SELECT COUNT(*) as count FROM route_baselines")
                stats["route_baselines_count"] = cursor.fetchone()["count"]
            
            if "booking_outcomes" in stats["tables"]:
                cursor = conn.execute("SELECT COUNT(*) as count FROM booking_outcomes")
                stats["booking_outcomes_count"] = cursor.fetchone()["count"]
            
            stats["healthy"] = True
            
    except sqlite3.Error as e:
        logger.error(f"Database verification failed: {e}")
        stats["error"] = str(e)
    
    return stats


# =============================================================================
# QUERY UTILITIES
# =============================================================================

def execute_query(
    query: str,
    params: Optional[tuple] = None,
    fetch: str = "all"
) -> Optional[list[dict] | dict]:
    """Execute a database query with error handling."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(query, params or ())
            
            if fetch == "all":
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            elif fetch == "one":
                row = cursor.fetchone()
                return dict(row) if row else None
            else:
                return None
                
    except sqlite3.Error as e:
        logger.error(f"Query execution failed: {e}\nQuery: {query}\nParams: {params}")
        raise


def execute_many(query: str, params_list: list[tuple]) -> int:
    """Execute a query multiple times with different parameters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount
    except sqlite3.Error as e:
        logger.error(f"Batch query execution failed: {e}")
        raise


# =============================================================================
# FLIGHT COMBINATIONS OPERATIONS (ENHANCED)
# =============================================================================

# New column list with enhanced fields
FLIGHT_COMBO_COLUMNS = [
    'scrape_timestamp', 'departure_airport', 'arrival_airport',
    'departure_date', 'return_date', 'trip_duration_days', 'offer_rank',
    'price', 'currency', 'base_fare', 'taxes_fees',
    'included_bags', 'checked_bag_price', 'carry_on_allowed',
    'available_seats', 'last_ticketing_date',
    'airline_outbound', 'airline_return',
    'stops_outbound', 'stops_return', 'duration_outbound_minutes',
    'duration_return_minutes', 'cabin_class',
    'overnight_layover', 'connection_airports',
    'lead_time_days', 'departure_day_of_week', 'return_day_of_week',
    'is_weekend_departure', 'is_weekend_return',
    'booking_token', 'deep_link_url', 'search_query_id', 'other_metadata'
]


def insert_flight_combination(combination: dict) -> Optional[int]:
    """
    Insert a single flight combination.
    
    Returns:
        The ID of the inserted row, or None if failed
    """
    placeholders = ', '.join(['?' for _ in FLIGHT_COMBO_COLUMNS])
    columns_str = ', '.join(FLIGHT_COMBO_COLUMNS)
    
    try:
        with get_db_connection() as conn:
            values = tuple(combination.get(col) for col in FLIGHT_COMBO_COLUMNS)
            cursor = conn.execute(
                f"INSERT OR REPLACE INTO flight_combinations ({columns_str}) VALUES ({placeholders})",
                values
            )
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to insert flight combination: {e}")
        return None


def insert_flight_combinations_batch(combinations: List[dict]) -> int:
    """
    Insert multiple flight combinations efficiently.
    
    Returns:
        Number of rows inserted
    """
    if not combinations:
        return 0
    
    placeholders = ', '.join(['?' for _ in FLIGHT_COMBO_COLUMNS])
    columns_str = ', '.join(FLIGHT_COMBO_COLUMNS)
    
    try:
        with get_db_connection() as conn:
            params_list = [
                tuple(combo.get(col) for col in FLIGHT_COMBO_COLUMNS)
                for combo in combinations
            ]
            cursor = conn.executemany(
                f"INSERT OR REPLACE INTO flight_combinations ({columns_str}) VALUES ({placeholders})",
                params_list
            )
            return cursor.rowcount
    except sqlite3.Error as e:
        logger.error(f"Failed to insert flight combinations batch: {e}")
        return 0


def get_combinations_for_route(
    origin: str,
    destination: str,
    target_month: Optional[str] = None,
    scrape_date: Optional[date] = None,
    offer_rank: Optional[int] = None
) -> List[dict]:
    """
    Get flight combinations for a route.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        target_month: Optional month filter (YYYY-MM)
        scrape_date: Optional specific scrape date filter
        offer_rank: Optional filter for specific rank (1=cheapest)
    """
    query = """
        SELECT * FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
    """
    params = [origin.upper(), destination.upper()]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    if scrape_date:
        query += " AND date(scrape_timestamp) = ?"
        params.append(str(scrape_date))
    
    if offer_rank is not None:
        query += " AND offer_rank = ?"
        params.append(offer_rank)
    
    query += " ORDER BY departure_date, return_date, offer_rank"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_combinations_for_calendar(
    origin: str,
    destination: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    cheapest_only: bool = True
) -> List[dict]:
    """
    Get flight combinations for calendar view.
    
    Returns combinations where trip duration is within flexibility range.
    """
    min_duration = trip_duration_days - flexibility_days
    max_duration = trip_duration_days + flexibility_days
    
    query = """
        SELECT * FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND strftime('%Y-%m', departure_date) = ?
        AND trip_duration_days BETWEEN ? AND ?
    """
    params = [origin.upper(), destination.upper(), target_month, min_duration, max_duration]
    
    if cheapest_only:
        query += " AND offer_rank = 1"
    
    query += " ORDER BY departure_date, trip_duration_days, price"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_latest_combinations(
    origin: str,
    destination: str,
    target_month: Optional[str] = None,
    offer_rank: int = 1
) -> List[dict]:
    """
    Get the most recent price for each departure-return combination.
    """
    query = """
        SELECT fc.* FROM flight_combinations fc
        INNER JOIN (
            SELECT departure_date, return_date, MAX(scrape_timestamp) as max_scrape
            FROM flight_combinations
            WHERE departure_airport = ? AND arrival_airport = ?
            AND offer_rank = ?
    """
    params = [origin.upper(), destination.upper(), offer_rank]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    query += """
            GROUP BY departure_date, return_date
        ) latest ON fc.departure_date = latest.departure_date 
            AND fc.return_date = latest.return_date
            AND fc.scrape_timestamp = latest.max_scrape
        WHERE fc.departure_airport = ? AND fc.arrival_airport = ?
        AND fc.offer_rank = ?
    """
    params.extend([origin.upper(), destination.upper(), offer_rank])
    
    if target_month:
        query += " AND strftime('%Y-%m', fc.departure_date) = ?"
        params.append(target_month)
    
    query += " ORDER BY fc.departure_date, fc.return_date"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_all_offers_for_combo(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    scrape_timestamp: Optional[datetime] = None
) -> List[dict]:
    """
    Get all ranked offers (top-K) for a specific departure-return combination.
    Useful for analyzing offer depth and price dispersion.
    """
    query = """
        SELECT * FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND departure_date = ? AND return_date = ?
    """
    params = [origin.upper(), destination.upper(), str(departure_date), str(return_date)]
    
    if scrape_timestamp:
        query += " AND scrape_timestamp = ?"
        params.append(str(scrape_timestamp))
    else:
        # Get latest scrape
        query += """
            AND scrape_timestamp = (
                SELECT MAX(scrape_timestamp) FROM flight_combinations
                WHERE departure_airport = ? AND arrival_airport = ?
                AND departure_date = ? AND return_date = ?
            )
        """
        params.extend([origin.upper(), destination.upper(), str(departure_date), str(return_date)])
    
    query += " ORDER BY offer_rank"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_cheapest_combination(
    origin: str,
    destination: str,
    target_month: Optional[str] = None,
    trip_duration_days: Optional[int] = None,
    flexibility_days: int = 2
) -> Optional[dict]:
    """Get the cheapest flight combination for a route."""
    query = """
        SELECT * FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND offer_rank = 1
    """
    params = [origin.upper(), destination.upper()]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    if trip_duration_days is not None:
        min_duration = trip_duration_days - flexibility_days
        max_duration = trip_duration_days + flexibility_days
        query += " AND trip_duration_days BETWEEN ? AND ?"
        params.extend([min_duration, max_duration])
    
    query += " ORDER BY price ASC LIMIT 1"
    
    return execute_query(query, tuple(params), fetch="one")


def get_price_history_for_combo(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    offer_rank: int = 1
) -> List[dict]:
    """Get price history for a specific departure-return combination."""
    query = """
        SELECT scrape_timestamp, price, lead_time_days, 
               available_seats, base_fare, taxes_fees
        FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND departure_date = ? AND return_date = ?
        AND offer_rank = ?
        ORDER BY scrape_timestamp ASC
    """
    return execute_query(
        query,
        (origin.upper(), destination.upper(), str(departure_date), str(return_date), offer_rank),
        fetch="all"
    ) or []


def get_intraday_prices(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    scrape_date: date,
    offer_rank: int = 1
) -> List[dict]:
    """Get all intraday price observations for a specific date."""
    query = """
        SELECT scrape_timestamp, price, available_seats
        FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND departure_date = ? AND return_date = ?
        AND date(scrape_timestamp) = ?
        AND offer_rank = ?
        ORDER BY scrape_timestamp ASC
    """
    return execute_query(
        query,
        (origin.upper(), destination.upper(), str(departure_date), 
         str(return_date), str(scrape_date), offer_rank),
        fetch="all"
    ) or []


def get_price_statistics_for_month(
    origin: str,
    destination: str,
    target_month: str
) -> dict:
    """Get price statistics for a specific month."""
    query = """
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            COUNT(*) as count,
            COUNT(DISTINCT departure_date) as unique_departure_dates,
            COUNT(DISTINCT date(scrape_timestamp)) as scrape_days,
            MIN(scrape_timestamp) as first_scrape,
            MAX(scrape_timestamp) as last_scrape
        FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND strftime('%Y-%m', departure_date) = ?
        AND offer_rank = 1
    """
    result = execute_query(
        query,
        (origin.upper(), destination.upper(), target_month),
        fetch="one"
    )
    return result or {
        "min_price": None, "max_price": None, "avg_price": None,
        "count": 0, "unique_departure_dates": 0, "scrape_days": 0,
        "first_scrape": None, "last_scrape": None
    }


def get_weekday_statistics(
    origin: str,
    destination: str,
    target_month: Optional[str] = None
) -> List[dict]:
    """Get average prices by departure day of week."""
    query = """
        SELECT 
            departure_day_of_week,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            COUNT(*) as count
        FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND offer_rank = 1
    """
    params = [origin.upper(), destination.upper()]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    query += " GROUP BY departure_day_of_week ORDER BY avg_price ASC"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_lead_time_statistics(
    origin: str,
    destination: str,
    target_month: Optional[str] = None
) -> List[dict]:
    """Get average prices by lead time buckets."""
    query = """
        SELECT 
            CASE
                WHEN lead_time_days BETWEEN 0 AND 7 THEN '0-7'
                WHEN lead_time_days BETWEEN 8 AND 14 THEN '8-14'
                WHEN lead_time_days BETWEEN 15 AND 30 THEN '15-30'
                WHEN lead_time_days BETWEEN 31 AND 60 THEN '31-60'
                ELSE '60+'
            END as lead_time_bucket,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            COUNT(*) as count
        FROM flight_combinations
        WHERE departure_airport = ? AND arrival_airport = ?
        AND offer_rank = 1
    """
    params = [origin.upper(), destination.upper()]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    query += " GROUP BY lead_time_bucket"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_offer_depth_stats(
    origin: str,
    destination: str,
    target_month: Optional[str] = None
) -> dict:
    """Get statistics about offer depth (how many offers per query)."""
    query = """
        SELECT 
            MAX(offer_rank) as max_offers_per_query,
            AVG(offer_count) as avg_offers_per_query,
            COUNT(*) as total_queries
        FROM (
            SELECT scrape_timestamp, departure_date, return_date, 
                   COUNT(*) as offer_count
            FROM flight_combinations
            WHERE departure_airport = ? AND arrival_airport = ?
    """
    params = [origin.upper(), destination.upper()]
    
    if target_month:
        query += " AND strftime('%Y-%m', departure_date) = ?"
        params.append(target_month)
    
    query += """
            GROUP BY scrape_timestamp, departure_date, return_date
        )
    """
    
    result = execute_query(query, tuple(params), fetch="one")
    return result or {"max_offers_per_query": 0, "avg_offers_per_query": 0, "total_queries": 0}


def get_price_dispersion(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date
) -> Optional[dict]:
    """
    Get price dispersion metrics for a specific date combination.
    Returns spread between offers as indicator of inventory pressure.
    """
    offers = get_all_offers_for_combo(origin, destination, departure_date, return_date)
    
    if not offers:
        return None
    
    prices = [o['price'] for o in offers if o['price']]
    
    if len(prices) < 2:
        return {
            "offer_count": len(prices),
            "min_price": prices[0] if prices else None,
            "max_price": prices[0] if prices else None,
            "spread": 0,
            "spread_pct": 0,
        }
    
    min_price = min(prices)
    max_price = max(prices)
    spread = max_price - min_price
    spread_pct = (spread / min_price * 100) if min_price > 0 else 0
    
    return {
        "offer_count": len(prices),
        "min_price": min_price,
        "max_price": max_price,
        "spread": spread,
        "spread_pct": spread_pct,
        "prices": prices,
    }


# =============================================================================
# SEARCH PROFILES OPERATIONS
# =============================================================================

def create_search_profile(profile: dict) -> Optional[int]:
    """Create a new search profile."""
    columns = [
        'profile_name', 'origin', 'destination', 'target_departure_month',
        'trip_duration_days', 'flexibility_days', 'max_stops',
        'preferred_airlines', 'max_price_threshold', 'notify_on_drop_percent',
        'is_active', 'risk_profile'
    ]
    
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join(columns)
    
    try:
        with get_db_connection() as conn:
            values = tuple(profile.get(col) for col in columns)
            cursor = conn.execute(
                f"INSERT OR REPLACE INTO search_profiles ({columns_str}) VALUES ({placeholders})",
                values
            )
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to create search profile: {e}")
        return None


def get_search_profiles(active_only: bool = True) -> List[dict]:
    """Get all search profiles."""
    query = "SELECT * FROM search_profiles"
    if active_only:
        query += " WHERE is_active = TRUE"
    query += " ORDER BY origin, destination"
    
    return execute_query(query, fetch="all") or []


def get_search_profile(profile_id: int) -> Optional[dict]:
    """Get a specific search profile by ID."""
    return execute_query(
        "SELECT * FROM search_profiles WHERE id = ?",
        (profile_id,),
        fetch="one"
    )


def get_profile_by_id(profile_id: int) -> Optional[dict]:
    """Alias for get_search_profile."""
    return get_search_profile(profile_id)


def get_search_profile_by_route(
    origin: str,
    destination: str,
    target_month: str,
    trip_duration: int
) -> Optional[dict]:
    """Get a search profile by route and parameters."""
    return execute_query(
        """SELECT * FROM search_profiles 
           WHERE origin = ? AND destination = ? 
           AND target_departure_month = ? AND trip_duration_days = ?""",
        (origin.upper(), destination.upper(), target_month, trip_duration),
        fetch="one"
    )


def update_profile_last_searched(profile_id: int, timestamp: Optional[datetime] = None) -> bool:
    """Update the last_searched timestamp for a search profile."""
    if timestamp is None:
        timestamp = datetime.now()
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE search_profiles SET last_searched = ? WHERE id = ?",
                (timestamp.isoformat(), profile_id)
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update last_searched: {e}")
        return False


def update_profile_risk_profile(profile_id: int, risk_profile: str) -> bool:
    """Update the risk profile for a search profile."""
    if risk_profile not in ('aggressive', 'moderate', 'conservative'):
        logger.error(f"Invalid risk profile: {risk_profile}")
        return False
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE search_profiles SET risk_profile = ? WHERE id = ?",
                (risk_profile, profile_id)
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update risk profile: {e}")
        return False


def delete_search_profile(profile_id: int) -> bool:
    """Delete a search profile and its associated alerts."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM price_alerts WHERE search_profile_id = ?", (profile_id,))
            conn.execute("DELETE FROM booking_outcomes WHERE search_profile_id = ?", (profile_id,))
            conn.execute("DELETE FROM search_profiles WHERE id = ?", (profile_id,))
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to delete search profile: {e}")
        return False


def toggle_profile_active(profile_id: int, is_active: bool) -> bool:
    """Toggle a search profile's active status."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE search_profiles SET is_active = ? WHERE id = ?",
                (is_active, profile_id)
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to toggle profile active status: {e}")
        return False


# =============================================================================
# PRICE ALERTS OPERATIONS
# =============================================================================

def create_price_alert(alert: dict) -> Optional[int]:
    """Create a new price alert."""
    columns = [
        'search_profile_id', 'alert_date', 'departure_date', 'return_date',
        'old_price', 'new_price', 'price_change_percent', 'alert_type'
    ]
    
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join(columns)
    
    try:
        with get_db_connection() as conn:
            values = tuple(alert.get(col) for col in columns)
            cursor = conn.execute(
                f"INSERT INTO price_alerts ({columns_str}) VALUES ({placeholders})",
                values
            )
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to create price alert: {e}")
        return None


def insert_price_alert(alert: dict) -> Optional[int]:
    """Alias for create_price_alert."""
    return create_price_alert(alert)


def get_price_alerts(
    profile_id: Optional[int] = None,
    unread_only: bool = False,
    limit: int = 50
) -> List[dict]:
    """Get price alerts."""
    query = "SELECT * FROM price_alerts WHERE 1=1"
    params = []
    
    if profile_id is not None:
        query += " AND search_profile_id = ?"
        params.append(profile_id)
    
    if unread_only:
        query += " AND is_read = FALSE"
    
    query += " ORDER BY alert_date DESC LIMIT ?"
    params.append(limit)
    
    return execute_query(query, tuple(params), fetch="all") or []


def mark_alert_read(alert_id: int) -> bool:
    """Mark a price alert as read."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE price_alerts SET is_read = TRUE WHERE id = ?",
                (alert_id,)
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to mark alert as read: {e}")
        return False


def mark_all_alerts_read(profile_id: Optional[int] = None) -> bool:
    """Mark all alerts as read."""
    try:
        with get_db_connection() as conn:
            if profile_id is not None:
                conn.execute(
                    "UPDATE price_alerts SET is_read = TRUE WHERE search_profile_id = ?",
                    (profile_id,)
                )
            else:
                conn.execute("UPDATE price_alerts SET is_read = TRUE")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to mark alerts as read: {e}")
        return False


def get_unread_alert_count(profile_id: Optional[int] = None) -> int:
    """Get count of unread alerts."""
    query = "SELECT COUNT(*) as count FROM price_alerts WHERE is_read = FALSE"
    params = []
    
    if profile_id is not None:
        query += " AND search_profile_id = ?"
        params.append(profile_id)
    
    result = execute_query(query, tuple(params), fetch="one")
    return result["count"] if result else 0


# =============================================================================
# ROUTE BASELINES OPERATIONS
# =============================================================================

def upsert_route_baseline(baseline: dict) -> Optional[int]:
    """Insert or update a route baseline."""
    columns = [
        'origin', 'destination', 'month', 'day_of_week', 'lead_time_bucket',
        'p10', 'p25', 'p50', 'p75', 'p90', 'mean', 'std_dev',
        'sample_count', 'last_updated'
    ]
    
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join(columns)
    
    try:
        with get_db_connection() as conn:
            values = tuple(baseline.get(col) for col in columns)
            cursor = conn.execute(
                f"INSERT OR REPLACE INTO route_baselines ({columns_str}) VALUES ({placeholders})",
                values
            )
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to upsert route baseline: {e}")
        return None


def get_route_baseline(
    origin: str,
    destination: str,
    month: Optional[int] = None,
    day_of_week: Optional[str] = None,
    lead_time_bucket: Optional[str] = None
) -> Optional[dict]:
    """Get a specific route baseline."""
    query = "SELECT * FROM route_baselines WHERE origin = ? AND destination = ?"
    params = [origin.upper(), destination.upper()]
    
    if month is not None:
        query += " AND month = ?"
        params.append(month)
    else:
        query += " AND month IS NULL"
    
    if day_of_week is not None:
        query += " AND day_of_week = ?"
        params.append(day_of_week)
    else:
        query += " AND day_of_week IS NULL"
    
    if lead_time_bucket is not None:
        query += " AND lead_time_bucket = ?"
        params.append(lead_time_bucket)
    else:
        query += " AND lead_time_bucket IS NULL"
    
    return execute_query(query, tuple(params), fetch="one")


def get_all_route_baselines(origin: str, destination: str) -> List[dict]:
    """Get all baselines for a route."""
    query = """
        SELECT * FROM route_baselines
        WHERE origin = ? AND destination = ?
        ORDER BY month, day_of_week, lead_time_bucket
    """
    return execute_query(query, (origin.upper(), destination.upper()), fetch="all") or []


def delete_route_baselines(origin: str, destination: str) -> bool:
    """Delete all baselines for a route."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM route_baselines WHERE origin = ? AND destination = ?",
                (origin.upper(), destination.upper())
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to delete route baselines: {e}")
        return False


# =============================================================================
# BOOKING OUTCOMES OPERATIONS
# =============================================================================

def insert_booking_outcome(outcome: dict) -> Optional[int]:
    """Insert a booking outcome for tracking."""
    columns = [
        'search_profile_id', 'departure_date', 'return_date',
        'decision_date', 'decision_made', 'price_at_decision',
        'fair_value_at_decision', 'regime_at_decision',
        'booked_date', 'booked_price', 'final_lowest_price',
        'booked_price_percentile', 'regret_vs_optimal', 'decision_was_correct'
    ]
    
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join(columns)
    
    try:
        with get_db_connection() as conn:
            values = tuple(outcome.get(col) for col in columns)
            cursor = conn.execute(
                f"INSERT INTO booking_outcomes ({columns_str}) VALUES ({placeholders})",
                values
            )
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Failed to insert booking outcome: {e}")
        return None


def update_booking_outcome(outcome_id: int, updates: dict) -> bool:
    """Update a booking outcome with actual results."""
    set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [outcome_id]
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE booking_outcomes SET {set_clause} WHERE id = ?",
                tuple(values)
            )
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update booking outcome: {e}")
        return False


def get_booking_outcomes(
    profile_id: Optional[int] = None,
    limit: int = 100
) -> List[dict]:
    """Get booking outcomes for analysis."""
    query = "SELECT * FROM booking_outcomes WHERE 1=1"
    params = []
    
    if profile_id is not None:
        query += " AND search_profile_id = ?"
        params.append(profile_id)
    
    query += " ORDER BY decision_date DESC LIMIT ?"
    params.append(limit)
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_regret_statistics(profile_id: Optional[int] = None) -> dict:
    """Get aggregate regret statistics for backtesting."""
    query = """
        SELECT 
            COUNT(*) as total_decisions,
            SUM(CASE WHEN decision_was_correct = 1 THEN 1 ELSE 0 END) as correct_decisions,
            AVG(regret_vs_optimal) as avg_regret,
            SUM(regret_vs_optimal) as total_regret,
            MIN(regret_vs_optimal) as best_outcome,
            MAX(regret_vs_optimal) as worst_outcome
        FROM booking_outcomes
        WHERE regret_vs_optimal IS NOT NULL
    """
    params = []
    
    if profile_id is not None:
        query += " AND search_profile_id = ?"
        params.append(profile_id)
    
    result = execute_query(query, tuple(params), fetch="one")
    return result or {
        "total_decisions": 0, "correct_decisions": 0,
        "avg_regret": 0, "total_regret": 0,
        "best_outcome": 0, "worst_outcome": 0
    }


# =============================================================================
# DATABASE MAINTENANCE
# =============================================================================

def cleanup_old_data(days_to_keep: int = 365) -> dict:
    """
    Remove data older than specified days.
    
    Returns:
        Dictionary with counts of deleted records
    """
    deleted = {"combinations": 0, "alerts": 0}
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM flight_combinations WHERE date(scrape_timestamp) < date('now', ?)",
                (f"-{days_to_keep} days",)
            )
            deleted["combinations"] = cursor.rowcount
            
            cursor = conn.execute(
                "DELETE FROM price_alerts WHERE date(alert_date) < date('now', ?)",
                (f"-{days_to_keep} days",)
            )
            deleted["alerts"] = cursor.rowcount
            
            logger.info(f"Cleaned up {deleted['combinations']} old combinations and {deleted['alerts']} alerts")
    except sqlite3.Error as e:
        logger.error(f"Cleanup failed: {e}")
    
    return deleted


def vacuum_database() -> bool:
    """Optimize database by running VACUUM."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("VACUUM")
        conn.close()
        logger.info("Database vacuumed successfully")
        return True
    except sqlite3.Error as e:
        logger.error(f"Vacuum failed: {e}")
        return False


def get_database_stats() -> dict:
    """Get comprehensive database statistics."""
    stats = {
        "total_combinations": 0,
        "total_profiles": 0,
        "active_profiles": 0,
        "total_alerts": 0,
        "unread_alerts": 0,
        "routes_with_data": 0,
        "months_with_data": [],
        "date_range": {"earliest": None, "latest": None},
        "route_baselines_count": 0,
        "booking_outcomes_count": 0,
    }
    
    try:
        with get_db_connection() as conn:
            # Combinations count
            cursor = conn.execute("SELECT COUNT(*) as count FROM flight_combinations")
            stats["total_combinations"] = cursor.fetchone()["count"]
            
            # Profiles count
            cursor = conn.execute("SELECT COUNT(*) as count FROM search_profiles")
            stats["total_profiles"] = cursor.fetchone()["count"]
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM search_profiles WHERE is_active = TRUE")
            stats["active_profiles"] = cursor.fetchone()["count"]
            
            # Alerts count
            cursor = conn.execute("SELECT COUNT(*) as count FROM price_alerts")
            stats["total_alerts"] = cursor.fetchone()["count"]
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM price_alerts WHERE is_read = FALSE")
            stats["unread_alerts"] = cursor.fetchone()["count"]
            
            # Routes with data
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT departure_airport || '-' || arrival_airport) as count FROM flight_combinations"
            )
            stats["routes_with_data"] = cursor.fetchone()["count"]
            
            # Months with data
            cursor = conn.execute(
                "SELECT DISTINCT strftime('%Y-%m', departure_date) as month FROM flight_combinations ORDER BY month"
            )
            stats["months_with_data"] = [row["month"] for row in cursor.fetchall() if row["month"]]
            
            # Date range
            cursor = conn.execute(
                "SELECT MIN(scrape_timestamp) as earliest, MAX(scrape_timestamp) as latest FROM flight_combinations"
            )
            row = cursor.fetchone()
            stats["date_range"]["earliest"] = row["earliest"]
            stats["date_range"]["latest"] = row["latest"]
            
            # Route baselines
            try:
                cursor = conn.execute("SELECT COUNT(*) as count FROM route_baselines")
                stats["route_baselines_count"] = cursor.fetchone()["count"]
            except sqlite3.OperationalError:
                pass  # Table may not exist yet
            
            # Booking outcomes
            try:
                cursor = conn.execute("SELECT COUNT(*) as count FROM booking_outcomes")
                stats["booking_outcomes_count"] = cursor.fetchone()["count"]
            except sqlite3.OperationalError:
                pass  # Table may not exist yet
            
    except sqlite3.Error as e:
        logger.error(f"Failed to get database stats: {e}")
        stats["error"] = str(e)
    
    return stats
