"""
Utility functions for Flight Price Tracker.

Contains helper functions for date parsing, formatting, validation, and common operations.
"""

import functools
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Generator, Optional, TypeVar
from calendar import monthrange

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# DATE UTILITIES
# =============================================================================

def parse_date_range(month_str: str) -> tuple[date, date]:
    """
    Convert a month string to start and end dates.
    
    Args:
        month_str: Month string in format 'YYYY-MM' (e.g., '2026-10')
        
    Returns:
        Tuple of (start_date, end_date) for the month
        
    Raises:
        ValueError: If month_str is not in valid format
        
    Example:
        >>> parse_date_range('2026-10')
        (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31))
    """
    try:
        year, month = map(int, month_str.strip().split('-'))
        # Validate month is in valid range
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be 1-12, got {month}")
        if year < 1900 or year > 2100:
            raise ValueError(f"Year must be between 1900-2100, got {year}")
        start_date = date(year, month, 1)
        _, last_day = monthrange(year, month)
        end_date = date(year, month, last_day)
        return start_date, end_date
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid month format '{month_str}'. Expected 'YYYY-MM' (e.g., '2026-10')"
        ) from e


def calculate_lead_time(departure_date: date, scrape_date: Optional[date] = None) -> int:
    """
    Calculate the number of days between scrape date and departure date.
    
    Args:
        departure_date: The flight departure date
        scrape_date: The date when price was recorded (defaults to today)
        
    Returns:
        Number of days until departure (can be negative if departure has passed)
        
    Example:
        >>> calculate_lead_time(date(2026, 10, 15), date(2026, 9, 1))
        44
    """
    if scrape_date is None:
        scrape_date = date.today()
    
    # Handle string dates
    if isinstance(departure_date, str):
        departure_date = datetime.strptime(departure_date, "%Y-%m-%d").date()
    if isinstance(scrape_date, str):
        scrape_date = datetime.strptime(scrape_date, "%Y-%m-%d").date()
    
    return (departure_date - scrape_date).days


def get_weekday_name(d: date | str) -> str:
    """
    Get the weekday name for a given date.
    
    Args:
        d: Date object or string in 'YYYY-MM-DD' format
        
    Returns:
        Weekday name (e.g., 'Monday', 'Tuesday')
        
    Example:
        >>> get_weekday_name(date(2026, 10, 15))
        'Thursday'
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return weekdays[d.weekday()]


def get_weekday_number(d: date | str) -> int:
    """
    Get the weekday number (0=Monday, 6=Sunday) for a given date.
    
    Args:
        d: Date object or string in 'YYYY-MM-DD' format
        
    Returns:
        Weekday number (0-6)
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    return d.weekday()


def parse_date(date_str: str) -> date:
    """
    Parse a date string in various formats.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD', 'MM/DD/YYYY', or 'DD-MM-YYYY' format
        
    Returns:
        Parsed date object
        
    Raises:
        ValueError: If date string cannot be parsed
    """
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse date '{date_str}'. Supported formats: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY")


def get_dates_in_range(start_date: date, end_date: date) -> list[date]:
    """
    Generate a list of all dates in a range (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of dates from start to end (inclusive)
    """
    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(days)]


def get_month_boundaries(d: date) -> tuple[date, date]:
    """
    Get the first and last day of the month for a given date.
    
    Args:
        d: Any date within the month
        
    Returns:
        Tuple of (first_day, last_day) of the month
    """
    first_day = date(d.year, d.month, 1)
    _, last = monthrange(d.year, d.month)
    last_day = date(d.year, d.month, last)
    return first_day, last_day


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a currency amount with proper symbol and thousand separators.
    
    Args:
        amount: The monetary amount
        currency: Currency code (default: USD)
        
    Returns:
        Formatted currency string (e.g., '$1,234.56')
        
    Example:
        >>> format_currency(1234.56)
        '$1,234.56'
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "MXN": "$",
        "CAD": "C$",
        "AUD": "A$",
    }
    symbol = symbols.get(currency.upper(), currency + " ")
    return f"{symbol}{amount:,.2f}"


def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to a human-readable string.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted duration string (e.g., '2h 30m')
        
    Example:
        >>> format_duration(150)
        '2h 30m'
    """
    if minutes < 0:
        return "N/A"
    
    hours = minutes // 60
    mins = minutes % 60
    
    if hours == 0:
        return f"{mins}m"
    elif mins == 0:
        return f"{hours}h"
    else:
        return f"{hours}h {mins}m"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., '15.0%')
    """
    return f"{value * 100:.{decimals}f}%"


def format_date_range(start: date, end: date) -> str:
    """
    Format a date range as a human-readable string.
    
    Args:
        start: Start date
        end: End date
        
    Returns:
        Formatted date range (e.g., 'Oct 1-15, 2026')
    """
    if start.year == end.year:
        if start.month == end.month:
            return f"{start.strftime('%b')} {start.day}-{end.day}, {start.year}"
        else:
            return f"{start.strftime('%b %d')} - {end.strftime('%b %d')}, {start.year}"
    else:
        return f"{start.strftime('%b %d, %Y')} - {end.strftime('%b %d, %Y')}"


# =============================================================================
# ITERATION UTILITIES
# =============================================================================

def chunked(iterable: list[T], size: int) -> Generator[list[T], None, None]:
    """
    Yield successive chunks of specified size from an iterable.
    
    Args:
        iterable: List or sequence to chunk
        size: Size of each chunk
        
    Yields:
        Lists of maximum `size` elements
        
    Example:
        >>> list(chunked([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# =============================================================================
# RETRY AND ERROR HANDLING
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3)
        def fetch_data():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter to control API request frequency.
    
    Ensures minimum delay between calls and limits requests per time window.
    """
    
    def __init__(self, min_delay: float = 0.3, max_per_minute: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            min_delay: Minimum seconds between requests
            max_per_minute: Maximum requests per minute
        """
        self.min_delay = min_delay
        self.max_per_minute = max_per_minute
        self.last_call_time = 0.0
        self.calls_this_minute: list[float] = []
    
    def wait(self) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Call this before each API request.
        """
        now = time.time()
        
        # Enforce minimum delay
        elapsed = now - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        # Clean up old calls (older than 1 minute)
        self.calls_this_minute = [t for t in self.calls_this_minute if now - t < 60]
        
        # Check if we've hit the per-minute limit
        if len(self.calls_this_minute) >= self.max_per_minute:
            # Wait until the oldest call is more than 1 minute old
            sleep_time = 60 - (now - self.calls_this_minute[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        # Record this call
        self.last_call_time = time.time()
        self.calls_this_minute.append(self.last_call_time)
    
    def get_remaining(self) -> int:
        """Get remaining requests allowed this minute."""
        now = time.time()
        self.calls_this_minute = [t for t in self.calls_this_minute if now - t < 60]
        return max(0, self.max_per_minute - len(self.calls_this_minute))


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_date_range(start_date: date, end_date: date) -> bool:
    """
    Validate that a date range is valid.
    
    Args:
        start_date: Start of range
        end_date: End of range
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
    
    today = date.today()
    if end_date < today:
        raise ValueError(f"End date {end_date} is in the past")
    
    return True


def validate_trip_duration(duration: int) -> bool:
    """
    Validate trip duration is within acceptable range.
    
    Args:
        duration: Trip duration in days
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if duration < 1:
        raise ValueError("Trip duration must be at least 1 day")
    if duration > 30:
        raise ValueError("Trip duration cannot exceed 30 days")
    return True


def sanitize_airport_code(code: str) -> str:
    """
    Sanitize and normalize an airport code.
    
    Args:
        code: Raw airport code input
        
    Returns:
        Uppercase, trimmed airport code
        
    Raises:
        ValueError: If code is invalid
    """
    if not code:
        raise ValueError("Airport code cannot be empty")
    
    code = code.strip().upper()
    
    if len(code) != 3:
        raise ValueError(f"Airport code must be exactly 3 characters, got '{code}'")
    
    if not code.isalpha():
        raise ValueError(f"Airport code must contain only letters, got '{code}'")
    
    return code


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def calculate_savings_percentage(price: float, reference_price: float) -> float:
    """
    Calculate percentage savings compared to a reference price.
    
    Args:
        price: The price to evaluate
        reference_price: The reference/baseline price
        
    Returns:
        Percentage savings (positive = cheaper, negative = more expensive)
    """
    if reference_price <= 0:
        return 0.0
    return ((reference_price - price) / reference_price) * 100


def calculate_percentile(values: list[float], percentile: float) -> float:
    """
    Calculate a percentile value from a list of numbers.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        The percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    
    lower = sorted_values[int(index)]
    upper = sorted_values[int(index) + 1]
    fraction = index - int(index)
    
    return lower + (upper - lower) * fraction


def get_quartiles(values: list[float]) -> dict[str, float]:
    """
    Calculate quartile statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with Q1, median (Q2), Q3, min, max
    """
    if not values:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
    
    return {
        "min": min(values),
        "q1": calculate_percentile(values, 25),
        "median": calculate_percentile(values, 50),
        "q3": calculate_percentile(values, 75),
        "max": max(values),
    }


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Optional custom log format
        log_file: Optional file path to log to
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def log_progress(current: int, total: int, prefix: str = "Progress", interval: int = 50) -> None:
    """
    Log progress for long-running operations.
    
    Only logs at specified intervals to avoid spam.
    
    Args:
        current: Current item number (1-indexed)
        total: Total number of items
        prefix: Prefix for log message
        interval: How often to log (every N items)
    """
    if current % interval == 0 or current == total:
        percentage = (current / total) * 100
        logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")


# =============================================================================
# GENERALIZED COST CALCULATOR
# =============================================================================

def calculate_generalized_cost(
    price: float,
    bags_needed: int = 0,
    checked_bag_price: Optional[float] = None,
    included_bags: int = 0,
    stops: int = 0,
    duration_minutes: int = 0,
    overnight_layover: bool = False,
    value_of_time: float = 25.0,  # $/hour
    stop_penalty: float = 30.0,   # $ per stop
    overnight_penalty: float = 100.0,  # $ for overnight layover
    default_bag_price: float = 35.0,  # Default checked bag price if unknown
) -> float:
    """
    Calculate all-in generalized cost including time and inconvenience penalties.
    
    This provides a "true cost" comparison between flights that accounts for:
    - Checked bag fees (if bags needed exceed included)
    - Time value (longer flights cost more in opportunity cost)
    - Stop inconvenience (each stop adds hassle)
    - Overnight layover penalty (significant inconvenience)
    
    Args:
        price: Base ticket price
        bags_needed: Number of checked bags the traveler needs
        checked_bag_price: Price per checked bag (if known)
        included_bags: Number of bags included in fare
        stops: Total number of stops (outbound + return)
        duration_minutes: Total travel time in minutes
        overnight_layover: Whether there's an overnight connection
        value_of_time: Dollar value per hour of travel time
        stop_penalty: Dollar penalty per stop
        overnight_penalty: Dollar penalty for overnight layover
        default_bag_price: Default bag price if not specified
        
    Returns:
        Generalized cost in dollars
        
    Example:
        >>> # Flight A: $300, no bags, direct, 3 hours
        >>> calculate_generalized_cost(300, bags_needed=1, included_bags=0, stops=0, duration_minutes=180)
        410.0  # 300 + 35 (bag) + 75 (time)
        
        >>> # Flight B: $250, 1 bag included, 1 stop, 5 hours
        >>> calculate_generalized_cost(250, bags_needed=1, included_bags=1, stops=1, duration_minutes=300)
        405.0  # 250 + 0 (bag included) + 30 (stop) + 125 (time)
    """
    # Start with base price
    total = price
    
    # Add bag costs (only for bags exceeding included)
    bags_to_pay_for = max(0, bags_needed - included_bags)
    bag_price = checked_bag_price if checked_bag_price is not None else default_bag_price
    total += bags_to_pay_for * bag_price
    
    # Add stop penalty
    total += stops * stop_penalty
    
    # Add overnight layover penalty
    if overnight_layover:
        total += overnight_penalty
    
    # Add time cost
    if duration_minutes > 0:
        hours = duration_minutes / 60
        total += hours * value_of_time
    
    return total


def calculate_generalized_cost_for_combo(
    combo: dict,
    bags_needed: int = 1,
    value_of_time: float = 25.0,
) -> float:
    """
    Calculate generalized cost for a flight combination dictionary.
    
    Convenience wrapper for database result rows.
    
    Args:
        combo: Flight combination dict (from database)
        bags_needed: Number of checked bags needed
        value_of_time: Dollar value per hour
        
    Returns:
        Generalized cost in dollars
    """
    price = combo.get('price', 0)
    included_bags = combo.get('included_bags', 0)
    checked_bag_price = combo.get('checked_bag_price')
    stops = (combo.get('stops_outbound', 0) or 0) + (combo.get('stops_return', 0) or 0)
    
    duration_out = combo.get('duration_outbound_minutes', 0) or 0
    duration_ret = combo.get('duration_return_minutes', 0) or 0
    total_duration = duration_out + duration_ret
    
    overnight = combo.get('overnight_layover', False)
    
    return calculate_generalized_cost(
        price=price,
        bags_needed=bags_needed,
        checked_bag_price=checked_bag_price,
        included_bags=included_bags,
        stops=stops,
        duration_minutes=total_duration,
        overnight_layover=overnight,
        value_of_time=value_of_time,
    )


def compare_flights_by_generalized_cost(
    combos: list[dict],
    bags_needed: int = 1,
    value_of_time: float = 25.0,
) -> list[dict]:
    """
    Compare flights by generalized cost instead of raw price.
    
    Returns combinations sorted by generalized cost with cost breakdown.
    
    Args:
        combos: List of flight combination dicts
        bags_needed: Number of checked bags needed
        value_of_time: Dollar value per hour
        
    Returns:
        List of dicts with generalized cost and breakdown, sorted by cost
    """
    results = []
    
    for combo in combos:
        price = combo.get('price', 0)
        gen_cost = calculate_generalized_cost_for_combo(
            combo, bags_needed, value_of_time
        )
        
        # Calculate breakdown
        stops = (combo.get('stops_outbound', 0) or 0) + (combo.get('stops_return', 0) or 0)
        duration_out = combo.get('duration_outbound_minutes', 0) or 0
        duration_ret = combo.get('duration_return_minutes', 0) or 0
        total_hours = (duration_out + duration_ret) / 60
        
        included_bags = combo.get('included_bags', 0)
        bags_to_pay = max(0, bags_needed - included_bags)
        bag_price = combo.get('checked_bag_price') or 35.0
        
        results.append({
            **combo,
            'generalized_cost': gen_cost,
            'raw_price': price,
            'bag_cost': bags_to_pay * bag_price,
            'time_cost': total_hours * value_of_time,
            'stop_penalty': stops * 30.0,
            'overnight_penalty': 100.0 if combo.get('overnight_layover') else 0.0,
            'total_stops': stops,
            'total_hours': round(total_hours, 1),
        })
    
    # Sort by generalized cost
    results.sort(key=lambda x: x['generalized_cost'])
    
    return results


# =============================================================================
# EXTENDED STATISTICAL UTILITIES
# =============================================================================

def calculate_quantiles(
    values: list[float],
    quantiles: list[float] = [10, 25, 50, 75, 90]
) -> dict[str, float]:
    """
    Calculate multiple quantile values from a list of numbers.
    
    Args:
        values: List of numeric values
        quantiles: List of percentiles to calculate (0-100)
        
    Returns:
        Dictionary mapping quantile names to values (e.g., {'p10': 100, 'p50': 150})
    """
    if not values:
        return {f"p{q}": 0.0 for q in quantiles}
    
    result = {}
    for q in quantiles:
        result[f"p{q}"] = calculate_percentile(values, q)
    
    return result


def calculate_statistics(values: list[float]) -> dict[str, float]:
    """
    Calculate comprehensive statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with min, max, mean, median, std_dev, and quantiles
    """
    if not values:
        return {
            'min': 0, 'max': 0, 'mean': 0, 'median': 0, 
            'std_dev': 0, 'count': 0,
            'p10': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0
        }
    
    n = len(values)
    mean = sum(values) / n
    
    # Calculate standard deviation
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0
    
    quantiles = calculate_quantiles(values)
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean,
        'median': quantiles['p50'],
        'std_dev': std_dev,
        'count': n,
        **quantiles
    }


def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """
    Calculate z-score for a value given mean and standard deviation.
    
    Args:
        value: The value to score
        mean: Population/sample mean
        std_dev: Population/sample standard deviation
        
    Returns:
        Z-score (number of standard deviations from mean)
    """
    if std_dev == 0:
        return 0.0
    return (value - mean) / std_dev


def detect_outliers_zscore(
    values: list[float],
    threshold: float = 2.0
) -> list[tuple[int, float, float]]:
    """
    Detect outliers using z-score method.
    
    Args:
        values: List of numeric values
        threshold: Z-score threshold for outlier detection
        
    Returns:
        List of (index, value, z_score) for outliers
    """
    if len(values) < 3:
        return []
    
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5
    
    if std_dev == 0:
        return []
    
    outliers = []
    for i, value in enumerate(values):
        z = (value - mean) / std_dev
        if abs(z) > threshold:
            outliers.append((i, value, z))
    
    return outliers
