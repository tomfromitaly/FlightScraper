"""
Baseline Builder for Flight Price Tracker.

Builds and maintains route-month-weekday-leadtime baselines from historical data.
Used to establish "normal" price distributions for fair value calculations.
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from database import (
    get_combinations_for_route,
    upsert_route_baseline,
    get_all_route_baselines,
    delete_route_baselines,
    get_db_connection,
)
from models import FairValueDistribution
from utils import calculate_percentile, calculate_statistics, calculate_quantiles

# Configure logging
logger = logging.getLogger(__name__)

# Lead time buckets for baseline building
LEAD_TIME_BUCKETS = [
    ('0-7', 0, 7),
    ('8-14', 8, 14),
    ('15-30', 15, 30),
    ('31-60', 31, 60),
    ('60+', 61, 999),
]

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# =============================================================================
# BASELINE BUILDING
# =============================================================================

def build_route_baselines(
    origin: str,
    destination: str,
    min_sample_count: int = 10,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build comprehensive baselines for a route from historical data.
    
    Creates baselines at multiple granularities:
    - Overall route baseline
    - By month (1-12)
    - By day of week
    - By lead time bucket
    - Combinations of the above where sufficient data exists
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        min_sample_count: Minimum samples required to create a baseline
        rebuild: If True, delete existing baselines first
        
    Returns:
        Summary of baselines created
    """
    logger.info(f"Building baselines for {origin}-{destination}")
    
    if rebuild:
        delete_route_baselines(origin, destination)
        logger.info("Deleted existing baselines")
    
    # Get all historical combinations (cheapest offers only)
    combos = get_combinations_for_route(origin, destination, offer_rank=1)
    
    if not combos:
        logger.warning(f"No data for {origin}-{destination}")
        return {'baselines_created': 0, 'error': 'No historical data'}
    
    logger.info(f"Found {len(combos)} historical data points")
    
    # Extract prices with metadata
    price_data = []
    for combo in combos:
        dep_date = combo['departure_date']
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        
        price_data.append({
            'price': combo['price'],
            'month': dep_date.month,
            'day_of_week': combo.get('departure_day_of_week'),
            'lead_time_days': combo.get('lead_time_days', 30),
        })
    
    baselines_created = 0
    
    # 1. Overall route baseline (no filters)
    if len(price_data) >= min_sample_count:
        baseline = _create_baseline(
            origin, destination, 
            [p['price'] for p in price_data],
            month=None, day_of_week=None, lead_time_bucket=None
        )
        if baseline:
            baselines_created += 1
    
    # 2. By month
    for month in range(1, 13):
        month_prices = [p['price'] for p in price_data if p['month'] == month]
        if len(month_prices) >= min_sample_count:
            baseline = _create_baseline(
                origin, destination, month_prices,
                month=month, day_of_week=None, lead_time_bucket=None
            )
            if baseline:
                baselines_created += 1
    
    # 3. By day of week
    for weekday in WEEKDAYS:
        weekday_prices = [p['price'] for p in price_data if p['day_of_week'] == weekday]
        if len(weekday_prices) >= min_sample_count:
            baseline = _create_baseline(
                origin, destination, weekday_prices,
                month=None, day_of_week=weekday, lead_time_bucket=None
            )
            if baseline:
                baselines_created += 1
    
    # 4. By lead time bucket
    for bucket_name, min_days, max_days in LEAD_TIME_BUCKETS:
        bucket_prices = [
            p['price'] for p in price_data 
            if min_days <= p['lead_time_days'] <= max_days
        ]
        if len(bucket_prices) >= min_sample_count:
            baseline = _create_baseline(
                origin, destination, bucket_prices,
                month=None, day_of_week=None, lead_time_bucket=bucket_name
            )
            if baseline:
                baselines_created += 1
    
    # 5. Month × Lead Time (most useful combination)
    for month in range(1, 13):
        for bucket_name, min_days, max_days in LEAD_TIME_BUCKETS:
            combo_prices = [
                p['price'] for p in price_data 
                if p['month'] == month and min_days <= p['lead_time_days'] <= max_days
            ]
            if len(combo_prices) >= min_sample_count:
                baseline = _create_baseline(
                    origin, destination, combo_prices,
                    month=month, day_of_week=None, lead_time_bucket=bucket_name
                )
                if baseline:
                    baselines_created += 1
    
    logger.info(f"Created {baselines_created} baselines for {origin}-{destination}")
    
    return {
        'origin': origin,
        'destination': destination,
        'baselines_created': baselines_created,
        'total_data_points': len(price_data),
    }


def _create_baseline(
    origin: str,
    destination: str,
    prices: List[float],
    month: Optional[int],
    day_of_week: Optional[str],
    lead_time_bucket: Optional[str],
) -> Optional[int]:
    """
    Create a single baseline entry from price data.
    
    Returns the baseline ID if successful, None otherwise.
    """
    if not prices:
        return None
    
    stats = calculate_statistics(prices)
    
    baseline_data = {
        'origin': origin.upper(),
        'destination': destination.upper(),
        'month': month,
        'day_of_week': day_of_week,
        'lead_time_bucket': lead_time_bucket,
        'p10': stats['p10'],
        'p25': stats['p25'],
        'p50': stats['p50'],
        'p75': stats['p75'],
        'p90': stats['p90'],
        'mean': stats['mean'],
        'std_dev': stats['std_dev'],
        'sample_count': stats['count'],
        'last_updated': datetime.now().isoformat(),
    }
    
    return upsert_route_baseline(baseline_data)


def rebuild_all_baselines(
    min_sample_count: int = 10,
    routes: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    """
    Rebuild baselines for all routes (or specified routes).
    
    Args:
        min_sample_count: Minimum samples required
        routes: Optional list of (origin, dest) tuples. If None, discovers from data.
        
    Returns:
        Summary of all baselines built
    """
    # Discover routes if not specified
    if routes is None:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """SELECT DISTINCT departure_airport, arrival_airport 
                   FROM flight_combinations
                   WHERE offer_rank = 1"""
            )
            routes = [(row['departure_airport'], row['arrival_airport']) 
                      for row in cursor.fetchall()]
    
    logger.info(f"Rebuilding baselines for {len(routes)} routes")
    
    results = []
    total_baselines = 0
    
    for origin, dest in routes:
        result = build_route_baselines(
            origin, dest, 
            min_sample_count=min_sample_count,
            rebuild=True
        )
        results.append(result)
        total_baselines += result.get('baselines_created', 0)
    
    return {
        'routes_processed': len(routes),
        'total_baselines_created': total_baselines,
        'route_details': results,
    }


# =============================================================================
# BASELINE RETRIEVAL
# =============================================================================

def get_best_baseline(
    origin: str,
    destination: str,
    month: Optional[int] = None,
    day_of_week: Optional[str] = None,
    lead_time_days: Optional[int] = None,
) -> Optional[FairValueDistribution]:
    """
    Get the most specific applicable baseline for given parameters.
    
    Tries to find baseline in order of specificity:
    1. Exact match (month + day + lead_time)
    2. Month + lead_time
    3. Month only
    4. Lead time only
    5. Day of week only
    6. Overall route baseline
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        month: Month number (1-12)
        day_of_week: Day name (Monday-Sunday)
        lead_time_days: Days until departure
        
    Returns:
        FairValueDistribution or None if no baseline exists
    """
    all_baselines = get_all_route_baselines(origin, destination)
    
    if not all_baselines:
        return None
    
    # Determine lead time bucket
    lead_time_bucket = None
    if lead_time_days is not None:
        for bucket_name, min_days, max_days in LEAD_TIME_BUCKETS:
            if min_days <= lead_time_days <= max_days:
                lead_time_bucket = bucket_name
                break
    
    # Try to find matching baselines in order of specificity
    match_attempts = [
        # Most specific: month + lead_time
        lambda b: b['month'] == month and b['lead_time_bucket'] == lead_time_bucket,
        # Month only
        lambda b: b['month'] == month and b['lead_time_bucket'] is None and b['day_of_week'] is None,
        # Lead time only
        lambda b: b['lead_time_bucket'] == lead_time_bucket and b['month'] is None and b['day_of_week'] is None,
        # Day of week only
        lambda b: b['day_of_week'] == day_of_week and b['month'] is None and b['lead_time_bucket'] is None,
        # Overall baseline
        lambda b: b['month'] is None and b['day_of_week'] is None and b['lead_time_bucket'] is None,
    ]
    
    for matcher in match_attempts:
        matches = [b for b in all_baselines if matcher(b)]
        if matches:
            # Return first match (should only be one)
            baseline = matches[0]
            return _baseline_to_fair_value(baseline)
    
    return None


def _baseline_to_fair_value(baseline: dict) -> FairValueDistribution:
    """Convert a database baseline row to FairValueDistribution model."""
    return FairValueDistribution(
        origin=baseline['origin'],
        destination=baseline['destination'],
        p10=baseline['p10'],
        p25=baseline['p25'],
        p50=baseline['p50'],
        p75=baseline['p75'],
        p90=baseline['p90'],
        mean=baseline['mean'],
        std_dev=baseline['std_dev'],
        sample_count=baseline['sample_count'],
        month=baseline.get('month'),
        day_of_week=baseline.get('day_of_week'),
        lead_time_bucket=baseline.get('lead_time_bucket'),
        last_updated=datetime.fromisoformat(baseline['last_updated']) if baseline.get('last_updated') else None,
    )


def get_baseline_coverage(origin: str, destination: str) -> Dict[str, Any]:
    """
    Get information about baseline coverage for a route.
    
    Returns:
        Dictionary with coverage statistics
    """
    baselines = get_all_route_baselines(origin, destination)
    
    if not baselines:
        return {
            'has_baselines': False,
            'total_baselines': 0,
        }
    
    # Count by type
    overall = sum(1 for b in baselines 
                  if b['month'] is None and b['day_of_week'] is None and b['lead_time_bucket'] is None)
    by_month = sum(1 for b in baselines 
                   if b['month'] is not None and b['day_of_week'] is None and b['lead_time_bucket'] is None)
    by_weekday = sum(1 for b in baselines 
                     if b['day_of_week'] is not None and b['month'] is None and b['lead_time_bucket'] is None)
    by_lead_time = sum(1 for b in baselines 
                       if b['lead_time_bucket'] is not None and b['month'] is None and b['day_of_week'] is None)
    by_month_lead = sum(1 for b in baselines 
                        if b['month'] is not None and b['lead_time_bucket'] is not None)
    
    # Get overall baseline stats
    overall_baseline = next(
        (b for b in baselines 
         if b['month'] is None and b['day_of_week'] is None and b['lead_time_bucket'] is None),
        None
    )
    
    return {
        'has_baselines': True,
        'total_baselines': len(baselines),
        'overall_baseline': overall > 0,
        'by_month_count': by_month,
        'by_weekday_count': by_weekday,
        'by_lead_time_count': by_lead_time,
        'by_month_lead_count': by_month_lead,
        'overall_stats': {
            'p50': overall_baseline['p50'] if overall_baseline else None,
            'sample_count': overall_baseline['sample_count'] if overall_baseline else 0,
        } if overall_baseline else None,
        'months_covered': list(set(b['month'] for b in baselines if b['month'] is not None)),
        'lead_time_buckets_covered': list(set(b['lead_time_bucket'] for b in baselines if b['lead_time_bucket'])),
    }


# =============================================================================
# BASELINE ANALYSIS
# =============================================================================

def analyze_seasonal_patterns(
    origin: str,
    destination: str,
) -> Dict[str, Any]:
    """
    Analyze seasonal price patterns from baselines.
    
    Returns:
        Dictionary with seasonal insights
    """
    baselines = get_all_route_baselines(origin, destination)
    
    # Get monthly baselines
    monthly = [b for b in baselines 
               if b['month'] is not None and b['day_of_week'] is None and b['lead_time_bucket'] is None]
    
    if not monthly:
        return {'has_seasonal_data': False}
    
    # Sort by month
    monthly.sort(key=lambda x: x['month'])
    
    # Find cheapest and most expensive months
    by_median = sorted(monthly, key=lambda x: x['p50'])
    cheapest_month = by_median[0]['month']
    expensive_month = by_median[-1]['month']
    
    # Calculate seasonal variation
    median_prices = [b['p50'] for b in monthly]
    if max(median_prices) > 0:
        seasonal_variation = (max(median_prices) - min(median_prices)) / min(median_prices) * 100
    else:
        seasonal_variation = 0
    
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    return {
        'has_seasonal_data': True,
        'cheapest_month': month_names[cheapest_month],
        'cheapest_month_median': by_median[0]['p50'],
        'most_expensive_month': month_names[expensive_month],
        'expensive_month_median': by_median[-1]['p50'],
        'seasonal_variation_percent': round(seasonal_variation, 1),
        'monthly_medians': {
            month_names[b['month']]: b['p50'] for b in monthly
        },
    }


def analyze_lead_time_patterns(
    origin: str,
    destination: str,
) -> Dict[str, Any]:
    """
    Analyze booking window patterns from baselines.
    
    Returns:
        Dictionary with lead time insights
    """
    baselines = get_all_route_baselines(origin, destination)
    
    # Get lead time baselines
    lead_time = [b for b in baselines 
                 if b['lead_time_bucket'] is not None and b['month'] is None and b['day_of_week'] is None]
    
    if not lead_time:
        return {'has_lead_time_data': False}
    
    # Sort by bucket order
    bucket_order = {name: i for i, (name, _, _) in enumerate(LEAD_TIME_BUCKETS)}
    lead_time.sort(key=lambda x: bucket_order.get(x['lead_time_bucket'], 99))
    
    # Find optimal booking window
    by_median = sorted(lead_time, key=lambda x: x['p50'])
    optimal_bucket = by_median[0]['lead_time_bucket']
    optimal_median = by_median[0]['p50']
    
    # Check if last-minute is more expensive
    last_minute = next((b for b in lead_time if b['lead_time_bucket'] == '0-7'), None)
    advance = next((b for b in lead_time if b['lead_time_bucket'] == '31-60'), None)
    
    last_minute_premium = 0
    if last_minute and advance and advance['p50'] > 0:
        last_minute_premium = (last_minute['p50'] - advance['p50']) / advance['p50'] * 100
    
    return {
        'has_lead_time_data': True,
        'optimal_booking_window': optimal_bucket,
        'optimal_median_price': optimal_median,
        'last_minute_premium_percent': round(last_minute_premium, 1),
        'bucket_medians': {
            b['lead_time_bucket']: b['p50'] for b in lead_time
        },
        'recommendation': f"Book {optimal_bucket} days before departure for best prices",
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Build Route Baselines")
    parser.add_argument("--origin", "-o", help="Origin airport code")
    parser.add_argument("--dest", "-d", help="Destination airport code")
    parser.add_argument("--rebuild-all", action="store_true", help="Rebuild all baselines")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum samples required")
    parser.add_argument("--analyze", action="store_true", help="Analyze patterns after building")
    
    args = parser.parse_args()
    
    if args.rebuild_all:
        result = rebuild_all_baselines(min_sample_count=args.min_samples)
        print(f"Rebuilt {result['total_baselines_created']} baselines for {result['routes_processed']} routes")
    
    elif args.origin and args.dest:
        result = build_route_baselines(
            args.origin, args.dest,
            min_sample_count=args.min_samples,
            rebuild=True
        )
        print(f"Created {result['baselines_created']} baselines from {result['total_data_points']} data points")
        
        if args.analyze:
            seasonal = analyze_seasonal_patterns(args.origin, args.dest)
            lead_time = analyze_lead_time_patterns(args.origin, args.dest)
            
            print(f"\nSeasonal Patterns:")
            if seasonal['has_seasonal_data']:
                print(f"  Cheapest month: {seasonal['cheapest_month']} (${seasonal['cheapest_month_median']:.0f})")
                print(f"  Most expensive: {seasonal['most_expensive_month']} (${seasonal['expensive_month_median']:.0f})")
                print(f"  Variation: {seasonal['seasonal_variation_percent']}%")
            
            print(f"\nBooking Window Patterns:")
            if lead_time['has_lead_time_data']:
                print(f"  Optimal window: {lead_time['optimal_booking_window']} days")
                print(f"  Last-minute premium: {lead_time['last_minute_premium_percent']}%")
    
    else:
        parser.print_help()
