"""
Calendar Builder module for Flight Price Tracker.

Provides functions for building price matrices, calculating flexibility savings,
and finding optimal booking dates.
"""

import logging
from datetime import date, timedelta
from typing import Optional, List, Dict, Any, Tuple
from statistics import mean, median, stdev

import pandas as pd

from database import (
    get_combinations_for_calendar,
    get_latest_combinations,
    get_price_history_for_combo,
    get_lead_time_statistics,
)
from models import (
    PriceMatrix,
    FlexibilitySavings,
    OptimalBuyDate,
    CombinationRanking,
)
from utils import parse_date_range, get_dates_in_range

logger = logging.getLogger(__name__)


# =============================================================================
# PRICE MATRIX FUNCTIONS
# =============================================================================

def build_price_matrix(
    origin: str,
    dest: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    use_latest_only: bool = True
) -> PriceMatrix:
    """
    Create a 2D price matrix for a target month.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_month: Target month in 'YYYY-MM' format
        trip_duration_days: Base trip duration in days
        flexibility_days: Flexibility range (±days)
        use_latest_only: If True, only use the most recent scrape for each combo
        
    Returns:
        PriceMatrix object with:
        - Rows indexed by departure date
        - Columns indexed by duration offset (-2, -1, 0, +1, +2)
        - Values are prices
    """
    # Get data from database
    if use_latest_only:
        combinations = get_latest_combinations(origin, dest, target_month)
    else:
        combinations = get_combinations_for_calendar(
            origin, dest, target_month, trip_duration_days, flexibility_days
        )
    
    # Initialize price matrix
    start_date, end_date = parse_date_range(target_month)
    departure_dates = get_dates_in_range(start_date, end_date)
    offsets = list(range(-flexibility_days, flexibility_days + 1))
    
    matrix = PriceMatrix(
        origin=origin.upper(),
        destination=dest.upper(),
        target_month=target_month,
        base_trip_duration=trip_duration_days,
        flexibility_days=flexibility_days,
        departure_dates=departure_dates,
        return_date_offsets=offsets,
    )
    
    # Populate matrix from combinations
    all_prices = []
    
    for combo in combinations:
        dep_date = combo['departure_date']
        if isinstance(dep_date, str):
            dep_date = date.fromisoformat(dep_date)
        
        duration = combo['trip_duration_days']
        offset = duration - trip_duration_days
        
        if -flexibility_days <= offset <= flexibility_days:
            price = combo['price']
            matrix.set_price(dep_date, offset, price)
            all_prices.append(price)
    
    # Calculate statistics
    if all_prices:
        matrix.min_price = min(all_prices)
        matrix.max_price = max(all_prices)
        matrix.avg_price = mean(all_prices)
    
    return matrix


def price_matrix_to_dataframe(matrix: PriceMatrix) -> pd.DataFrame:
    """
    Convert a PriceMatrix to a pandas DataFrame for easier manipulation.
    
    Returns:
        DataFrame with departure dates as index, duration offsets as columns
    """
    data = {}
    
    for offset in matrix.return_date_offsets:
        offset_key = f"+{offset}" if offset > 0 else str(offset)
        column = []
        for dep_date in matrix.departure_dates:
            price = matrix.get_price(dep_date, offset)
            column.append(price)
        data[offset_key] = column
    
    df = pd.DataFrame(data, index=[str(d) for d in matrix.departure_dates])
    df.index.name = 'departure_date'
    
    return df


def find_cheapest_combinations(
    origin: str,
    dest: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    top_n: int = 10
) -> List[CombinationRanking]:
    """
    Find the N cheapest departure-return combinations.
    
    Returns:
        List of CombinationRanking objects sorted by price
    """
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations:
        return []
    
    # Filter by duration flexibility
    min_duration = trip_duration_days - flexibility_days
    max_duration = trip_duration_days + flexibility_days
    
    filtered = [
        c for c in combinations
        if min_duration <= c['trip_duration_days'] <= max_duration
    ]
    
    if not filtered:
        return []
    
    # Calculate average price for comparison
    all_prices = [c['price'] for c in filtered]
    avg_price = mean(all_prices)
    
    # Sort by price and take top N
    sorted_combos = sorted(filtered, key=lambda x: x['price'])[:top_n]
    
    rankings = []
    for rank, combo in enumerate(sorted_combos, 1):
        dep_date = combo['departure_date']
        if isinstance(dep_date, str):
            dep_date = date.fromisoformat(dep_date)
        
        ret_date = combo['return_date']
        if isinstance(ret_date, str):
            ret_date = date.fromisoformat(ret_date)
        
        vs_avg = ((combo['price'] - avg_price) / avg_price) * 100
        
        ranking = CombinationRanking(
            rank=rank,
            departure_date=dep_date,
            return_date=ret_date,
            trip_duration=combo['trip_duration_days'],
            price=combo['price'],
            airline_outbound=combo.get('airline_outbound'),
            airline_return=combo.get('airline_return'),
            stops_outbound=combo.get('stops_outbound', 0),
            stops_return=combo.get('stops_return', 0),
            vs_avg_percent=round(vs_avg, 1),
            booking_url=combo.get('deep_link_url'),
        )
        rankings.append(ranking)
    
    return rankings


# =============================================================================
# FLEXIBILITY SAVINGS FUNCTIONS
# =============================================================================

def calculate_flexibility_savings(
    origin: str,
    dest: str,
    base_departure: date,
    base_return: date,
    flexibility_days: int = 2
) -> FlexibilitySavings:
    """
    Calculate potential savings from flexible travel dates.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        base_departure: User's preferred departure date
        base_return: User's preferred return date
        flexibility_days: How many days flexibility to consider
        
    Returns:
        FlexibilitySavings object with alternatives and savings info
    """
    # Get combinations around the base dates
    target_month = base_departure.strftime('%Y-%m')
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations:
        logger.warning(f"No data found for {origin}-{dest} in {target_month}")
        # Return empty result with base price as 0
        return FlexibilitySavings(
            base_departure=base_departure,
            base_return=base_return,
            base_price=0.0,
        )
    
    # Find base combination price
    base_price = None
    for combo in combinations:
        dep = combo['departure_date']
        ret = combo['return_date']
        if isinstance(dep, str):
            dep = date.fromisoformat(dep)
        if isinstance(ret, str):
            ret = date.fromisoformat(ret)
        
        if dep == base_departure and ret == base_return:
            base_price = combo['price']
            break
    
    if base_price is None:
        # Use average price as baseline if exact combo not found
        all_prices = [c['price'] for c in combinations]
        base_price = mean(all_prices)
        logger.info(f"Base combo not found, using average price ${base_price:.2f}")
    
    # Create FlexibilitySavings object
    savings = FlexibilitySavings(
        base_departure=base_departure,
        base_return=base_return,
        base_price=base_price,
    )
    
    # Find alternatives within flexibility range
    for combo in combinations:
        dep = combo['departure_date']
        ret = combo['return_date']
        if isinstance(dep, str):
            dep = date.fromisoformat(dep)
        if isinstance(ret, str):
            ret = date.fromisoformat(ret)
        
        dep_diff = (dep - base_departure).days
        ret_diff = (ret - base_return).days
        
        # Skip if outside flexibility range
        if abs(dep_diff) > flexibility_days or abs(ret_diff) > flexibility_days:
            continue
        
        # Skip exact base combination
        if dep_diff == 0 and ret_diff == 0:
            continue
        
        savings.add_alternative(dep, ret, combo['price'])
    
    # Calculate average savings
    savings.calculate_avg_savings()
    
    return savings


def get_flexibility_recommendations(
    savings: FlexibilitySavings,
    min_savings: float = 20.0
) -> List[str]:
    """
    Generate human-readable recommendations from flexibility savings.
    
    Args:
        savings: FlexibilitySavings object
        min_savings: Minimum savings to include in recommendations
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if not savings.alternatives:
        recommendations.append("No alternative dates available for comparison.")
        return recommendations
    
    # Filter alternatives with significant savings
    good_alternatives = [
        a for a in savings.alternatives
        if a['savings'] >= min_savings
    ]
    
    if not good_alternatives:
        recommendations.append(
            f"Your selected dates are already competitive. "
            f"No alternatives save more than ${min_savings:.0f}."
        )
        return recommendations
    
    # Sort by savings
    good_alternatives.sort(key=lambda x: x['savings'], reverse=True)
    
    # Top 3 recommendations
    for alt in good_alternatives[:3]:
        recommendations.append(alt['recommendation'])
    
    # Summary
    if savings.best_alternative:
        recommendations.append(
            f"Best option: {savings.best_alternative['recommendation']}"
        )
    
    if savings.avg_savings_if_flexible > 0:
        recommendations.append(
            f"On average, being flexible saves ${savings.avg_savings_if_flexible:.2f}"
        )
    
    return recommendations


# =============================================================================
# OPTIMAL BUY DATE FUNCTIONS
# =============================================================================

def find_optimal_buy_date(
    origin: str,
    dest: str,
    target_departure: date,
    target_return: date
) -> OptimalBuyDate:
    """
    Analyze historical data to recommend when to buy tickets.
    
    Uses lead_time patterns to determine optimal booking window.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_departure: Target departure date
        target_return: Target return date
        
    Returns:
        OptimalBuyDate object with recommendation
    """
    # Get price history for this specific combination
    history = get_price_history_for_combo(
        origin, dest, target_departure, target_return
    )
    
    today = date.today()
    days_until_departure = (target_departure - today).days
    
    # Default values
    optimal_lead_days = 45  # Default recommendation
    confidence = "low"
    price_trend = "stable"
    expected_price = None
    historical_min = None
    historical_max = None
    historical_avg = None
    
    if history:
        # Analyze historical prices at different lead times
        prices = [h['price'] for h in history]
        historical_min = min(prices)
        historical_max = max(prices)
        historical_avg = mean(prices)
        
        # Find which lead time had lowest prices
        lead_time_prices = {}
        for h in history:
            lt = h['lead_time_days']
            # Bucket lead times
            if lt <= 14:
                bucket = 7
            elif lt <= 30:
                bucket = 21
            elif lt <= 60:
                bucket = 45
            elif lt <= 90:
                bucket = 75
            else:
                bucket = 120
            
            if bucket not in lead_time_prices:
                lead_time_prices[bucket] = []
            lead_time_prices[bucket].append(h['price'])
        
        # Find bucket with lowest average
        if lead_time_prices:
            bucket_avgs = {
                lt: mean(prices)
                for lt, prices in lead_time_prices.items()
            }
            optimal_lead_days = min(bucket_avgs, key=bucket_avgs.get)
            expected_price = bucket_avgs[optimal_lead_days]
        
        # Determine confidence based on data points
        data_points = len(history)
        if data_points >= 30:
            confidence = "high"
        elif data_points >= 10:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Determine price trend (is it going up or down as we get closer?)
        if len(history) >= 2:
            recent_prices = sorted(history, key=lambda x: x['scrape_date'])[-5:]
            if len(recent_prices) >= 2:
                first_price = recent_prices[0]['price']
                last_price = recent_prices[-1]['price']
                change_pct = ((last_price - first_price) / first_price) * 100
                
                if change_pct > 5:
                    price_trend = "increasing"
                elif change_pct < -5:
                    price_trend = "decreasing"
                else:
                    price_trend = "stable"
    else:
        # No history - use general lead time statistics
        lead_stats = get_lead_time_statistics(origin, dest)
        
        if lead_stats:
            # Find bucket with lowest average
            best_bucket = min(lead_stats, key=lambda x: x['avg_price'])
            # Parse bucket name to get lead days
            bucket_name = best_bucket['bucket_name']
            if '31-60' in bucket_name:
                optimal_lead_days = 45
            elif '15-30' in bucket_name:
                optimal_lead_days = 21
            elif '61-90' in bucket_name:
                optimal_lead_days = 75
            elif '91-120' in bucket_name:
                optimal_lead_days = 105
            else:
                optimal_lead_days = 45
            
            expected_price = best_bucket['avg_price']
    
    # Calculate recommended buy date
    if days_until_departure > optimal_lead_days:
        recommended_buy_date = target_departure - timedelta(days=optimal_lead_days)
    else:
        # It's already past optimal window, recommend buying soon
        recommended_buy_date = today + timedelta(days=1)
    
    return OptimalBuyDate(
        target_departure=target_departure,
        target_return=target_return,
        optimal_lead_days=optimal_lead_days,
        recommended_buy_date=recommended_buy_date,
        expected_price=expected_price,
        confidence=confidence,
        data_points=len(history) if history else 0,
        price_trend=price_trend,
        historical_min=historical_min,
        historical_max=historical_max,
        historical_avg=historical_avg,
    )


# =============================================================================
# CALENDAR HEATMAP DATA
# =============================================================================

def generate_calendar_heatmap_data(
    origin: str,
    dest: str,
    year_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2
) -> Dict[str, Any]:
    """
    Prepare data for a calendar heatmap visualization.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        year_month: Target month in 'YYYY-MM' format
        trip_duration_days: Base trip duration
        flexibility_days: Flexibility range
        
    Returns:
        Dictionary with calendar heatmap data
    """
    combinations = get_latest_combinations(origin, dest, year_month)
    
    start_date, end_date = parse_date_range(year_month)
    all_dates = get_dates_in_range(start_date, end_date)
    
    # Group by departure date and find min price for each
    date_prices = {}
    for combo in combinations:
        dep = combo['departure_date']
        if isinstance(dep, str):
            dep = date.fromisoformat(dep)
        
        duration = combo['trip_duration_days']
        if abs(duration - trip_duration_days) > flexibility_days:
            continue
        
        dep_str = str(dep)
        if dep_str not in date_prices:
            date_prices[dep_str] = []
        date_prices[dep_str].append(combo['price'])
    
    # Build output
    dates = []
    prices = []
    min_prices = []
    avg_prices = []
    
    for d in all_dates:
        d_str = str(d)
        dates.append(d_str)
        
        if d_str in date_prices:
            day_prices = date_prices[d_str]
            prices.append(min(day_prices))  # Show min for the heatmap
            min_prices.append(min(day_prices))
            avg_prices.append(mean(day_prices))
        else:
            prices.append(None)
            min_prices.append(None)
            avg_prices.append(None)
    
    # Calculate overall stats
    valid_prices = [p for p in prices if p is not None]
    
    return {
        'dates': dates,
        'prices': prices,
        'min_prices': min_prices,
        'avg_prices': avg_prices,
        'min_price': min(valid_prices) if valid_prices else None,
        'max_price': max(valid_prices) if valid_prices else None,
        'avg_price': mean(valid_prices) if valid_prices else None,
        'data_coverage': len(valid_prices) / len(dates) if dates else 0,
        'month': year_month,
        'origin': origin,
        'destination': dest,
    }


# =============================================================================
# DEAL DETECTION
# =============================================================================

def find_deals(
    origin: str,
    dest: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    deal_threshold_pct: float = 15.0
) -> List[Dict[str, Any]]:
    """
    Find combinations that are significantly below average (deals).
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_month: Target month
        trip_duration_days: Base trip duration
        flexibility_days: Flexibility range
        deal_threshold_pct: Percentage below average to qualify as deal
        
    Returns:
        List of deal dictionaries
    """
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations:
        return []
    
    # Filter by duration
    min_duration = trip_duration_days - flexibility_days
    max_duration = trip_duration_days + flexibility_days
    
    filtered = [
        c for c in combinations
        if min_duration <= c['trip_duration_days'] <= max_duration
    ]
    
    if not filtered:
        return []
    
    # Calculate average
    all_prices = [c['price'] for c in filtered]
    avg_price = mean(all_prices)
    
    # Find deals
    deals = []
    for combo in filtered:
        vs_avg = ((combo['price'] - avg_price) / avg_price) * 100
        
        if vs_avg < -deal_threshold_pct:
            dep = combo['departure_date']
            ret = combo['return_date']
            if isinstance(dep, str):
                dep = date.fromisoformat(dep)
            if isinstance(ret, str):
                ret = date.fromisoformat(ret)
            
            deals.append({
                'departure': str(dep),
                'return': str(ret),
                'price': combo['price'],
                'vs_avg': round(vs_avg, 1),
                'savings': round(avg_price - combo['price'], 2),
                'status': 'DEAL',
                'airline_outbound': combo.get('airline_outbound'),
                'airline_return': combo.get('airline_return'),
                'stops_outbound': combo.get('stops_outbound', 0),
                'stops_return': combo.get('stops_return', 0),
            })
    
    # Sort by savings
    deals.sort(key=lambda x: x['savings'], reverse=True)
    
    return deals


# =============================================================================
# DURATION COMPARISON
# =============================================================================

def compare_duration_options(
    origin: str,
    dest: str,
    target_month: str,
    durations: List[int] = [7, 14, 21, 28]
) -> Dict[str, Any]:
    """
    Compare different trip lengths to help users decide on duration.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_month: Target month
        durations: List of trip durations to compare
        
    Returns:
        Dictionary with duration comparison data
    """
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations:
        return {'durations': [], 'error': 'No data available'}
    
    results = []
    
    for duration in durations:
        # Filter combinations for this duration (±1 day)
        duration_combos = [
            c for c in combinations
            if abs(c['trip_duration_days'] - duration) <= 1
        ]
        
        if duration_combos:
            prices = [c['price'] for c in duration_combos]
            min_price = min(prices)
            avg_price = mean(prices)
            
            # Find cheapest combo
            cheapest = min(duration_combos, key=lambda x: x['price'])
            
            results.append({
                'duration': duration,
                'min_price': round(min_price, 2),
                'avg_price': round(avg_price, 2),
                'sample_count': len(duration_combos),
                'best_departure': cheapest['departure_date'],
                'best_return': cheapest['return_date'],
            })
        else:
            results.append({
                'duration': duration,
                'min_price': None,
                'avg_price': None,
                'sample_count': 0,
                'best_departure': None,
                'best_return': None,
            })
    
    # Sort by average price
    results.sort(key=lambda x: x['avg_price'] or float('inf'))
    
    # Generate insights
    insights = []
    valid_results = [r for r in results if r['avg_price'] is not None]
    
    if len(valid_results) >= 2:
        cheapest = valid_results[0]
        most_expensive = valid_results[-1]
        
        insights.append(
            f"Cheapest: {cheapest['duration']}-day trip at ${cheapest['avg_price']:.2f} avg"
        )
        
        if cheapest['duration'] != most_expensive['duration']:
            diff = most_expensive['avg_price'] - cheapest['avg_price']
            insights.append(
                f"A {most_expensive['duration']}-day trip costs ${diff:.2f} more on average"
            )
    
    return {
        'durations': results,
        'insights': insights,
        'origin': origin,
        'destination': dest,
        'month': target_month,
    }
