"""
Trend analysis and recommendations for Flight Price Tracker.

Analyzes historical price data to provide insights on optimal booking strategies.
New architecture: Comprehensive calendar analysis with flexibility insights.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import lru_cache
import statistics

import pandas as pd
import numpy as np

from config import (
    LEAD_TIME_BUCKETS,
    DEAL_THRESHOLD_PERCENT,
    MIN_DATA_POINTS_FOR_ANALYSIS,
)
from database import (
    get_combinations_for_calendar,
    get_latest_combinations,
    get_price_statistics_for_month,
    get_weekday_statistics,
    get_lead_time_statistics,
    get_price_history_for_combo,
)
from models import (
    ComprehensiveAnalysisResult,
    WeekdayAnalysis,
    LeadTimeBucket,
    CombinationRanking,
)
from calendar_builder import (
    find_cheapest_combinations,
    calculate_flexibility_savings,
    find_optimal_buy_date,
    find_deals,
    generate_calendar_heatmap_data,
)
from utils import (
    get_weekday_name,
    calculate_lead_time,
    parse_date_range,
    calculate_savings_percentage,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE CALENDAR ANALYSIS
# =============================================================================

def analyze_comprehensive_calendar(
    origin: str,
    dest: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2
) -> ComprehensiveAnalysisResult:
    """
    Complete analysis combining all insights for a calendar search.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_month: Target month in 'YYYY-MM' format
        trip_duration_days: Base trip duration
        flexibility_days: Flexibility range (±days)
        
    Returns:
        ComprehensiveAnalysisResult with all insights
    """
    origin = origin.upper()
    dest = dest.upper()
    
    # Initialize result
    result = ComprehensiveAnalysisResult(
        route=f"{origin}-{dest}",
        origin=origin,
        destination=dest,
        target_month=target_month,
        trip_duration=trip_duration_days,
        flexibility_days=flexibility_days,
    )
    
    # Get combinations from database
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations:
        result.metadata = {
            'total_combinations': 0,
            'data_coverage_percent': 0,
            'confidence': 'none',
            'error': 'No data available for this route and month'
        }
        return result
    
    # Filter by duration flexibility
    min_duration = trip_duration_days - flexibility_days
    max_duration = trip_duration_days + flexibility_days
    
    filtered = [
        c for c in combinations
        if min_duration <= c['trip_duration_days'] <= max_duration
    ]
    
    if not filtered:
        result.metadata = {
            'total_combinations': len(combinations),
            'data_coverage_percent': 0,
            'confidence': 'none',
            'error': f'No combinations found for {trip_duration_days}±{flexibility_days} days'
        }
        return result
    
    # Calculate price statistics
    all_prices = [c['price'] for c in filtered]
    result.price_stats = {
        'min': min(all_prices),
        'max': max(all_prices),
        'avg': statistics.mean(all_prices),
        'median': statistics.median(all_prices),
        'std_dev': statistics.stdev(all_prices) if len(all_prices) > 1 else 0,
    }
    
    # Find top 10 cheapest combinations
    result.top_10_combos = find_cheapest_combinations(
        origin, dest, target_month, trip_duration_days, flexibility_days, top_n=10
    )
    
    # Set best combo
    if result.top_10_combos:
        best = result.top_10_combos[0]
        
        # Get optimal buy date for best combo
        optimal = find_optimal_buy_date(origin, dest, best.departure_date, best.return_date)
        
        result.best_combo = {
            'departure': str(best.departure_date),
            'return': str(best.return_date),
            'price': best.price,
            'buy_date': str(optimal.recommended_buy_date),
            'airline_outbound': best.airline_outbound,
            'airline_return': best.airline_return,
            'stops_outbound': best.stops_outbound,
            'stops_return': best.stops_return,
        }
    
    # Analyze flexibility savings
    if result.best_combo:
        # Use median-priced combo as baseline for flexibility comparison
        sorted_combos = sorted(filtered, key=lambda x: x['price'])
        mid_idx = len(sorted_combos) // 2
        base_combo = sorted_combos[mid_idx]
        
        base_dep = base_combo['departure_date']
        base_ret = base_combo['return_date']
        if isinstance(base_dep, str):
            base_dep = date.fromisoformat(base_dep)
        if isinstance(base_ret, str):
            base_ret = date.fromisoformat(base_ret)
        
        flex_savings = calculate_flexibility_savings(
            origin, dest, base_dep, base_ret, flexibility_days
        )
        
        # Build flexibility insights
        examples = []
        for alt in flex_savings.alternatives[:5]:  # Top 5 examples
            if alt['savings'] > 10:  # Only show significant savings
                examples.append(alt['recommendation'])
        
        result.flexibility_insights = {
            'avg_savings_if_flexible': flex_savings.avg_savings_if_flexible,
            'max_savings': flex_savings.max_savings,
            'examples': examples,
            'best_alternative': flex_savings.best_alternative,
        }
    
    # Weekday analysis
    weekday_stats = get_weekday_statistics(origin, dest, target_month)
    
    if weekday_stats:
        # Convert to WeekdayAnalysis objects
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        max_avg = max(w['avg_price'] for w in weekday_stats)
        
        result.weekday_details = []
        for w in weekday_stats:
            weekday_name = w['departure_day_of_week']
            if weekday_name and weekday_name in weekday_map:
                savings = calculate_savings_percentage(w['avg_price'], max_avg)
                result.weekday_details.append(WeekdayAnalysis(
                    weekday=weekday_name,
                    weekday_num=weekday_map[weekday_name],
                    avg_price=w['avg_price'],
                    min_price=w['min_price'],
                    max_price=w['max_price'],
                    sample_count=w['count'],
                    savings_vs_expensive=savings,
                ))
        
        # Sort by avg price
        result.weekday_details.sort(key=lambda x: x.avg_price)
        
        if result.weekday_details:
            cheapest = result.weekday_details[0]
            most_expensive = result.weekday_details[-1]
            
            # Calculate weekend premium
            weekday_prices = [w.avg_price for w in result.weekday_details if w.weekday_num < 5]
            weekend_prices = [w.avg_price for w in result.weekday_details if w.weekday_num >= 5]
            
            weekend_premium = 0
            if weekday_prices and weekend_prices:
                weekend_premium = statistics.mean(weekend_prices) - statistics.mean(weekday_prices)
            
            result.weekday_analysis = {
                'cheapest_departure_day': cheapest.weekday,
                'most_expensive_departure_day': most_expensive.weekday,
                'avg_weekend_premium': round(weekend_premium, 2),
                'savings_by_choosing_best_day': round(cheapest.savings_vs_expensive, 1),
            }
    
    # Lead time / booking window analysis
    lead_stats = get_lead_time_statistics(origin, dest, target_month)
    
    if lead_stats:
        result.lead_time_details = []
        bucket_avgs = {}
        
        for lt in lead_stats:
            bucket_name = lt['lead_time_bucket']
            
            # Parse bucket to get min/max days
            min_days, max_days = _parse_lead_time_bucket(bucket_name)
            
            result.lead_time_details.append(LeadTimeBucket(
                bucket_name=bucket_name,
                min_days=min_days,
                max_days=max_days,
                avg_price=lt['avg_price'],
                min_price=lt['min_price'],
                sample_count=lt['count'],
            ))
            
            bucket_avgs[bucket_name] = {
                'avg_price': lt['avg_price'],
                'count': lt['count']
            }
        
        # Sort by min_days
        result.lead_time_details.sort(key=lambda x: x.min_days)
        
        # Find optimal booking window
        if result.lead_time_details:
            optimal_bucket = min(result.lead_time_details, key=lambda x: x.avg_price)
            
            result.booking_windows = {
                'buckets': bucket_avgs,
                'optimal_window': optimal_bucket.bucket_name,
                'optimal_lead_days': (optimal_bucket.min_days + optimal_bucket.max_days) // 2,
                'optimal_avg_price': optimal_bucket.avg_price,
            }
    
    # Find current deals
    result.current_deals = find_deals(
        origin, dest, target_month, trip_duration_days, flexibility_days
    )
    
    # Metadata
    start_date, end_date = parse_date_range(target_month)
    total_possible = (end_date - start_date).days + 1
    total_possible *= (2 * flexibility_days + 1)  # For flexibility options
    
    result.metadata = {
        'total_combinations_searched': len(filtered),
        'total_possible_combinations': total_possible,
        'data_coverage_percent': round((len(filtered) / total_possible) * 100, 1) if total_possible > 0 else 0,
        'confidence': _calculate_confidence(len(filtered), total_possible),
    }
    
    return result


def _parse_lead_time_bucket(bucket_name: str) -> tuple:
    """Parse bucket name like '0-14 days' to (0, 14)."""
    try:
        if '+' in bucket_name:
            # Handle '120+ days'
            num = int(bucket_name.split('+')[0])
            return (num, 365)
        parts = bucket_name.replace(' days', '').split('-')
        return (int(parts[0]), int(parts[1]))
    except:
        return (0, 365)


def _calculate_confidence(actual: int, total: int) -> str:
    """Calculate confidence level based on data coverage."""
    if total == 0:
        return 'none'
    
    coverage = actual / total
    
    if coverage >= 0.8:
        return 'high'
    elif coverage >= 0.5:
        return 'medium'
    elif coverage >= 0.2:
        return 'low'
    else:
        return 'very_low'


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_price_patterns(
    origin: str,
    dest: str,
    target_month: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identify recurring patterns in price data.
    
    Detects:
    - Weekday price spikes
    - Lead time trends
    - Price volatility periods
    
    Returns:
        Dictionary with detected patterns
    """
    combinations = get_latest_combinations(origin, dest, target_month)
    
    if not combinations or len(combinations) < 10:
        return {'patterns': [], 'message': 'Insufficient data for pattern detection'}
    
    patterns = []
    
    # 1. Weekday pattern detection
    weekday_stats = get_weekday_statistics(origin, dest, target_month)
    if weekday_stats:
        prices_by_day = {w['departure_day_of_week']: w['avg_price'] for w in weekday_stats}
        
        if prices_by_day:
            avg_price = statistics.mean(prices_by_day.values())
            
            # Find days with significant deviation
            for day, price in prices_by_day.items():
                deviation_pct = ((price - avg_price) / avg_price) * 100
                
                if deviation_pct > 10:
                    patterns.append({
                        'type': 'weekday_spike',
                        'description': f'{day} departures are {deviation_pct:.0f}% more expensive than average',
                        'impact': 'high' if deviation_pct > 20 else 'medium',
                    })
                elif deviation_pct < -10:
                    patterns.append({
                        'type': 'weekday_deal',
                        'description': f'{day} departures are {abs(deviation_pct):.0f}% cheaper than average',
                        'impact': 'high' if deviation_pct < -20 else 'medium',
                    })
    
    # 2. Lead time trend
    lead_stats = get_lead_time_statistics(origin, dest, target_month)
    if lead_stats and len(lead_stats) >= 3:
        # Check if prices consistently increase as departure approaches
        sorted_buckets = sorted(lead_stats, key=lambda x: x.get('lead_time_bucket', ''))
        
        prices = [b['avg_price'] for b in sorted_buckets]
        
        # Simple trend detection
        if len(prices) >= 3:
            early_avg = statistics.mean(prices[:len(prices)//2])
            late_avg = statistics.mean(prices[len(prices)//2:])
            
            change_pct = ((late_avg - early_avg) / early_avg) * 100
            
            if change_pct > 15:
                patterns.append({
                    'type': 'last_minute_spike',
                    'description': f'Prices increase ~{change_pct:.0f}% in the final weeks before departure',
                    'impact': 'high',
                    'recommendation': 'Book early to avoid last-minute price increases',
                })
            elif change_pct < -15:
                patterns.append({
                    'type': 'last_minute_deals',
                    'description': f'Prices drop ~{abs(change_pct):.0f}% closer to departure',
                    'impact': 'medium',
                    'recommendation': 'Consider waiting for last-minute deals if flexible',
                })
    
    # 3. Price volatility
    all_prices = [c['price'] for c in combinations]
    if len(all_prices) >= 5:
        cv = statistics.stdev(all_prices) / statistics.mean(all_prices)
        
        if cv > 0.25:
            patterns.append({
                'type': 'high_volatility',
                'description': f'Prices vary significantly (CV={cv:.0%})',
                'impact': 'high',
                'recommendation': 'Set price alerts - good deals appear frequently',
            })
        elif cv < 0.1:
            patterns.append({
                'type': 'stable_prices',
                'description': 'Prices are relatively stable',
                'impact': 'low',
                'recommendation': 'Book when convenient - prices rarely change much',
            })
    
    return {
        'patterns': patterns,
        'patterns_detected': len(patterns),
        'origin': origin,
        'destination': dest,
        'month': target_month,
    }


def compare_duration_options(
    origin: str,
    dest: str,
    durations: List[int],
    target_month: str
) -> Dict[str, Any]:
    """
    Compare different trip lengths to help users decide on duration.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        durations: List of trip durations to compare
        target_month: Target month
        
    Returns:
        Dictionary with duration comparison data
    """
    from calendar_builder import compare_duration_options as _compare
    return _compare(origin, dest, target_month, durations)


def predict_future_price(
    origin: str,
    dest: str,
    departure_date: date,
    return_date: date,
    prediction_date: date
) -> Dict[str, Any]:
    """
    Predict price based on historical lead-time trends.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        departure_date: Target departure date
        return_date: Target return date
        prediction_date: Date to predict price for
        
    Returns:
        Dictionary with prediction and confidence
    """
    # Get price history for this combo
    history = get_price_history_for_combo(origin, dest, departure_date, return_date)
    
    if not history or len(history) < 3:
        # Fall back to general lead time trends
        lead_stats = get_lead_time_statistics(origin, dest)
        
        if not lead_stats:
            return {
                'prediction_available': False,
                'message': 'Insufficient historical data for prediction',
            }
        
        # Use general trend
        target_lead_time = (departure_date - prediction_date).days
        
        # Find appropriate bucket
        for lt in lead_stats:
            bucket = lt['lead_time_bucket']
            min_d, max_d = _parse_lead_time_bucket(bucket)
            if min_d <= target_lead_time <= max_d:
                return {
                    'prediction_available': True,
                    'predicted_price': lt['avg_price'],
                    'confidence': 'low',
                    'method': 'lead_time_bucket_average',
                    'message': f'Based on average prices for {bucket} lead time',
                }
        
        return {
            'prediction_available': False,
            'message': 'No matching lead time bucket found',
        }
    
    # Analyze price history
    prices = [h['price'] for h in history]
    lead_times = [h['lead_time_days'] for h in history]
    
    # Simple linear regression
    try:
        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(lead_times, prices)
        
        target_lead_time = (departure_date - prediction_date).days
        predicted_price = slope * target_lead_time + intercept
        
        r_squared = r_value ** 2
        confidence = 'high' if r_squared > 0.5 else 'medium' if r_squared > 0.25 else 'low'
        
        return {
            'prediction_available': True,
            'predicted_price': round(max(predicted_price, min(prices) * 0.8), 2),  # Floor at 80% of min
            'confidence': confidence,
            'r_squared': round(r_squared, 3),
            'price_trend': 'increasing' if slope < 0 else 'decreasing',  # Negative slope = price increases as lead time decreases
            'current_min': min(prices),
            'current_max': max(prices),
            'current_avg': statistics.mean(prices),
            'method': 'linear_regression',
            'data_points': len(history),
        }
        
    except ImportError:
        # Scipy not available - use simple average
        return {
            'prediction_available': True,
            'predicted_price': statistics.mean(prices),
            'confidence': 'low',
            'method': 'historical_average',
            'message': 'Install scipy for more accurate predictions',
        }


# =============================================================================
# RECOMMENDATION GENERATION
# =============================================================================

def generate_recommendations(analysis: ComprehensiveAnalysisResult) -> List[str]:
    """
    Generate actionable recommendations from analysis results.
    
    Args:
        analysis: ComprehensiveAnalysisResult object
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Best combo recommendation
    if analysis.best_combo:
        combo = analysis.best_combo
        recommendations.append(
            f"Best deal: Fly {combo['departure']} to {combo['return']} "
            f"at ${combo['price']:.2f}"
        )
        
        if combo.get('buy_date'):
            recommendations.append(
                f"Optimal booking date: {combo['buy_date']}"
            )
    
    # Flexibility recommendation
    if analysis.flexibility_insights:
        flex = analysis.flexibility_insights
        if flex.get('max_savings', 0) > 50:
            recommendations.append(
                f"Flexible dates could save up to ${flex['max_savings']:.2f}"
            )
        if flex.get('examples'):
            recommendations.append(flex['examples'][0])
    
    # Weekday recommendation
    if analysis.weekday_analysis:
        wa = analysis.weekday_analysis
        if wa.get('savings_by_choosing_best_day', 0) > 5:
            recommendations.append(
                f"Fly on {wa['cheapest_departure_day']} to save "
                f"~{wa['savings_by_choosing_best_day']:.0f}% vs {wa['most_expensive_departure_day']}"
            )
    
    # Booking window recommendation
    if analysis.booking_windows:
        bw = analysis.booking_windows
        if bw.get('optimal_window'):
            recommendations.append(
                f"Book {bw['optimal_window']} before departure for best prices"
            )
    
    # Deal alert
    if analysis.current_deals:
        best_deal = analysis.current_deals[0]
        recommendations.append(
            f"Deal alert: {best_deal['departure']} departure is "
            f"{abs(best_deal['vs_avg']):.0f}% below average!"
        )
    
    return recommendations


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def analyze_route(
    origin: str,
    destination: str,
    target_start_date: Optional[date] = None,
    target_end_date: Optional[date] = None,
    min_data_points: int = MIN_DATA_POINTS_FOR_ANALYSIS,
) -> ComprehensiveAnalysisResult:
    """
    Legacy function - redirects to comprehensive analysis.
    """
    if target_start_date:
        target_month = target_start_date.strftime('%Y-%m')
    else:
        target_month = date.today().strftime('%Y-%m')
    
    return analyze_comprehensive_calendar(
        origin, destination, target_month,
        trip_duration_days=7,
        flexibility_days=2
    )


def get_weekday_summary(origin: str, destination: str) -> dict:
    """Get a simple weekday price summary for a route."""
    stats = get_weekday_statistics(origin, destination)
    
    if not stats:
        return {}
    
    return {
        w['departure_day_of_week']: {
            'avg_price': w['avg_price'],
            'min_price': w['min_price'],
            'sample_count': w['count'],
        }
        for w in stats if w['departure_day_of_week']
    }


def get_optimal_booking_window(
    origin: str,
    destination: str,
    target_month: Optional[str] = None,
) -> dict:
    """Get the optimal booking window for a route."""
    lead_stats = get_lead_time_statistics(origin, destination, target_month)
    
    if not lead_stats:
        return {"message": "No data available"}
    
    # Find optimal bucket
    optimal = min(lead_stats, key=lambda x: x['avg_price'])
    worst = max(lead_stats, key=lambda x: x['avg_price'])
    
    savings = calculate_savings_percentage(optimal['avg_price'], worst['avg_price'])
    
    return {
        "optimal_window": optimal['lead_time_bucket'],
        "optimal_avg_price": optimal['avg_price'],
        "worst_window": worst['lead_time_bucket'],
        "worst_avg_price": worst['avg_price'],
        "potential_savings_pct": savings,
        "recommendation": f"Book {optimal['lead_time_bucket']} before departure for best prices",
    }


def get_current_deals(
    origin: str,
    destination: str,
    threshold_pct: float = DEAL_THRESHOLD_PERCENT,
) -> list[dict]:
    """Get current deals for a route."""
    target_month = date.today().strftime('%Y-%m')
    return find_deals(origin, destination, target_month, 7, 2, threshold_pct)


def get_price_trend_direction(origin: str, destination: str) -> dict:
    """Determine if prices are trending up or down."""
    patterns = detect_price_patterns(origin, destination)
    
    for p in patterns.get('patterns', []):
        if p['type'] == 'last_minute_spike':
            return {
                "trend": "increasing",
                "message": "Prices tend to increase as departure approaches - book early!",
            }
        elif p['type'] == 'last_minute_deals':
            return {
                "trend": "decreasing", 
                "message": "Prices tend to decrease closer to departure - last-minute deals may be available",
            }
    
    return {
        "trend": "stable",
        "message": "Prices are relatively stable regardless of booking time",
    }
