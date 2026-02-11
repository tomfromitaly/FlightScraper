"""
Route Archetype Classifier for Flight Price Tracker.

Classifies routes as business-heavy, leisure-heavy, or mixed
based on pricing patterns. Different archetypes warrant different
decision strategies.
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from models import RouteArchetype
from database import (
    get_weekday_statistics,
    get_lead_time_statistics,
    get_combinations_for_route,
)
from baseline_builder import (
    get_all_route_baselines,
    analyze_seasonal_patterns,
    analyze_lead_time_patterns,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ROUTE CLASSIFICATION
# =============================================================================

def classify_route(
    origin: str,
    destination: str,
    min_data_points: int = 50,
) -> RouteArchetype:
    """
    Classify a route based on weekday patterns and lead-time behavior.
    
    Business-heavy routes typically show:
    - Mon/Fri peaks (business travelers)
    - Last-minute premium (urgent business travel)
    - Lower weekend prices
    
    Leisure-heavy routes typically show:
    - Weekend departure preference
    - Advance booking discounts
    - Seasonal variations (holidays, summer)
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        min_data_points: Minimum data points required for classification
        
    Returns:
        RouteArchetype enum value
    """
    # Get weekday statistics
    weekday_stats = get_weekday_statistics(origin, destination)
    lead_time_stats = get_lead_time_statistics(origin, destination)
    
    if not weekday_stats or not lead_time_stats:
        return RouteArchetype.MIXED  # Default if insufficient data
    
    # Calculate business indicators
    business_score = 0
    leisure_score = 0
    
    # 1. Weekday pattern analysis
    weekday_pattern = _analyze_weekday_pattern(weekday_stats)
    business_score += weekday_pattern['business_signal']
    leisure_score += weekday_pattern['leisure_signal']
    
    # 2. Lead time pattern analysis
    lead_time_pattern = _analyze_lead_time_pattern(lead_time_stats)
    business_score += lead_time_pattern['business_signal']
    leisure_score += lead_time_pattern['leisure_signal']
    
    # Classify based on score difference
    score_diff = business_score - leisure_score
    
    if score_diff >= 2:
        return RouteArchetype.BUSINESS_HEAVY
    elif score_diff <= -2:
        return RouteArchetype.LEISURE_HEAVY
    else:
        return RouteArchetype.MIXED


def _analyze_weekday_pattern(weekday_stats: List[dict]) -> Dict[str, float]:
    """
    Analyze weekday pricing pattern for business/leisure signals.
    
    Returns signal strengths (0-3 scale).
    """
    if not weekday_stats:
        return {'business_signal': 0, 'leisure_signal': 0}
    
    # Create lookup by weekday
    by_weekday = {w['departure_day_of_week']: w for w in weekday_stats}
    
    business_signal = 0
    leisure_signal = 0
    
    # Get prices for key days
    mon = by_weekday.get('Monday', {}).get('avg_price', 0)
    tue = by_weekday.get('Tuesday', {}).get('avg_price', 0)
    wed = by_weekday.get('Wednesday', {}).get('avg_price', 0)
    thu = by_weekday.get('Thursday', {}).get('avg_price', 0)
    fri = by_weekday.get('Friday', {}).get('avg_price', 0)
    sat = by_weekday.get('Saturday', {}).get('avg_price', 0)
    sun = by_weekday.get('Sunday', {}).get('avg_price', 0)
    
    if not any([mon, tue, wed, thu, fri, sat, sun]):
        return {'business_signal': 0, 'leisure_signal': 0}
    
    weekday_avg = (mon + tue + wed + thu + fri) / 5 if (mon + tue + wed + thu + fri) else 0
    weekend_avg = (sat + sun) / 2 if (sat + sun) else 0
    overall_avg = sum([mon, tue, wed, thu, fri, sat, sun]) / 7
    
    if overall_avg == 0:
        return {'business_signal': 0, 'leisure_signal': 0}
    
    # Business signal: Mon/Fri higher than mid-week
    midweek_avg = (tue + wed + thu) / 3 if (tue + wed + thu) else weekday_avg
    mon_fri_avg = (mon + fri) / 2 if (mon + fri) else weekday_avg
    
    if midweek_avg > 0:
        mon_fri_premium = (mon_fri_avg - midweek_avg) / midweek_avg
        if mon_fri_premium > 0.10:
            business_signal += 2
        elif mon_fri_premium > 0.05:
            business_signal += 1
    
    # Leisure signal: Weekend prices higher (demand for leisure travel)
    # OR weekend prices notably lower (business routes discount weekends)
    if weekday_avg > 0:
        weekend_diff = (weekend_avg - weekday_avg) / weekday_avg
        
        # If weekends significantly cheaper, it's business route
        if weekend_diff < -0.10:
            business_signal += 2
        elif weekend_diff < -0.05:
            business_signal += 1
        
        # If weekends more expensive, it's leisure route
        elif weekend_diff > 0.10:
            leisure_signal += 2
        elif weekend_diff > 0.05:
            leisure_signal += 1
    
    # Sunday evening premium = business (returning from weekend)
    if sun and sat and sat > 0:
        sun_premium = (sun - sat) / sat
        if sun_premium > 0.15:
            business_signal += 1
    
    return {
        'business_signal': business_signal,
        'leisure_signal': leisure_signal,
    }


def _analyze_lead_time_pattern(lead_time_stats: List[dict]) -> Dict[str, float]:
    """
    Analyze lead time pricing pattern for business/leisure signals.
    
    Business routes: steep last-minute premium
    Leisure routes: advance booking discounts, flatter curve
    """
    if not lead_time_stats:
        return {'business_signal': 0, 'leisure_signal': 0}
    
    # Create lookup by bucket
    by_bucket = {l['lead_time_bucket']: l for l in lead_time_stats}
    
    business_signal = 0
    leisure_signal = 0
    
    # Get average prices by bucket
    last_min = by_bucket.get('0-7', {}).get('avg_price', 0) or \
               by_bucket.get('0-7 days', {}).get('avg_price', 0) or \
               by_bucket.get('0-14 days', {}).get('avg_price', 0)
    
    one_month = by_bucket.get('15-30', {}).get('avg_price', 0) or \
                by_bucket.get('15-30 days', {}).get('avg_price', 0)
    
    two_month = by_bucket.get('31-60', {}).get('avg_price', 0) or \
                by_bucket.get('31-60 days', {}).get('avg_price', 0)
    
    advance = by_bucket.get('60+', {}).get('avg_price', 0) or \
              by_bucket.get('61-90 days', {}).get('avg_price', 0) or \
              by_bucket.get('91-120 days', {}).get('avg_price', 0)
    
    # Calculate last-minute premium
    reference_price = one_month or two_month or advance
    
    if reference_price and last_min:
        last_minute_premium = (last_min - reference_price) / reference_price
        
        # High last-minute premium = business route
        if last_minute_premium > 0.30:
            business_signal += 3
        elif last_minute_premium > 0.20:
            business_signal += 2
        elif last_minute_premium > 0.10:
            business_signal += 1
        
        # Low or negative last-minute premium = leisure route
        # (or deals for unsold inventory)
        elif last_minute_premium < 0:
            leisure_signal += 1
    
    # Check if advance booking is rewarded
    if advance and one_month and one_month > 0:
        advance_discount = (one_month - advance) / one_month
        if advance_discount > 0.15:
            leisure_signal += 2
        elif advance_discount > 0.05:
            leisure_signal += 1
    
    return {
        'business_signal': business_signal,
        'leisure_signal': leisure_signal,
    }


def get_route_classification_details(
    origin: str,
    destination: str,
) -> Dict[str, Any]:
    """
    Get detailed route classification with supporting evidence.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        
    Returns:
        Dictionary with classification and supporting data
    """
    archetype = classify_route(origin, destination)
    
    weekday_stats = get_weekday_statistics(origin, destination)
    lead_time_stats = get_lead_time_statistics(origin, destination)
    
    weekday_pattern = _analyze_weekday_pattern(weekday_stats) if weekday_stats else {}
    lead_time_pattern = _analyze_lead_time_pattern(lead_time_stats) if lead_time_stats else {}
    
    # Get seasonal patterns if available
    seasonal = analyze_seasonal_patterns(origin, destination)
    lead_time_analysis = analyze_lead_time_patterns(origin, destination)
    
    # Build characteristics based on archetype
    if archetype == RouteArchetype.BUSINESS_HEAVY:
        characteristics = [
            "Higher prices on Monday and Friday",
            "Last-minute bookings carry significant premium",
            "Weekends often discounted",
            "Price less sensitive to seasons",
        ]
        recommendation = "Book 2-3 weeks ahead. Avoid Monday morning and Friday evening flights."
    elif archetype == RouteArchetype.LEISURE_HEAVY:
        characteristics = [
            "Weekend departures in higher demand",
            "Advance booking rewards with lower prices",
            "Strong seasonal price variations",
            "Holiday periods significantly more expensive",
        ]
        recommendation = "Book 6-8 weeks ahead for best prices. Consider weekday departures for savings."
    else:
        characteristics = [
            "Mixed business and leisure demand",
            "Moderate last-minute premium",
            "Some seasonal variation",
        ]
        recommendation = "Book 3-4 weeks ahead. Monitor for promotional pricing."
    
    return {
        'origin': origin,
        'destination': destination,
        'archetype': archetype.value,
        'archetype_name': {
            RouteArchetype.BUSINESS_HEAVY: "Business-Heavy",
            RouteArchetype.LEISURE_HEAVY: "Leisure-Heavy",
            RouteArchetype.MIXED: "Mixed",
        }[archetype],
        'characteristics': characteristics,
        'recommendation': recommendation,
        'signals': {
            'weekday_business': weekday_pattern.get('business_signal', 0),
            'weekday_leisure': weekday_pattern.get('leisure_signal', 0),
            'lead_time_business': lead_time_pattern.get('business_signal', 0),
            'lead_time_leisure': lead_time_pattern.get('leisure_signal', 0),
        },
        'weekday_stats': weekday_stats,
        'lead_time_stats': lead_time_stats,
        'seasonal_analysis': seasonal if seasonal.get('has_seasonal_data') else None,
        'lead_time_analysis': lead_time_analysis if lead_time_analysis.get('has_lead_time_data') else None,
    }


# =============================================================================
# ARCHETYPE-SPECIFIC STRATEGIES
# =============================================================================

def get_booking_strategy(
    archetype: RouteArchetype,
    lead_time_days: int,
) -> Dict[str, Any]:
    """
    Get optimal booking strategy based on route archetype.
    
    Args:
        archetype: Route archetype classification
        lead_time_days: Current days until departure
        
    Returns:
        Strategy recommendation dictionary
    """
    if archetype == RouteArchetype.BUSINESS_HEAVY:
        # Business routes: book earlier to avoid last-minute premium
        if lead_time_days > 21:
            urgency = "low"
            action = "WAIT"
            reasoning = "Business routes show stable pricing until 2-3 weeks out"
        elif lead_time_days > 14:
            urgency = "medium"
            action = "CONSIDER"
            reasoning = "Approaching optimal booking window for business routes"
        elif lead_time_days > 7:
            urgency = "high"
            action = "BUY"
            reasoning = "Last-minute premium typically starts within 1 week"
        else:
            urgency = "critical"
            action = "BUY NOW"
            reasoning = "In last-minute premium zone - prices likely highest"
        
        optimal_lead_time = "14-21 days"
        avoid_periods = ["Within 7 days of departure", "Monday/Friday peak times"]
        
    elif archetype == RouteArchetype.LEISURE_HEAVY:
        # Leisure routes: book well in advance
        if lead_time_days > 60:
            urgency = "low"
            action = "WAIT"
            reasoning = "Very early - watch for promotional pricing"
        elif lead_time_days > 45:
            urgency = "medium"
            action = "CONSIDER"
            reasoning = "Good window for advance booking savings"
        elif lead_time_days > 21:
            urgency = "medium"
            action = "BUY"
            reasoning = "Within typical optimal booking window for leisure"
        elif lead_time_days > 7:
            urgency = "high"
            action = "BUY"
            reasoning = "Prices typically rise as departure approaches"
        else:
            urgency = "critical"
            action = "BUY NOW"
            reasoning = "Limited inventory likely - book to secure seat"
        
        optimal_lead_time = "45-60 days"
        avoid_periods = ["Holiday weekends", "School vacation periods"]
        
    else:  # MIXED
        if lead_time_days > 30:
            urgency = "low"
            action = "WAIT"
            reasoning = "Monitor for deals - mixed routes have variable pricing"
        elif lead_time_days > 14:
            urgency = "medium"
            action = "CONSIDER"
            reasoning = "Good general booking window"
        elif lead_time_days > 7:
            urgency = "high"
            action = "BUY"
            reasoning = "Don't wait much longer"
        else:
            urgency = "critical"
            action = "BUY NOW"
            reasoning = "Inside last week - book now"
        
        optimal_lead_time = "21-30 days"
        avoid_periods = ["Major holidays", "Last-minute bookings"]
    
    return {
        'archetype': archetype.value,
        'lead_time_days': lead_time_days,
        'urgency': urgency,
        'action': action,
        'reasoning': reasoning,
        'optimal_lead_time': optimal_lead_time,
        'avoid_periods': avoid_periods,
    }


def get_price_expectation_adjustments(
    archetype: RouteArchetype,
    departure_date: date,
) -> Dict[str, float]:
    """
    Get price expectation adjustments based on archetype and date.
    
    Returns multipliers to apply to baseline expectations.
    
    Args:
        archetype: Route archetype
        departure_date: Travel date
        
    Returns:
        Dictionary of adjustment factors
    """
    weekday = departure_date.weekday()
    month = departure_date.month
    
    # Day of week multiplier
    if archetype == RouteArchetype.BUSINESS_HEAVY:
        weekday_multipliers = {
            0: 1.15,  # Monday - peak
            1: 1.05,  # Tuesday
            2: 1.00,  # Wednesday
            3: 1.05,  # Thursday
            4: 1.15,  # Friday - peak
            5: 0.90,  # Saturday - discount
            6: 1.05,  # Sunday - return traffic
        }
    elif archetype == RouteArchetype.LEISURE_HEAVY:
        weekday_multipliers = {
            0: 0.95,  # Monday - lower demand
            1: 0.90,  # Tuesday - lowest
            2: 0.90,  # Wednesday - low
            3: 0.95,  # Thursday
            4: 1.05,  # Friday - weekend start
            5: 1.10,  # Saturday - peak leisure
            6: 1.05,  # Sunday
        }
    else:
        weekday_multipliers = {i: 1.0 for i in range(7)}
    
    # Seasonal multiplier
    if archetype == RouteArchetype.LEISURE_HEAVY:
        # Summer and holiday months premium
        if month in [6, 7, 8, 12]:
            seasonal_mult = 1.20
        elif month in [11, 3, 4]:
            seasonal_mult = 1.10
        else:
            seasonal_mult = 1.00
    else:
        seasonal_mult = 1.00  # Business routes less seasonal
    
    return {
        'weekday_multiplier': weekday_multipliers.get(weekday, 1.0),
        'seasonal_multiplier': seasonal_mult,
        'combined_multiplier': weekday_multipliers.get(weekday, 1.0) * seasonal_mult,
    }


# =============================================================================
# COMMON ROUTE CLASSIFICATIONS
# =============================================================================

# Pre-defined classifications for well-known routes
KNOWN_ROUTE_ARCHETYPES = {
    # Business-heavy routes
    ('JFK', 'LAX'): RouteArchetype.BUSINESS_HEAVY,
    ('LAX', 'JFK'): RouteArchetype.BUSINESS_HEAVY,
    ('JFK', 'SFO'): RouteArchetype.BUSINESS_HEAVY,
    ('SFO', 'JFK'): RouteArchetype.BUSINESS_HEAVY,
    ('ORD', 'LHR'): RouteArchetype.BUSINESS_HEAVY,
    ('LHR', 'ORD'): RouteArchetype.BUSINESS_HEAVY,
    ('JFK', 'LHR'): RouteArchetype.BUSINESS_HEAVY,
    ('LHR', 'JFK'): RouteArchetype.BUSINESS_HEAVY,
    
    # Leisure-heavy routes
    ('JFK', 'CUN'): RouteArchetype.LEISURE_HEAVY,
    ('CUN', 'JFK'): RouteArchetype.LEISURE_HEAVY,
    ('LAX', 'HNL'): RouteArchetype.LEISURE_HEAVY,
    ('HNL', 'LAX'): RouteArchetype.LEISURE_HEAVY,
    ('JFK', 'SJU'): RouteArchetype.LEISURE_HEAVY,
    ('SJU', 'JFK'): RouteArchetype.LEISURE_HEAVY,
    ('MIA', 'NAS'): RouteArchetype.LEISURE_HEAVY,
    ('NAS', 'MIA'): RouteArchetype.LEISURE_HEAVY,
    
    # Mixed routes
    ('JFK', 'MIA'): RouteArchetype.MIXED,
    ('MIA', 'JFK'): RouteArchetype.MIXED,
}


def get_archetype_with_fallback(
    origin: str,
    destination: str,
    use_known: bool = True,
) -> RouteArchetype:
    """
    Get route archetype with fallback to known classifications.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        use_known: Whether to use pre-defined classifications
        
    Returns:
        RouteArchetype
    """
    # Check known routes first
    if use_known:
        key = (origin.upper(), destination.upper())
        if key in KNOWN_ROUTE_ARCHETYPES:
            return KNOWN_ROUTE_ARCHETYPES[key]
    
    # Try data-driven classification
    return classify_route(origin, destination)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Route Classifier")
    parser.add_argument("origin", help="Origin airport code")
    parser.add_argument("dest", help="Destination airport code")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    if args.detailed:
        details = get_route_classification_details(args.origin, args.dest)
        print(f"\n{args.origin} → {args.dest}")
        print(f"Classification: {details['archetype_name']}")
        print(f"\nCharacteristics:")
        for c in details['characteristics']:
            print(f"  • {c}")
        print(f"\nRecommendation: {details['recommendation']}")
        
        if details['signals']:
            print(f"\nSignal Scores:")
            print(f"  Weekday Business: {details['signals']['weekday_business']}")
            print(f"  Weekday Leisure: {details['signals']['weekday_leisure']}")
            print(f"  Lead Time Business: {details['signals']['lead_time_business']}")
            print(f"  Lead Time Leisure: {details['signals']['lead_time_leisure']}")
    else:
        archetype = classify_route(args.origin, args.dest)
        print(f"{args.origin} → {args.dest}: {archetype.value}")
