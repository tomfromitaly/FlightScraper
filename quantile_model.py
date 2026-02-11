"""
Quantile Model for Flight Price Tracker.

Provides fair value distribution calculations, regime detection,
and anomaly identification for flight pricing.
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from models import FairValueDistribution, PriceRegime
from database import (
    get_combinations_for_route,
    get_all_offers_for_combo,
    get_price_dispersion,
    get_latest_combinations,
)
from baseline_builder import get_best_baseline
from utils import calculate_statistics, calculate_z_score

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# FAIR VALUE CALCULATION
# =============================================================================

def get_fair_value_distribution(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    lead_time_days: Optional[int] = None,
) -> Optional[FairValueDistribution]:
    """
    Get percentile thresholds for a specific itinerary.
    
    Tries to find the most specific baseline available:
    1. Month + lead time combination
    2. Month only
    3. Lead time only
    4. Overall route baseline
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        departure_date: Flight departure date
        return_date: Flight return date
        lead_time_days: Days until departure (calculated if not provided)
        
    Returns:
        FairValueDistribution or None if no baseline exists
    """
    if lead_time_days is None:
        lead_time_days = (departure_date - date.today()).days
    
    month = departure_date.month
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][departure_date.weekday()]
    
    return get_best_baseline(
        origin=origin,
        destination=destination,
        month=month,
        day_of_week=day_of_week,
        lead_time_days=lead_time_days,
    )


def calculate_price_percentile(
    price: float,
    fair_value: FairValueDistribution,
) -> float:
    """
    Calculate which percentile a price falls into.
    
    Args:
        price: The price to evaluate
        fair_value: The baseline distribution
        
    Returns:
        Estimated percentile (0-100)
    """
    return fair_value.get_percentile(price)


def is_good_deal(
    price: float,
    fair_value: FairValueDistribution,
    threshold_percentile: float = 25.0,
) -> bool:
    """
    Determine if a price is a good deal (below threshold percentile).
    
    Args:
        price: The price to evaluate
        fair_value: The baseline distribution
        threshold_percentile: Percentile threshold (default: P25)
        
    Returns:
        True if price is a deal
    """
    percentile = fair_value.get_percentile(price)
    return percentile <= threshold_percentile


def get_deal_quality(
    price: float,
    fair_value: FairValueDistribution,
) -> Dict[str, Any]:
    """
    Get detailed deal quality assessment.
    
    Args:
        price: The price to evaluate
        fair_value: The baseline distribution
        
    Returns:
        Dictionary with deal assessment
    """
    percentile = fair_value.get_percentile(price)
    z_score = fair_value.calculate_z_score(price)
    
    # Determine quality tier
    if percentile <= 10:
        tier = "exceptional"
        color = "green"
        recommendation = "Book immediately - exceptional deal"
    elif percentile <= 25:
        tier = "great"
        color = "green"
        recommendation = "Book soon - great deal"
    elif percentile <= 50:
        tier = "good"
        color = "blue"
        recommendation = "Reasonable price - consider booking"
    elif percentile <= 75:
        tier = "fair"
        color = "yellow"
        recommendation = "Above average - wait if flexible"
    elif percentile <= 90:
        tier = "expensive"
        color = "orange"
        recommendation = "Expensive - wait for better price"
    else:
        tier = "very_expensive"
        color = "red"
        recommendation = "Very expensive - definitely wait"
    
    # Calculate savings vs median
    savings_vs_median = fair_value.p50 - price
    savings_vs_median_pct = (savings_vs_median / fair_value.p50 * 100) if fair_value.p50 > 0 else 0
    
    return {
        'price': price,
        'percentile': round(percentile, 1),
        'z_score': round(z_score, 2),
        'tier': tier,
        'color': color,
        'recommendation': recommendation,
        'fair_value': fair_value.p50,
        'savings_vs_median': round(savings_vs_median, 2),
        'savings_vs_median_pct': round(savings_vs_median_pct, 1),
        'label': fair_value.get_percentile_label(price),
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(
    current_price: float,
    fair_value: FairValueDistribution,
    offer_count: Optional[int] = None,
    price_spread: Optional[float] = None,
) -> PriceRegime:
    """
    Detect current pricing regime using z-score and offer depth.
    
    Regimes:
    - PROMO: Z-score < -1.5 (anomaly low price)
    - NORMAL: -1.5 <= Z <= 1.5
    - ELEVATED: Z > 1.5 (inventory pressure starting)
    - SCARCITY: Z > 2.5 (likely to sell out)
    
    Args:
        current_price: Current offer price
        fair_value: Baseline distribution
        offer_count: Number of offers available (optional signal)
        price_spread: Spread between top-K offers (optional signal)
        
    Returns:
        PriceRegime enum value
    """
    z_score = fair_value.calculate_z_score(current_price)
    
    # Base regime from z-score
    if z_score < -1.5:
        base_regime = PriceRegime.PROMO
    elif z_score <= 1.5:
        base_regime = PriceRegime.NORMAL
    elif z_score <= 2.5:
        base_regime = PriceRegime.ELEVATED
    else:
        base_regime = PriceRegime.SCARCITY
    
    # Adjust based on offer depth signals
    if offer_count is not None and price_spread is not None:
        # Low offer count + tight spread = scarcity signal
        if offer_count <= 2 and price_spread < fair_value.p50 * 0.05:
            # Upgrade regime if inventory signals suggest scarcity
            if base_regime == PriceRegime.NORMAL:
                base_regime = PriceRegime.ELEVATED
            elif base_regime == PriceRegime.ELEVATED:
                base_regime = PriceRegime.SCARCITY
    
    return base_regime


def get_regime_with_context(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    current_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get full regime analysis with context.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        departure_date: Departure date
        return_date: Return date
        current_price: Current price (fetches latest if not provided)
        
    Returns:
        Dictionary with regime and context
    """
    # Get fair value
    lead_time = (departure_date - date.today()).days
    fair_value = get_fair_value_distribution(
        origin, destination, departure_date, return_date, lead_time
    )
    
    if not fair_value:
        return {
            'has_baseline': False,
            'regime': PriceRegime.NORMAL,
            'message': 'No baseline data available',
        }
    
    # Get current price if not provided
    if current_price is None:
        combos = get_latest_combinations(origin, destination, offer_rank=1)
        matching = [c for c in combos 
                   if str(c['departure_date']) == str(departure_date) 
                   and str(c['return_date']) == str(return_date)]
        if matching:
            current_price = matching[0]['price']
        else:
            return {
                'has_baseline': True,
                'fair_value': fair_value.model_dump(),
                'regime': PriceRegime.NORMAL,
                'message': 'No current price data',
            }
    
    # Get offer depth
    dispersion = get_price_dispersion(origin, destination, departure_date, return_date)
    offer_count = dispersion['offer_count'] if dispersion else None
    price_spread = dispersion['spread'] if dispersion else None
    
    # Detect regime
    regime = detect_regime(current_price, fair_value, offer_count, price_spread)
    z_score = fair_value.calculate_z_score(current_price)
    percentile = fair_value.get_percentile(current_price)
    
    # Generate message
    regime_messages = {
        PriceRegime.PROMO: "Promotional pricing detected - unusually low price",
        PriceRegime.NORMAL: "Normal pricing - within expected range",
        PriceRegime.ELEVATED: "Elevated pricing - inventory pressure building",
        PriceRegime.SCARCITY: "Scarcity pricing - high demand, limited inventory",
    }
    
    return {
        'has_baseline': True,
        'regime': regime,
        'regime_name': regime.value,
        'message': regime_messages[regime],
        'current_price': current_price,
        'z_score': round(z_score, 2),
        'percentile': round(percentile, 1),
        'fair_value': {
            'p10': fair_value.p10,
            'p25': fair_value.p25,
            'p50': fair_value.p50,
            'p75': fair_value.p75,
            'p90': fair_value.p90,
        },
        'offer_depth': {
            'count': offer_count,
            'spread': price_spread,
        } if dispersion else None,
    }


# =============================================================================
# PROMO ANOMALY DETECTION
# =============================================================================

def detect_promo_anomaly(
    current_price: float,
    fair_value: FairValueDistribution,
    threshold_z: float = -1.5,
) -> Dict[str, Any]:
    """
    Z-score based detector for promotional pricing anomalies.
    
    Args:
        current_price: Current price to evaluate
        fair_value: Baseline distribution
        threshold_z: Z-score threshold for anomaly (default: -1.5)
        
    Returns:
        Dictionary with anomaly detection results
    """
    z_score = fair_value.calculate_z_score(current_price)
    percentile = fair_value.get_percentile(current_price)
    
    is_anomaly = z_score < threshold_z or z_score > 2.0
    
    if z_score < threshold_z:
        anomaly_type = "promo"
        severity = "low" if z_score > -2.0 else "very_low"
        message = f"Price is {abs(z_score):.1f} std devs below normal - likely promotional"
    elif z_score > 2.5:
        anomaly_type = "scarcity"
        severity = "extreme_high"
        message = f"Price is {z_score:.1f} std devs above normal - extreme scarcity"
    elif z_score > 2.0:
        anomaly_type = "elevated"
        severity = "high"
        message = f"Price is {z_score:.1f} std devs above normal - elevated demand"
    else:
        anomaly_type = None
        severity = "normal"
        message = "Price within normal range"
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_type': anomaly_type,
        'severity': severity,
        'z_score': round(z_score, 2),
        'percentile': round(percentile, 1),
        'message': message,
        'current_price': current_price,
        'expected_price': fair_value.p50,
        'savings_if_promo': round(fair_value.p50 - current_price, 2) if z_score < 0 else 0,
    }


def find_promo_opportunities(
    origin: str,
    destination: str,
    target_month: Optional[str] = None,
    z_threshold: float = -1.0,
) -> List[Dict[str, Any]]:
    """
    Find all current promotional opportunities for a route.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        target_month: Optional month filter (YYYY-MM)
        z_threshold: Z-score threshold for promo detection
        
    Returns:
        List of promotional opportunities sorted by savings
    """
    combos = get_latest_combinations(origin, destination, target_month, offer_rank=1)
    
    opportunities = []
    
    for combo in combos:
        dep_date = combo['departure_date']
        ret_date = combo['return_date']
        price = combo['price']
        
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        if isinstance(ret_date, str):
            ret_date = datetime.strptime(ret_date, "%Y-%m-%d").date()
        
        lead_time = (dep_date - date.today()).days
        if lead_time < 0:
            continue
        
        fair_value = get_fair_value_distribution(
            origin, destination, dep_date, ret_date, lead_time
        )
        
        if not fair_value:
            continue
        
        z_score = fair_value.calculate_z_score(price)
        
        if z_score <= z_threshold:
            savings = fair_value.p50 - price
            opportunities.append({
                'departure_date': str(dep_date),
                'return_date': str(ret_date),
                'price': price,
                'fair_value': fair_value.p50,
                'savings': round(savings, 2),
                'savings_pct': round(savings / fair_value.p50 * 100, 1) if fair_value.p50 > 0 else 0,
                'z_score': round(z_score, 2),
                'percentile': round(fair_value.get_percentile(price), 1),
                'lead_time_days': lead_time,
            })
    
    # Sort by savings (best deals first)
    opportunities.sort(key=lambda x: x['savings'], reverse=True)
    
    return opportunities


# =============================================================================
# COMPETITIVE PRESSURE (CARRIER SPREAD)
# =============================================================================

def calculate_carrier_spread(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
) -> Dict[str, Any]:
    """
    Monitor spread between top carriers on same dates.
    
    Price compression (small spread) often precedes price moves.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        departure_date: Departure date
        return_date: Return date
        
    Returns:
        Dictionary with carrier spread analysis
    """
    offers = get_all_offers_for_combo(origin, destination, departure_date, return_date)
    
    if not offers or len(offers) < 2:
        return {
            'has_data': False,
            'message': 'Insufficient offer data for spread analysis',
        }
    
    # Group by carrier
    prices_by_carrier = {}
    for offer in offers:
        carrier = offer.get('airline_outbound', 'Unknown')
        if carrier not in prices_by_carrier:
            prices_by_carrier[carrier] = []
        prices_by_carrier[carrier].append(offer['price'])
    
    # Get min price per carrier
    carrier_prices = {
        carrier: min(prices) 
        for carrier, prices in prices_by_carrier.items()
    }
    
    prices = list(carrier_prices.values())
    min_price = min(prices)
    max_price = max(prices)
    spread = max_price - min_price
    median_price = sorted(prices)[len(prices) // 2]
    
    # Compression ratio: lower = more compressed = potential move coming
    compression_ratio = spread / median_price if median_price > 0 else 0
    
    # Identify leading carrier
    leading_carrier = min(carrier_prices, key=carrier_prices.get)
    
    # Compression signal: ratio < 5% often precedes price moves
    compression_signal = compression_ratio < 0.05
    
    return {
        'has_data': True,
        'carrier_count': len(carrier_prices),
        'spread': round(spread, 2),
        'compression_ratio': round(compression_ratio, 3),
        'compression_signal': compression_signal,
        'leading_carrier': leading_carrier,
        'min_price': min_price,
        'max_price': max_price,
        'carrier_prices': carrier_prices,
        'interpretation': (
            "Price compression detected - watch for potential move" 
            if compression_signal 
            else "Normal competitive spread"
        ),
    }


# =============================================================================
# OFFER DEPTH PROXY
# =============================================================================

def analyze_offer_depth(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
) -> Dict[str, Any]:
    """
    Analyze offer depth as inventory pressure signal.
    
    Fewer offers + narrowing price dispersion = inventory pressure.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        departure_date: Departure date
        return_date: Return date
        
    Returns:
        Dictionary with offer depth analysis
    """
    dispersion = get_price_dispersion(origin, destination, departure_date, return_date)
    
    if not dispersion:
        return {
            'has_data': False,
            'message': 'No offer data available',
        }
    
    offer_count = dispersion['offer_count']
    spread = dispersion['spread']
    spread_pct = dispersion['spread_pct']
    
    # Determine inventory pressure level
    if offer_count <= 1:
        pressure = "extreme"
        message = "Single offer available - high scarcity risk"
    elif offer_count <= 2 and spread_pct < 5:
        pressure = "high"
        message = "Few offers with tight spread - inventory pressure building"
    elif offer_count <= 3 and spread_pct < 10:
        pressure = "moderate"
        message = "Limited offers with moderate spread"
    else:
        pressure = "low"
        message = "Healthy offer depth - normal inventory"
    
    return {
        'has_data': True,
        'offer_count': offer_count,
        'price_spread': round(spread, 2),
        'spread_percent': round(spread_pct, 1),
        'inventory_pressure': pressure,
        'message': message,
        'prices': dispersion.get('prices', []),
        'signal_strength': {
            'extreme': 1.0,
            'high': 0.75,
            'moderate': 0.5,
            'low': 0.25,
        }.get(pressure, 0.25),
    }


# =============================================================================
# COMBINED ANALYSIS
# =============================================================================

def get_complete_price_analysis(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    current_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get complete price analysis combining all signals.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        departure_date: Departure date
        return_date: Return date
        current_price: Current price (fetches if not provided)
        
    Returns:
        Comprehensive analysis dictionary
    """
    lead_time = (departure_date - date.today()).days
    
    # Get fair value
    fair_value = get_fair_value_distribution(
        origin, destination, departure_date, return_date, lead_time
    )
    
    # Get current price if not provided
    if current_price is None:
        combos = get_latest_combinations(origin, destination, offer_rank=1)
        matching = [c for c in combos 
                   if str(c['departure_date']) == str(departure_date) 
                   and str(c['return_date']) == str(return_date)]
        if matching:
            current_price = matching[0]['price']
    
    result = {
        'origin': origin,
        'destination': destination,
        'departure_date': str(departure_date),
        'return_date': str(return_date),
        'lead_time_days': lead_time,
        'current_price': current_price,
        'has_baseline': fair_value is not None,
    }
    
    if not fair_value:
        result['message'] = 'No baseline data - cannot perform full analysis'
        return result
    
    if current_price is None:
        result['message'] = 'No current price data'
        return result
    
    # Deal quality
    deal_quality = get_deal_quality(current_price, fair_value)
    result['deal_quality'] = deal_quality
    
    # Regime
    regime_info = get_regime_with_context(
        origin, destination, departure_date, return_date, current_price
    )
    result['regime'] = regime_info['regime'].value
    result['regime_message'] = regime_info['message']
    
    # Promo anomaly
    anomaly = detect_promo_anomaly(current_price, fair_value)
    result['anomaly'] = anomaly
    
    # Offer depth
    depth = analyze_offer_depth(origin, destination, departure_date, return_date)
    result['offer_depth'] = depth
    
    # Carrier spread
    carrier = calculate_carrier_spread(origin, destination, departure_date, return_date)
    result['carrier_spread'] = carrier
    
    # Overall recommendation
    if anomaly['is_anomaly'] and anomaly['anomaly_type'] == 'promo':
        result['recommendation'] = "BUY - Promotional pricing detected"
        result['urgency'] = "high"
    elif regime_info['regime'] == PriceRegime.SCARCITY:
        result['recommendation'] = "BUY - Scarcity pricing, may sell out"
        result['urgency'] = "high"
    elif deal_quality['tier'] in ['exceptional', 'great']:
        result['recommendation'] = "BUY - Good deal"
        result['urgency'] = "medium"
    elif deal_quality['tier'] in ['expensive', 'very_expensive']:
        result['recommendation'] = "WAIT - Price above normal"
        result['urgency'] = "low"
    else:
        result['recommendation'] = "WATCH - Fair price, monitor for better deal"
        result['urgency'] = "low"
    
    return result
