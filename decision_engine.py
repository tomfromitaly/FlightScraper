"""
Decision Engine for Flight Price Tracker.

Outputs BUY, WAIT, or WATCH_CLOSELY recommendations with confidence
levels and deadlines based on fair value analysis, regime detection,
and user risk profile.
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

from models import (
    Decision,
    DecisionResult,
    FairValueDistribution,
    PriceRegime,
    RiskProfile,
    RouteArchetype,
)
from quantile_model import (
    get_fair_value_distribution,
    detect_regime,
    get_deal_quality,
    analyze_offer_depth,
)
from route_classifier import (
    get_archetype_with_fallback,
    get_booking_strategy,
)
from database import (
    get_price_history_for_combo,
    get_latest_combinations,
    get_price_dispersion,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# EXPECTED VALUE OF WAITING
# =============================================================================

def compute_expected_value_of_waiting(
    current_price: float,
    fair_value: FairValueDistribution,
    days_until_deadline: int,
    days_until_departure: int,
    regime: PriceRegime,
    historical_prices: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute expected value of waiting vs buying now.
    
    EV_wait = E[min future price before deadline] - current_price - risk_penalty
    
    Buy when EV_wait <= 0 (expected savings from waiting <= risk)
    
    Args:
        current_price: Current offer price
        fair_value: Baseline distribution
        days_until_deadline: Days until latest acceptable booking date
        days_until_departure: Days until departure
        regime: Current pricing regime
        historical_prices: Recent price observations (for volatility)
        
    Returns:
        Dictionary with EV analysis
    """
    # Calculate historical volatility
    if historical_prices and len(historical_prices) >= 3:
        mean_price = sum(historical_prices) / len(historical_prices)
        variance = sum((p - mean_price) ** 2 for p in historical_prices) / len(historical_prices)
        volatility = math.sqrt(variance)
        volatility_pct = volatility / mean_price if mean_price > 0 else 0.1
    else:
        # Default volatility assumption: 10%
        volatility_pct = 0.10
        volatility = current_price * volatility_pct
    
    # Estimate expected minimum future price
    # Based on regime and time remaining
    expected_min_future = _estimate_future_minimum(
        current_price=current_price,
        fair_value=fair_value,
        days_remaining=days_until_deadline,
        volatility=volatility,
        regime=regime,
    )
    
    # Calculate risk penalty
    # Higher penalty for:
    # - Shorter time to departure
    # - Scarcity regime
    # - High volatility
    risk_penalty = _calculate_risk_penalty(
        days_until_departure=days_until_departure,
        days_until_deadline=days_until_deadline,
        regime=regime,
        volatility_pct=volatility_pct,
        fair_value=fair_value,
    )
    
    # EV of waiting
    ev_wait = expected_min_future - current_price - risk_penalty
    
    # Confidence based on data quality
    if historical_prices and len(historical_prices) >= 10:
        confidence = 0.8
    elif historical_prices and len(historical_prices) >= 5:
        confidence = 0.6
    else:
        confidence = 0.4
    
    return {
        'current_price': current_price,
        'expected_min_future': round(expected_min_future, 2),
        'risk_penalty': round(risk_penalty, 2),
        'ev_wait': round(ev_wait, 2),
        'should_buy': ev_wait <= 0,
        'volatility_pct': round(volatility_pct * 100, 1),
        'confidence': confidence,
        'explanation': (
            f"Expected future minimum: ${expected_min_future:.0f}, "
            f"Risk penalty: ${risk_penalty:.0f}, "
            f"EV of waiting: ${ev_wait:.0f}"
        ),
    }


def _estimate_future_minimum(
    current_price: float,
    fair_value: FairValueDistribution,
    days_remaining: int,
    volatility: float,
    regime: PriceRegime,
) -> float:
    """
    Estimate the expected minimum price before deadline.
    
    Uses regime and volatility to project price trajectory.
    """
    # Base expectation: prices tend toward P50
    mean_reversion_target = fair_value.p50
    
    # Regime-based adjustments
    if regime == PriceRegime.PROMO:
        # Promo prices often snap back - expect rebound
        # Future minimum likely near current (promo is the minimum)
        expected_min = current_price * 1.0
    
    elif regime == PriceRegime.SCARCITY:
        # Scarcity - prices likely to rise further
        expected_min = current_price * 1.05
    
    elif regime == PriceRegime.ELEVATED:
        # Elevated - might come down, but risky
        expected_min = current_price * 0.95
    
    else:  # NORMAL
        # Normal - might find a deal
        # More days = more chances to find lower price
        # Expected minimum decreases with more observations
        # Approximate using order statistics
        
        # Expected minimum of n observations from normal distribution
        # E[min] ≈ μ - σ * √(2 * ln(n))
        # where n = number of future observations
        
        n_future_observations = max(1, days_remaining // 2)  # Assume checking every 2 days
        
        if n_future_observations > 1 and volatility > 0:
            reduction_factor = math.sqrt(2 * math.log(n_future_observations))
            expected_min = current_price - volatility * reduction_factor * 0.5
        else:
            expected_min = current_price
    
    # Floor at P10 (unlikely to go below great deal threshold)
    expected_min = max(expected_min, fair_value.p10)
    
    return expected_min


def _calculate_risk_penalty(
    days_until_departure: int,
    days_until_deadline: int,
    regime: PriceRegime,
    volatility_pct: float,
    fair_value: FairValueDistribution,
) -> float:
    """
    Calculate risk penalty for waiting.
    
    Risk increases as:
    - Departure approaches
    - Inventory becomes scarce
    - Volatility is high
    """
    base_penalty = 0
    
    # Time-based risk
    if days_until_departure <= 3:
        base_penalty += fair_value.p50 * 0.20  # 20% penalty
    elif days_until_departure <= 7:
        base_penalty += fair_value.p50 * 0.15
    elif days_until_departure <= 14:
        base_penalty += fair_value.p50 * 0.10
    elif days_until_departure <= 30:
        base_penalty += fair_value.p50 * 0.05
    
    # Regime-based risk
    regime_multiplier = {
        PriceRegime.PROMO: 0.5,     # Low risk - lock in the deal
        PriceRegime.NORMAL: 1.0,
        PriceRegime.ELEVATED: 1.5,
        PriceRegime.SCARCITY: 2.5,  # High risk
    }
    base_penalty *= regime_multiplier.get(regime, 1.0)
    
    # Volatility adjustment
    # High volatility = higher risk of missing current price
    if volatility_pct > 0.15:
        base_penalty *= 1.3
    elif volatility_pct < 0.05:
        base_penalty *= 0.8
    
    return base_penalty


# =============================================================================
# REBOUND RISK SCORER
# =============================================================================

def calculate_rebound_risk(
    current_price: float,
    recent_prices: List[float],
    fair_value: FairValueDistribution,
    regime: PriceRegime,
    days_until_departure: int,
) -> Dict[str, Any]:
    """
    After a dip, estimate probability that price snaps back quickly.
    
    High rebound risk = should buy now to lock in low price.
    
    Args:
        current_price: Current price
        recent_prices: Last 5-10 price observations (oldest first)
        fair_value: Baseline distribution
        regime: Current regime
        days_until_departure: Days until flight
        
    Returns:
        Dictionary with rebound risk assessment (0-1 probability)
    """
    if len(recent_prices) < 2:
        return {
            'rebound_risk': 0.5,
            'confidence': 0.3,
            'message': 'Insufficient price history',
        }
    
    # Calculate how far below normal the current price is
    percentile = fair_value.get_percentile(current_price)
    below_median_pct = (fair_value.p50 - current_price) / fair_value.p50 * 100 if fair_value.p50 > 0 else 0
    
    # Calculate price velocity (rate of recent change)
    if len(recent_prices) >= 3:
        recent_change = recent_prices[-1] - recent_prices[-3]
        velocity = recent_change / recent_prices[-3] * 100 if recent_prices[-3] > 0 else 0
    else:
        recent_change = recent_prices[-1] - recent_prices[0]
        velocity = recent_change / recent_prices[0] * 100 if recent_prices[0] > 0 else 0
    
    # Base rebound probability factors
    rebound_risk = 0.0
    factors = []
    
    # Factor 1: How far below median (bigger dip = higher rebound risk)
    if below_median_pct > 20:
        rebound_risk += 0.35
        factors.append(f"Price {below_median_pct:.0f}% below median")
    elif below_median_pct > 10:
        rebound_risk += 0.25
        factors.append(f"Price {below_median_pct:.0f}% below median")
    elif below_median_pct > 5:
        rebound_risk += 0.15
        factors.append(f"Price slightly below median")
    
    # Factor 2: Promo regime (anomaly low = likely to rebound)
    if regime == PriceRegime.PROMO:
        rebound_risk += 0.30
        factors.append("Promotional pricing detected")
    elif regime == PriceRegime.NORMAL and percentile < 25:
        rebound_risk += 0.15
        factors.append("Good deal in normal market")
    
    # Factor 3: Price velocity (dropping fast = flash sale, rebounds fast)
    if velocity < -10:  # Dropped >10% recently
        rebound_risk += 0.20
        factors.append(f"Recent sharp drop ({velocity:.0f}%)")
    elif velocity < -5:
        rebound_risk += 0.10
        factors.append("Recent price decline")
    
    # Factor 4: Time to departure (closer = higher rebound risk)
    if days_until_departure <= 7:
        rebound_risk += 0.20
        factors.append("Very close to departure")
    elif days_until_departure <= 14:
        rebound_risk += 0.10
        factors.append("Close to departure")
    
    # Cap at 0.95 (never 100% certain)
    rebound_risk = min(0.95, rebound_risk)
    
    # Confidence based on data
    confidence = min(0.8, 0.3 + len(recent_prices) * 0.05)
    
    # Risk level interpretation
    if rebound_risk >= 0.7:
        risk_level = "high"
        message = "High rebound risk - lock in this price"
    elif rebound_risk >= 0.4:
        risk_level = "moderate"
        message = "Moderate rebound risk - consider booking soon"
    else:
        risk_level = "low"
        message = "Low rebound risk - safe to monitor"
    
    return {
        'rebound_risk': round(rebound_risk, 2),
        'risk_level': risk_level,
        'message': message,
        'factors': factors,
        'confidence': confidence,
        'percentile': round(percentile, 1),
        'velocity': round(velocity, 1),
    }


# =============================================================================
# MAIN DECISION ENGINE
# =============================================================================

def make_decision(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    current_price: Optional[float] = None,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    booking_deadline: Optional[date] = None,
) -> DecisionResult:
    """
    Make a BUY, WAIT, or WATCH_CLOSELY decision.
    
    Combines:
    - Fair value analysis
    - Regime detection
    - EV of waiting calculation
    - Rebound risk assessment
    - User risk profile
    - Route archetype
    
    Args:
        origin: Origin airport
        destination: Destination airport
        departure_date: Flight departure date
        return_date: Flight return date
        current_price: Current price (fetches if not provided)
        risk_profile: User's risk tolerance
        booking_deadline: Latest acceptable booking date
        
    Returns:
        DecisionResult with recommendation
    """
    today = date.today()
    days_until_departure = (departure_date - today).days
    
    # Set default deadline (2 days before departure)
    if booking_deadline is None:
        booking_deadline = departure_date - timedelta(days=2)
    days_until_deadline = (booking_deadline - today).days
    
    # Get fair value distribution
    fair_value = get_fair_value_distribution(
        origin, destination, departure_date, return_date, days_until_departure
    )
    
    # Get current price if not provided
    if current_price is None:
        combos = get_latest_combinations(origin, destination, offer_rank=1)
        matching = [c for c in combos 
                   if str(c['departure_date']) == str(departure_date) 
                   and str(c['return_date']) == str(return_date)]
        if matching:
            current_price = matching[0]['price']
    
    # Fallback if no data
    if current_price is None:
        return DecisionResult(
            decision=Decision.WATCH_CLOSELY,
            confidence=0.3,
            deadline=booking_deadline,
            current_price=0,
            fair_value=0,
            percentile=50,
            regime=PriceRegime.NORMAL,
            reasoning=["No current price data available"],
        )
    
    if fair_value is None:
        # No baseline - use heuristics
        return _make_decision_without_baseline(
            current_price, days_until_departure, days_until_deadline,
            booking_deadline, risk_profile
        )
    
    # Get regime
    dispersion = get_price_dispersion(origin, destination, departure_date, return_date)
    offer_count = dispersion['offer_count'] if dispersion else None
    price_spread = dispersion['spread'] if dispersion else None
    
    regime = detect_regime(current_price, fair_value, offer_count, price_spread)
    
    # Get price history for EV calculation
    history = get_price_history_for_combo(origin, destination, departure_date, return_date)
    historical_prices = [h['price'] for h in history[-10:]] if history else []
    
    # Calculate EV of waiting
    ev_analysis = compute_expected_value_of_waiting(
        current_price=current_price,
        fair_value=fair_value,
        days_until_deadline=days_until_deadline,
        days_until_departure=days_until_departure,
        regime=regime,
        historical_prices=historical_prices,
    )
    
    # Calculate rebound risk
    rebound_analysis = calculate_rebound_risk(
        current_price=current_price,
        recent_prices=historical_prices or [current_price],
        fair_value=fair_value,
        regime=regime,
        days_until_departure=days_until_departure,
    )
    
    # Get route archetype for strategy adjustment
    archetype = get_archetype_with_fallback(origin, destination)
    
    # Get deal quality
    deal_quality = get_deal_quality(current_price, fair_value)
    percentile = deal_quality['percentile']
    
    # Make decision based on all factors
    decision, confidence, reasoning = _synthesize_decision(
        ev_analysis=ev_analysis,
        rebound_analysis=rebound_analysis,
        deal_quality=deal_quality,
        regime=regime,
        risk_profile=risk_profile,
        days_until_departure=days_until_departure,
        days_until_deadline=days_until_deadline,
        archetype=archetype,
    )
    
    return DecisionResult(
        decision=decision,
        confidence=confidence,
        deadline=booking_deadline,
        expected_savings_if_wait=max(0, ev_analysis['ev_wait']) if ev_analysis['ev_wait'] > 0 else 0,
        rebound_risk=rebound_analysis['rebound_risk'],
        current_price=current_price,
        fair_value=fair_value.p50,
        percentile=percentile,
        regime=regime,
        reasoning=reasoning,
    )


def _make_decision_without_baseline(
    current_price: float,
    days_until_departure: int,
    days_until_deadline: int,
    booking_deadline: date,
    risk_profile: RiskProfile,
) -> DecisionResult:
    """
    Make decision when no baseline data exists.
    Uses conservative heuristics.
    """
    reasoning = ["No historical baseline available - using heuristics"]
    
    if days_until_departure <= 3:
        decision = Decision.BUY
        confidence = 0.7
        reasoning.append("Very close to departure - book now")
    elif days_until_departure <= 7:
        decision = Decision.BUY
        confidence = 0.6
        reasoning.append("Within 1 week - recommend booking")
    elif days_until_departure <= 14:
        if risk_profile == RiskProfile.CONSERVATIVE:
            decision = Decision.BUY
            confidence = 0.5
            reasoning.append("Conservative profile - book within 2 weeks")
        else:
            decision = Decision.WATCH_CLOSELY
            confidence = 0.5
            reasoning.append("Monitor prices closely")
    else:
        decision = Decision.WAIT
        confidence = 0.4
        reasoning.append("Sufficient time to monitor")
    
    return DecisionResult(
        decision=decision,
        confidence=confidence,
        deadline=booking_deadline,
        current_price=current_price,
        fair_value=current_price,  # Assume current is fair when no baseline
        percentile=50,
        regime=PriceRegime.NORMAL,
        reasoning=reasoning,
    )


def _synthesize_decision(
    ev_analysis: Dict[str, Any],
    rebound_analysis: Dict[str, Any],
    deal_quality: Dict[str, Any],
    regime: PriceRegime,
    risk_profile: RiskProfile,
    days_until_departure: int,
    days_until_deadline: int,
    archetype: RouteArchetype,
) -> tuple[Decision, float, List[str]]:
    """
    Synthesize final decision from all analysis components.
    
    Returns (decision, confidence, reasoning list)
    """
    reasoning = []
    buy_score = 0
    wait_score = 0
    
    # Factor 1: EV of waiting
    if ev_analysis['should_buy']:
        buy_score += 2
        reasoning.append(f"EV analysis: Waiting EV is ${ev_analysis['ev_wait']:.0f} - favor buying")
    else:
        wait_score += 2
        reasoning.append(f"EV analysis: Expected ${abs(ev_analysis['ev_wait']):.0f} savings if waiting")
    
    # Factor 2: Deal quality
    tier = deal_quality['tier']
    if tier in ['exceptional', 'great']:
        buy_score += 3
        reasoning.append(f"Deal quality: {tier} (P{deal_quality['percentile']:.0f})")
    elif tier in ['good']:
        buy_score += 1
        reasoning.append(f"Deal quality: {tier} - reasonable price")
    elif tier in ['expensive', 'very_expensive']:
        wait_score += 2
        reasoning.append(f"Deal quality: {tier} - price above normal")
    
    # Factor 3: Regime
    if regime == PriceRegime.PROMO:
        buy_score += 3
        reasoning.append("Regime: PROMO - promotional pricing, book now")
    elif regime == PriceRegime.SCARCITY:
        buy_score += 2
        reasoning.append("Regime: SCARCITY - limited inventory")
    elif regime == PriceRegime.ELEVATED:
        wait_score += 1
        reasoning.append("Regime: ELEVATED - prices higher than normal")
    else:
        reasoning.append("Regime: NORMAL")
    
    # Factor 4: Rebound risk
    rebound = rebound_analysis['rebound_risk']
    if rebound >= 0.7:
        buy_score += 2
        reasoning.append(f"Rebound risk: HIGH ({rebound:.0%}) - price may snap back")
    elif rebound >= 0.4:
        buy_score += 1
        reasoning.append(f"Rebound risk: MODERATE ({rebound:.0%})")
    
    # Factor 5: Time pressure
    if days_until_departure <= 3:
        buy_score += 3
        reasoning.append("Time: Critical - departure imminent")
    elif days_until_departure <= 7:
        buy_score += 2
        reasoning.append("Time: Urgent - within 1 week")
    elif days_until_departure <= 14:
        buy_score += 1
        reasoning.append("Time: Approaching - within 2 weeks")
    
    # Factor 6: Risk profile adjustment
    if risk_profile == RiskProfile.CONSERVATIVE:
        buy_score += 1
        reasoning.append("Profile: Conservative - favor certainty")
    elif risk_profile == RiskProfile.AGGRESSIVE:
        wait_score += 1
        reasoning.append("Profile: Aggressive - willing to wait for deals")
    
    # Make decision based on scores
    score_diff = buy_score - wait_score
    
    if score_diff >= 3:
        decision = Decision.BUY
        confidence = min(0.9, 0.5 + score_diff * 0.08)
    elif score_diff <= -2:
        decision = Decision.WAIT
        confidence = min(0.9, 0.5 + abs(score_diff) * 0.08)
    else:
        decision = Decision.WATCH_CLOSELY
        confidence = 0.5
    
    return decision, round(confidence, 2), reasoning


# =============================================================================
# BATCH RECOMMENDATIONS
# =============================================================================

def get_recommendations_for_profile(
    origin: str,
    destination: str,
    target_month: str,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get decision recommendations for all combinations in a search profile.
    
    Returns sorted by recommendation priority.
    """
    combos = get_latest_combinations(origin, destination, target_month, offer_rank=1)
    
    recommendations = []
    
    for combo in combos[:50]:  # Limit to avoid too many calculations
        dep_date = combo['departure_date']
        ret_date = combo['return_date']
        
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        if isinstance(ret_date, str):
            ret_date = datetime.strptime(ret_date, "%Y-%m-%d").date()
        
        # Skip past departures
        if dep_date <= date.today():
            continue
        
        try:
            decision_result = make_decision(
                origin=origin,
                destination=destination,
                departure_date=dep_date,
                return_date=ret_date,
                current_price=combo['price'],
                risk_profile=risk_profile,
            )
            
            recommendations.append({
                'departure_date': str(dep_date),
                'return_date': str(ret_date),
                'price': combo['price'],
                'decision': decision_result.decision.value,
                'confidence': decision_result.confidence,
                'percentile': decision_result.percentile,
                'regime': decision_result.regime.value,
                'rebound_risk': decision_result.rebound_risk,
                'deadline': str(decision_result.deadline),
                'recommendation': decision_result.get_recommendation_text(),
            })
        except Exception as e:
            logger.warning(f"Failed to get decision for {dep_date}: {e}")
    
    # Sort: BUY first (prioritized), then by confidence
    decision_priority = {Decision.BUY.value: 0, Decision.WATCH_CLOSELY.value: 1, Decision.WAIT.value: 2}
    recommendations.sort(key=lambda x: (decision_priority.get(x['decision'], 2), -x['confidence']))
    
    return recommendations[:max_results]


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Decision Engine")
    parser.add_argument("origin", help="Origin airport code")
    parser.add_argument("dest", help="Destination airport code")
    parser.add_argument("departure", help="Departure date (YYYY-MM-DD)")
    parser.add_argument("return_date", help="Return date (YYYY-MM-DD)")
    parser.add_argument("--price", "-p", type=float, help="Current price (optional)")
    parser.add_argument("--risk", "-r", choices=['aggressive', 'moderate', 'conservative'],
                       default='moderate', help="Risk profile")
    
    args = parser.parse_args()
    
    dep = datetime.strptime(args.departure, "%Y-%m-%d").date()
    ret = datetime.strptime(args.return_date, "%Y-%m-%d").date()
    risk = RiskProfile(args.risk)
    
    result = make_decision(
        origin=args.origin,
        destination=args.dest,
        departure_date=dep,
        return_date=ret,
        current_price=args.price,
        risk_profile=risk,
    )
    
    print(f"\n{'='*60}")
    print(f"Flight: {args.origin} → {args.dest}")
    print(f"Dates: {dep} → {ret}")
    print(f"Current Price: ${result.current_price:.2f}")
    print(f"Fair Value: ${result.fair_value:.2f}")
    print(f"{'='*60}")
    print(f"\nDECISION: {result.decision.value.upper()}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Regime: {result.regime.value}")
    print(f"Percentile: P{result.percentile:.0f}")
    print(f"Rebound Risk: {result.rebound_risk:.0%}")
    print(f"Deadline: {result.deadline}")
    print(f"\n{result.get_recommendation_text()}")
    print(f"\nReasoning:")
    for r in result.reasoning:
        print(f"  • {r}")
