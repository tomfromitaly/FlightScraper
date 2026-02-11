"""
Outcome Tracker for Flight Price Tracker.

Tracks actual booking outcomes to enable:
- Feedback loop for model improvement
- Regret analysis
- Decision accuracy measurement
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from models import Decision, PriceRegime, BookingOutcome
from database import (
    insert_booking_outcome,
    update_booking_outcome,
    get_booking_outcomes,
    get_regret_statistics,
    get_price_history_for_combo,
    get_latest_combinations,
)
from quantile_model import get_fair_value_distribution

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# OUTCOME RECORDING
# =============================================================================

def record_decision(
    profile_id: int,
    departure_date: date,
    return_date: date,
    decision_made: Decision,
    price_at_decision: float,
    fair_value: Optional[float] = None,
    regime: Optional[PriceRegime] = None,
) -> Optional[int]:
    """
    Record a booking decision for future tracking.
    
    Call this when a user sees a recommendation but hasn't booked yet.
    
    Args:
        profile_id: Search profile ID
        departure_date: Flight departure date
        return_date: Flight return date
        decision_made: The decision that was recommended
        price_at_decision: Price when decision was made
        fair_value: Fair value (P50) at decision time
        regime: Pricing regime at decision time
        
    Returns:
        Outcome ID if successful
    """
    outcome_data = {
        'search_profile_id': profile_id,
        'departure_date': str(departure_date),
        'return_date': str(return_date),
        'decision_date': datetime.now().isoformat(),
        'decision_made': decision_made.value,
        'price_at_decision': price_at_decision,
        'fair_value_at_decision': fair_value,
        'regime_at_decision': regime.value if regime else None,
    }
    
    return insert_booking_outcome(outcome_data)


def record_booking(
    outcome_id: int,
    booked_price: float,
    booked_date: Optional[datetime] = None,
) -> bool:
    """
    Record that a booking was made.
    
    Call this when the user actually books a flight.
    
    Args:
        outcome_id: The outcome tracking ID
        booked_price: The price at which flight was booked
        booked_date: When the booking was made (defaults to now)
        
    Returns:
        True if successful
    """
    if booked_date is None:
        booked_date = datetime.now()
    
    updates = {
        'booked_date': booked_date.isoformat(),
        'booked_price': booked_price,
    }
    
    return update_booking_outcome(outcome_id, updates)


def finalize_outcome(
    outcome_id: int,
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
) -> Dict[str, Any]:
    """
    Finalize an outcome after the departure date has passed.
    
    Calculates:
    - Final lowest price observed before departure
    - Regret vs optimal
    - Whether decision was correct
    - Booked price percentile
    
    Call this after departure to complete the feedback loop.
    
    Args:
        outcome_id: The outcome tracking ID
        origin: Origin airport
        destination: Destination airport
        departure_date: Flight departure date
        return_date: Flight return date
        
    Returns:
        Dictionary with finalized metrics
    """
    # Get outcome record
    outcomes = get_booking_outcomes()
    outcome = next((o for o in outcomes if o['id'] == outcome_id), None)
    
    if not outcome:
        return {'error': 'Outcome not found'}
    
    # Get full price history for this combo
    history = get_price_history_for_combo(origin, destination, departure_date, return_date)
    
    if not history:
        return {'error': 'No price history available'}
    
    # Find lowest price observed
    all_prices = [h['price'] for h in history if h['price']]
    final_lowest_price = min(all_prices) if all_prices else None
    
    booked_price = outcome.get('booked_price')
    price_at_decision = outcome.get('price_at_decision')
    
    # Calculate regret (only if booked)
    regret = None
    decision_correct = None
    
    if booked_price is not None and final_lowest_price is not None:
        regret = booked_price - final_lowest_price
        
        # Determine if decision was correct
        decision_made = outcome.get('decision_made')
        
        if decision_made == Decision.BUY.value:
            # BUY was correct if booked price was near the lowest
            # Allow 10% tolerance for "near optimal"
            decision_correct = regret <= (final_lowest_price * 0.10)
        
        elif decision_made == Decision.WAIT.value:
            # WAIT was correct if prices dropped after decision
            if booked_price < price_at_decision:
                decision_correct = True
            else:
                decision_correct = False
    
    # Calculate booked price percentile
    percentile = None
    if booked_price is not None:
        fair_value = get_fair_value_distribution(
            origin, destination, departure_date, return_date
        )
        if fair_value:
            percentile = fair_value.get_percentile(booked_price)
    
    # Update outcome record
    updates = {
        'final_lowest_price': final_lowest_price,
        'regret_vs_optimal': regret,
        'decision_was_correct': decision_correct,
        'booked_price_percentile': percentile,
    }
    
    update_booking_outcome(outcome_id, updates)
    
    return {
        'outcome_id': outcome_id,
        'final_lowest_price': final_lowest_price,
        'booked_price': booked_price,
        'regret_vs_optimal': regret,
        'decision_was_correct': decision_correct,
        'booked_price_percentile': percentile,
        'price_history_points': len(all_prices),
    }


# =============================================================================
# OUTCOME ANALYSIS
# =============================================================================

def get_decision_accuracy(profile_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Calculate decision accuracy across all tracked outcomes.
    
    Args:
        profile_id: Optional filter by profile
        
    Returns:
        Dictionary with accuracy metrics
    """
    outcomes = get_booking_outcomes(profile_id)
    
    if not outcomes:
        return {
            'total_outcomes': 0,
            'message': 'No outcomes tracked yet',
        }
    
    # Filter to completed outcomes (have decision_was_correct)
    completed = [o for o in outcomes if o.get('decision_was_correct') is not None]
    
    if not completed:
        return {
            'total_outcomes': len(outcomes),
            'completed_outcomes': 0,
            'message': 'No finalized outcomes yet',
        }
    
    # Calculate accuracy by decision type
    buy_outcomes = [o for o in completed if o['decision_made'] == Decision.BUY.value]
    wait_outcomes = [o for o in completed if o['decision_made'] == Decision.WAIT.value]
    watch_outcomes = [o for o in completed if o['decision_made'] == Decision.WATCH_CLOSELY.value]
    
    buy_correct = sum(1 for o in buy_outcomes if o['decision_was_correct'])
    wait_correct = sum(1 for o in wait_outcomes if o['decision_was_correct'])
    total_correct = sum(1 for o in completed if o['decision_was_correct'])
    
    return {
        'total_outcomes': len(outcomes),
        'completed_outcomes': len(completed),
        'overall_accuracy': round(total_correct / len(completed) * 100, 1) if completed else 0,
        'buy_decisions': {
            'count': len(buy_outcomes),
            'correct': buy_correct,
            'accuracy': round(buy_correct / len(buy_outcomes) * 100, 1) if buy_outcomes else 0,
        },
        'wait_decisions': {
            'count': len(wait_outcomes),
            'correct': wait_correct,
            'accuracy': round(wait_correct / len(wait_outcomes) * 100, 1) if wait_outcomes else 0,
        },
        'watch_decisions': {
            'count': len(watch_outcomes),
        },
    }


def get_regret_analysis(profile_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze regret across tracked outcomes.
    
    Args:
        profile_id: Optional filter by profile
        
    Returns:
        Dictionary with regret metrics
    """
    stats = get_regret_statistics(profile_id)
    
    if stats['total_decisions'] == 0:
        return {
            'total_decisions': 0,
            'message': 'No finalized outcomes with regret data',
        }
    
    return {
        'total_decisions': stats['total_decisions'],
        'correct_decisions': stats['correct_decisions'],
        'accuracy_rate': round(stats['correct_decisions'] / stats['total_decisions'] * 100, 1),
        'total_regret': round(stats['total_regret'], 2),
        'avg_regret_per_decision': round(stats['avg_regret'], 2),
        'best_outcome': round(stats['best_outcome'], 2),  # Best case (lowest regret)
        'worst_outcome': round(stats['worst_outcome'], 2),  # Worst case (highest regret)
    }


def get_percentile_distribution(profile_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze distribution of booked prices across percentiles.
    
    Shows how well users are timing their purchases.
    
    Args:
        profile_id: Optional filter by profile
        
    Returns:
        Dictionary with percentile distribution
    """
    outcomes = get_booking_outcomes(profile_id)
    
    # Filter to outcomes with percentile data
    with_percentile = [o for o in outcomes if o.get('booked_price_percentile') is not None]
    
    if not with_percentile:
        return {
            'total_bookings': 0,
            'message': 'No percentile data available',
        }
    
    percentiles = [o['booked_price_percentile'] for o in with_percentile]
    
    # Categorize
    great_deals = sum(1 for p in percentiles if p <= 10)
    good_deals = sum(1 for p in percentiles if 10 < p <= 25)
    fair_prices = sum(1 for p in percentiles if 25 < p <= 50)
    above_avg = sum(1 for p in percentiles if 50 < p <= 75)
    expensive = sum(1 for p in percentiles if p > 75)
    
    return {
        'total_bookings': len(percentiles),
        'avg_percentile': round(sum(percentiles) / len(percentiles), 1),
        'distribution': {
            'great_deals_p10': great_deals,
            'good_deals_p25': good_deals,
            'fair_prices_p50': fair_prices,
            'above_average_p75': above_avg,
            'expensive_p90plus': expensive,
        },
        'distribution_pct': {
            'great_deals': round(great_deals / len(percentiles) * 100, 1),
            'good_deals': round(good_deals / len(percentiles) * 100, 1),
            'fair_prices': round(fair_prices / len(percentiles) * 100, 1),
            'above_average': round(above_avg / len(percentiles) * 100, 1),
            'expensive': round(expensive / len(percentiles) * 100, 1),
        },
    }


# =============================================================================
# AUTOMATIC FINALIZATION
# =============================================================================

def finalize_past_outcomes(
    origin: str,
    destination: str,
) -> Dict[str, Any]:
    """
    Automatically finalize all past outcomes for a route.
    
    Call periodically to update outcome tracking.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        
    Returns:
        Summary of finalized outcomes
    """
    outcomes = get_booking_outcomes()
    today = date.today()
    finalized = 0
    errors = []
    
    for outcome in outcomes:
        # Check if already finalized
        if outcome.get('final_lowest_price') is not None:
            continue
        
        dep_date = outcome.get('departure_date')
        ret_date = outcome.get('return_date')
        
        if not dep_date or not ret_date:
            continue
        
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        if isinstance(ret_date, str):
            ret_date = datetime.strptime(ret_date, "%Y-%m-%d").date()
        
        # Only finalize if departure has passed
        if dep_date > today:
            continue
        
        try:
            result = finalize_outcome(
                outcome['id'],
                origin,
                destination,
                dep_date,
                ret_date,
            )
            
            if 'error' not in result:
                finalized += 1
            else:
                errors.append({
                    'outcome_id': outcome['id'],
                    'error': result['error'],
                })
        except Exception as e:
            errors.append({
                'outcome_id': outcome['id'],
                'error': str(e),
            })
    
    return {
        'finalized_count': finalized,
        'errors': errors,
    }


# =============================================================================
# FEEDBACK SUMMARY
# =============================================================================

def get_feedback_summary(profile_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get comprehensive feedback summary combining all metrics.
    
    Args:
        profile_id: Optional filter by profile
        
    Returns:
        Complete feedback summary
    """
    accuracy = get_decision_accuracy(profile_id)
    regret = get_regret_analysis(profile_id)
    percentiles = get_percentile_distribution(profile_id)
    
    # Calculate overall score (0-100)
    score = 0
    score_factors = []
    
    # Factor 1: Decision accuracy (0-40 points)
    if accuracy.get('overall_accuracy'):
        accuracy_points = accuracy['overall_accuracy'] * 0.4
        score += accuracy_points
        score_factors.append(f"Accuracy: {accuracy['overall_accuracy']}% (+{accuracy_points:.1f})")
    
    # Factor 2: Average percentile (0-30 points, lower is better)
    if percentiles.get('avg_percentile'):
        # 100 percentile = 0 points, 0 percentile = 30 points
        percentile_points = (100 - percentiles['avg_percentile']) * 0.3
        score += percentile_points
        score_factors.append(f"Avg percentile: P{percentiles['avg_percentile']} (+{percentile_points:.1f})")
    
    # Factor 3: Low regret (0-30 points)
    if regret.get('avg_regret_per_decision') is not None:
        # Normalize: $0 regret = 30 points, $100+ regret = 0 points
        regret_normalized = max(0, 100 - regret['avg_regret_per_decision']) / 100
        regret_points = regret_normalized * 30
        score += regret_points
        score_factors.append(f"Avg regret: ${regret['avg_regret_per_decision']:.0f} (+{regret_points:.1f})")
    
    # Grade the score
    if score >= 80:
        grade = "A"
        assessment = "Excellent timing - consistently finding great deals"
    elif score >= 60:
        grade = "B"
        assessment = "Good timing - booking at fair prices"
    elif score >= 40:
        grade = "C"
        assessment = "Average timing - room for improvement"
    else:
        grade = "D"
        assessment = "Below average - consider following recommendations more closely"
    
    return {
        'score': round(score, 1),
        'grade': grade,
        'assessment': assessment,
        'score_factors': score_factors,
        'accuracy': accuracy,
        'regret': regret,
        'percentile_distribution': percentiles,
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
    
    parser = argparse.ArgumentParser(description="Outcome Tracker")
    parser.add_argument("--summary", action="store_true", help="Show feedback summary")
    parser.add_argument("--accuracy", action="store_true", help="Show decision accuracy")
    parser.add_argument("--regret", action="store_true", help="Show regret analysis")
    parser.add_argument("--profile", "-p", type=int, help="Filter by profile ID")
    
    args = parser.parse_args()
    
    if args.summary:
        summary = get_feedback_summary(args.profile)
        print(f"\nFeedback Summary")
        print(f"================")
        print(f"Score: {summary['score']}/100 (Grade: {summary['grade']})")
        print(f"Assessment: {summary['assessment']}")
        print(f"\nScore Factors:")
        for f in summary['score_factors']:
            print(f"  • {f}")
    
    elif args.accuracy:
        acc = get_decision_accuracy(args.profile)
        print(f"\nDecision Accuracy")
        print(f"=================")
        print(f"Overall: {acc.get('overall_accuracy', 0)}%")
        if 'buy_decisions' in acc:
            print(f"BUY decisions: {acc['buy_decisions']['accuracy']}% ({acc['buy_decisions']['correct']}/{acc['buy_decisions']['count']})")
        if 'wait_decisions' in acc:
            print(f"WAIT decisions: {acc['wait_decisions']['accuracy']}% ({acc['wait_decisions']['correct']}/{acc['wait_decisions']['count']})")
    
    elif args.regret:
        reg = get_regret_analysis(args.profile)
        print(f"\nRegret Analysis")
        print(f"===============")
        print(f"Average regret: ${reg.get('avg_regret_per_decision', 0):.2f}")
        print(f"Total regret: ${reg.get('total_regret', 0):.2f}")
        print(f"Best outcome: ${reg.get('best_outcome', 0):.2f}")
        print(f"Worst outcome: ${reg.get('worst_outcome', 0):.2f}")
    
    else:
        parser.print_help()
