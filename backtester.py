"""
Backtesting Engine for Flight Price Tracker.

Simulates historical buy/wait decisions and reports regret vs ex-post best.
Used to validate and tune decision strategies.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

from models import (
    Decision,
    BacktestResult,
    FairValueDistribution,
    PriceRegime,
    RiskProfile,
)
from database import (
    get_price_history_for_combo,
    get_combinations_for_route,
    get_db_connection,
)
from baseline_builder import get_best_baseline
from quantile_model import detect_regime

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# BACKTEST DATA PREPARATION
# =============================================================================

@dataclass
class PricePoint:
    """A single price observation in backtest data."""
    timestamp: datetime
    price: float
    lead_time_days: int
    available_seats: Optional[int] = None


def get_backtest_data(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
) -> List[PricePoint]:
    """
    Get historical price observations for backtesting.
    
    Returns chronologically sorted price points.
    """
    history = get_price_history_for_combo(origin, destination, departure_date, return_date)
    
    if not history:
        return []
    
    points = []
    for h in history:
        timestamp = h.get('scrape_timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        elif timestamp is None:
            # Fall back to scrape_date if timestamp not available
            scrape_date = h.get('scrape_date')
            if isinstance(scrape_date, str):
                timestamp = datetime.strptime(scrape_date, "%Y-%m-%d")
            else:
                continue
        
        points.append(PricePoint(
            timestamp=timestamp,
            price=h['price'],
            lead_time_days=h.get('lead_time_days', 0),
            available_seats=h.get('available_seats'),
        ))
    
    # Sort by timestamp
    points.sort(key=lambda p: p.timestamp)
    
    return points


def get_all_backtestable_combos(
    origin: str,
    destination: str,
    min_observations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get all date combinations with sufficient history for backtesting.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        min_observations: Minimum price observations required
        
    Returns:
        List of (departure_date, return_date) combinations
    """
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT departure_date, return_date, COUNT(*) as obs_count,
                   MIN(price) as min_price, MAX(price) as max_price
            FROM flight_combinations
            WHERE departure_airport = ? AND arrival_airport = ?
            AND offer_rank = 1
            AND departure_date < date('now')
            GROUP BY departure_date, return_date
            HAVING COUNT(*) >= ?
            ORDER BY departure_date
        """, (origin.upper(), destination.upper(), min_observations))
        
        return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

def strategy_always_buy_immediately(
    price_points: List[PricePoint],
    fair_value: Optional[FairValueDistribution],
    risk_profile: RiskProfile,
) -> Optional[PricePoint]:
    """
    Naive strategy: always buy at first observation.
    Baseline for comparison.
    """
    if not price_points:
        return None
    return price_points[0]


def strategy_buy_below_p25(
    price_points: List[PricePoint],
    fair_value: Optional[FairValueDistribution],
    risk_profile: RiskProfile,
) -> Optional[PricePoint]:
    """
    Buy when price drops below P25.
    Falls back to last observation if never triggered.
    """
    if not price_points or not fair_value:
        return price_points[0] if price_points else None
    
    for point in price_points:
        if point.price <= fair_value.p25:
            return point
    
    # Never found a deal - buy at last observation
    return price_points[-1]


def strategy_buy_below_p50(
    price_points: List[PricePoint],
    fair_value: Optional[FairValueDistribution],
    risk_profile: RiskProfile,
) -> Optional[PricePoint]:
    """
    Buy when price drops below P50 (median).
    """
    if not price_points or not fair_value:
        return price_points[0] if price_points else None
    
    for point in price_points:
        if point.price <= fair_value.p50:
            return point
    
    return price_points[-1]


def strategy_wait_for_optimal_window(
    price_points: List[PricePoint],
    fair_value: Optional[FairValueDistribution],
    risk_profile: RiskProfile,
) -> Optional[PricePoint]:
    """
    Wait until optimal booking window (21-45 days) then buy lowest.
    """
    if not price_points:
        return None
    
    # Filter to optimal window
    in_window = [p for p in price_points if 21 <= p.lead_time_days <= 45]
    
    if in_window:
        # Buy the lowest in window
        return min(in_window, key=lambda p: p.price)
    else:
        # Fall back to last observation
        return price_points[-1]


def strategy_decision_engine_simulation(
    price_points: List[PricePoint],
    fair_value: Optional[FairValueDistribution],
    risk_profile: RiskProfile,
) -> Optional[PricePoint]:
    """
    Simulate the actual decision engine logic.
    """
    if not price_points or not fair_value:
        return price_points[0] if price_points else None
    
    # Track recent prices for volatility calculation
    recent_prices = []
    
    for i, point in enumerate(price_points):
        recent_prices.append(point.price)
        if len(recent_prices) > 10:
            recent_prices.pop(0)
        
        # Detect regime
        regime = detect_regime(point.price, fair_value, None, None)
        
        # Calculate z-score
        z_score = fair_value.calculate_z_score(point.price)
        percentile = fair_value.get_percentile(point.price)
        
        # Decision logic (simplified version of decision_engine)
        should_buy = False
        
        # Promo = buy immediately
        if regime == PriceRegime.PROMO:
            should_buy = True
        
        # Great deal = buy
        elif percentile <= 15:
            should_buy = True
        
        # Scarcity with short lead time = buy
        elif regime == PriceRegime.SCARCITY and point.lead_time_days <= 14:
            should_buy = True
        
        # Good deal with moderate lead time = buy
        elif percentile <= 25 and point.lead_time_days <= 30:
            should_buy = True
        
        # Getting close to departure with fair price = buy
        elif percentile <= 50 and point.lead_time_days <= 7:
            should_buy = True
        
        # Last observation = must buy
        elif i == len(price_points) - 1:
            should_buy = True
        
        # Risk profile adjustments
        if risk_profile == RiskProfile.CONSERVATIVE:
            # More willing to buy early
            if percentile <= 50 and point.lead_time_days <= 21:
                should_buy = True
        elif risk_profile == RiskProfile.AGGRESSIVE:
            # Hold out longer for better deals
            if percentile > 20 and point.lead_time_days > 14:
                should_buy = False
        
        if should_buy:
            return point
    
    # Should never reach here, but safety fallback
    return price_points[-1]


# Strategy registry
STRATEGIES = {
    'always_buy': strategy_always_buy_immediately,
    'buy_below_p25': strategy_buy_below_p25,
    'buy_below_p50': strategy_buy_below_p50,
    'optimal_window': strategy_wait_for_optimal_window,
    'decision_engine': strategy_decision_engine_simulation,
}


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest_single_combo(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    strategy_name: str,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
) -> Dict[str, Any]:
    """
    Run backtest for a single date combination.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        departure_date: Flight departure date
        return_date: Flight return date
        strategy_name: Name of strategy to test
        risk_profile: Risk profile to use
        
    Returns:
        Dictionary with backtest results
    """
    # Get price history
    price_points = get_backtest_data(origin, destination, departure_date, return_date)
    
    if not price_points:
        return {
            'error': 'No price history',
            'departure_date': str(departure_date),
        }
    
    # Get fair value baseline
    # Use the first observation's lead time for baseline lookup
    first_lead_time = price_points[0].lead_time_days
    fair_value = get_best_baseline(
        origin, destination,
        month=departure_date.month,
        lead_time_days=first_lead_time,
    )
    
    # Get strategy function
    strategy_func = STRATEGIES.get(strategy_name, strategy_always_buy_immediately)
    
    # Run strategy
    buy_point = strategy_func(price_points, fair_value, risk_profile)
    
    if not buy_point:
        return {
            'error': 'Strategy returned no buy point',
            'departure_date': str(departure_date),
        }
    
    # Calculate results
    all_prices = [p.price for p in price_points]
    optimal_price = min(all_prices)
    worst_price = max(all_prices)
    avg_price = sum(all_prices) / len(all_prices)
    
    regret = buy_point.price - optimal_price
    regret_pct = (regret / optimal_price * 100) if optimal_price > 0 else 0
    
    savings_vs_worst = worst_price - buy_point.price
    savings_vs_avg = avg_price - buy_point.price
    
    # Determine if decision was optimal (within 5% of best)
    was_optimal = regret <= (optimal_price * 0.05)
    
    return {
        'departure_date': str(departure_date),
        'return_date': str(return_date),
        'strategy': strategy_name,
        'buy_price': buy_point.price,
        'buy_timestamp': buy_point.timestamp.isoformat(),
        'buy_lead_time': buy_point.lead_time_days,
        'optimal_price': optimal_price,
        'worst_price': worst_price,
        'avg_price': round(avg_price, 2),
        'regret': round(regret, 2),
        'regret_pct': round(regret_pct, 1),
        'savings_vs_worst': round(savings_vs_worst, 2),
        'savings_vs_avg': round(savings_vs_avg, 2),
        'was_optimal': was_optimal,
        'price_observations': len(price_points),
    }


def backtest_strategy(
    origin: str,
    destination: str,
    strategy_name: str,
    date_range_start: Optional[date] = None,
    date_range_end: Optional[date] = None,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    min_observations: int = 5,
) -> BacktestResult:
    """
    Run backtest for a strategy across all historical data.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        strategy_name: Name of strategy to test
        date_range_start: Start of date range (departure dates)
        date_range_end: End of date range
        risk_profile: Risk profile to use
        min_observations: Minimum observations per combo
        
    Returns:
        BacktestResult with aggregate metrics
    """
    # Get all testable combinations
    combos = get_all_backtestable_combos(origin, destination, min_observations)
    
    if not combos:
        return BacktestResult(
            strategy_name=strategy_name,
            route=f"{origin}-{destination}",
            date_range_start=date_range_start or date.today(),
            date_range_end=date_range_end or date.today(),
            total_decisions=0,
        )
    
    # Filter by date range if specified
    if date_range_start or date_range_end:
        filtered = []
        for c in combos:
            dep = c['departure_date']
            if isinstance(dep, str):
                dep = datetime.strptime(dep, "%Y-%m-%d").date()
            
            if date_range_start and dep < date_range_start:
                continue
            if date_range_end and dep > date_range_end:
                continue
            filtered.append(c)
        combos = filtered
    
    if not combos:
        return BacktestResult(
            strategy_name=strategy_name,
            route=f"{origin}-{destination}",
            date_range_start=date_range_start or date.today(),
            date_range_end=date_range_end or date.today(),
            total_decisions=0,
        )
    
    # Run backtest for each combination
    results = []
    for combo in combos:
        dep = combo['departure_date']
        ret = combo['return_date']
        
        if isinstance(dep, str):
            dep = datetime.strptime(dep, "%Y-%m-%d").date()
        if isinstance(ret, str):
            ret = datetime.strptime(ret, "%Y-%m-%d").date()
        
        result = backtest_single_combo(
            origin, destination, dep, ret,
            strategy_name, risk_profile
        )
        
        if 'error' not in result:
            results.append(result)
    
    if not results:
        return BacktestResult(
            strategy_name=strategy_name,
            route=f"{origin}-{destination}",
            date_range_start=date_range_start or date.today(),
            date_range_end=date_range_end or date.today(),
            total_decisions=0,
        )
    
    # Aggregate results
    total_regret = sum(r['regret'] for r in results)
    avg_regret = total_regret / len(results)
    correct_decisions = sum(1 for r in results if r['was_optimal'])
    
    # Calculate savings vs naive (always buy immediately)
    naive_results = []
    for combo in combos:
        dep = combo['departure_date']
        ret = combo['return_date']
        if isinstance(dep, str):
            dep = datetime.strptime(dep, "%Y-%m-%d").date()
        if isinstance(ret, str):
            ret = datetime.strptime(ret, "%Y-%m-%d").date()
        
        naive = backtest_single_combo(
            origin, destination, dep, ret,
            'always_buy', risk_profile
        )
        if 'error' not in naive:
            naive_results.append(naive)
    
    total_savings = 0
    if naive_results:
        naive_total = sum(r['buy_price'] for r in naive_results)
        strategy_total = sum(r['buy_price'] for r in results)
        total_savings = naive_total - strategy_total
    
    # Get date range from results
    all_deps = [datetime.strptime(r['departure_date'], "%Y-%m-%d").date() for r in results]
    actual_start = min(all_deps)
    actual_end = max(all_deps)
    
    return BacktestResult(
        strategy_name=strategy_name,
        route=f"{origin}-{destination}",
        date_range_start=actual_start,
        date_range_end=actual_end,
        total_decisions=len(results),
        buy_decisions=len(results),
        correct_buy_decisions=correct_decisions,
        accuracy=round(correct_decisions / len(results) * 100, 1),
        total_regret=round(total_regret, 2),
        avg_regret_per_decision=round(avg_regret, 2),
        total_savings_vs_naive=round(total_savings, 2),
        avg_savings_per_booking=round(total_savings / len(results), 2) if results else 0,
    )


def compare_strategies(
    origin: str,
    destination: str,
    strategies: Optional[List[str]] = None,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
) -> Dict[str, Any]:
    """
    Compare multiple strategies against each other.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        strategies: List of strategy names to compare (all if None)
        risk_profile: Risk profile to use
        
    Returns:
        Comparison results with rankings
    """
    if strategies is None:
        strategies = list(STRATEGIES.keys())
    
    results = {}
    for strategy_name in strategies:
        try:
            result = backtest_strategy(
                origin, destination, strategy_name,
                risk_profile=risk_profile
            )
            results[strategy_name] = result
        except Exception as e:
            logger.error(f"Error backtesting {strategy_name}: {e}")
            results[strategy_name] = None
    
    # Filter successful results
    valid_results = {k: v for k, v in results.items() if v and v.total_decisions > 0}
    
    if not valid_results:
        return {
            'error': 'No valid backtest results',
            'strategies_tested': strategies,
        }
    
    # Rank by average regret (lower is better)
    ranked_by_regret = sorted(
        valid_results.items(),
        key=lambda x: x[1].avg_regret_per_decision
    )
    
    # Rank by accuracy (higher is better)
    ranked_by_accuracy = sorted(
        valid_results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True
    )
    
    # Find best strategy
    best_strategy = ranked_by_regret[0][0]
    best_result = ranked_by_regret[0][1]
    
    return {
        'route': f"{origin}-{destination}",
        'strategies_tested': strategies,
        'best_strategy': best_strategy,
        'ranking_by_regret': [
            {
                'rank': i + 1,
                'strategy': name,
                'avg_regret': result.avg_regret_per_decision,
                'total_regret': result.total_regret,
                'accuracy': result.accuracy,
                'decisions': result.total_decisions,
            }
            for i, (name, result) in enumerate(ranked_by_regret)
        ],
        'ranking_by_accuracy': [
            {
                'rank': i + 1,
                'strategy': name,
                'accuracy': result.accuracy,
                'avg_regret': result.avg_regret_per_decision,
            }
            for i, (name, result) in enumerate(ranked_by_accuracy)
        ],
        'summary': {
            'best_strategy': best_strategy,
            'best_avg_regret': best_result.avg_regret_per_decision,
            'best_accuracy': best_result.accuracy,
            'total_decisions_tested': best_result.total_decisions,
        },
    }


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def optimize_percentile_threshold(
    origin: str,
    destination: str,
    threshold_range: List[int] = [10, 15, 20, 25, 30, 35, 40, 50],
    risk_profile: RiskProfile = RiskProfile.MODERATE,
) -> Dict[str, Any]:
    """
    Find optimal percentile threshold for buying.
    
    Tests different "buy below P{X}" thresholds to find best performing.
    
    Args:
        origin: Origin airport
        destination: Destination airport
        threshold_range: List of percentile thresholds to test
        risk_profile: Risk profile to use
        
    Returns:
        Optimization results with best threshold
    """
    # Get testable combos
    combos = get_all_backtestable_combos(origin, destination, min_observations=5)
    
    if not combos:
        return {'error': 'No testable data'}
    
    results = []
    
    for threshold in threshold_range:
        # Create custom strategy for this threshold
        def make_threshold_strategy(thresh):
            def strategy(price_points, fair_value, risk_profile):
                if not price_points or not fair_value:
                    return price_points[0] if price_points else None
                
                for point in price_points:
                    percentile = fair_value.get_percentile(point.price)
                    if percentile <= thresh:
                        return point
                return price_points[-1]
            return strategy
        
        # Temporarily add strategy
        strategy_name = f'buy_below_p{threshold}'
        STRATEGIES[strategy_name] = make_threshold_strategy(threshold)
        
        # Run backtest
        try:
            backtest_result = backtest_strategy(
                origin, destination, strategy_name,
                risk_profile=risk_profile
            )
            
            if backtest_result.total_decisions > 0:
                results.append({
                    'threshold': threshold,
                    'avg_regret': backtest_result.avg_regret_per_decision,
                    'total_regret': backtest_result.total_regret,
                    'accuracy': backtest_result.accuracy,
                    'decisions': backtest_result.total_decisions,
                })
        finally:
            # Clean up temporary strategy
            del STRATEGIES[strategy_name]
    
    if not results:
        return {'error': 'No valid results'}
    
    # Find optimal threshold (minimize regret)
    best_by_regret = min(results, key=lambda x: x['avg_regret'])
    best_by_accuracy = max(results, key=lambda x: x['accuracy'])
    
    return {
        'route': f"{origin}-{destination}",
        'threshold_results': results,
        'optimal_by_regret': {
            'threshold': best_by_regret['threshold'],
            'avg_regret': best_by_regret['avg_regret'],
            'accuracy': best_by_regret['accuracy'],
        },
        'optimal_by_accuracy': {
            'threshold': best_by_accuracy['threshold'],
            'accuracy': best_by_accuracy['accuracy'],
            'avg_regret': best_by_accuracy['avg_regret'],
        },
        'recommendation': f"Use P{best_by_regret['threshold']} threshold for this route",
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
    
    parser = argparse.ArgumentParser(description="Backtesting Engine")
    parser.add_argument("origin", help="Origin airport code")
    parser.add_argument("dest", help="Destination airport code")
    parser.add_argument("--strategy", "-s", default="decision_engine",
                       help="Strategy to test")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare all strategies")
    parser.add_argument("--optimize", "-o", action="store_true",
                       help="Optimize percentile threshold")
    parser.add_argument("--risk", "-r", choices=['aggressive', 'moderate', 'conservative'],
                       default='moderate', help="Risk profile")
    
    args = parser.parse_args()
    risk = RiskProfile(args.risk)
    
    if args.compare:
        print(f"\nComparing strategies for {args.origin} → {args.dest}")
        print("=" * 60)
        
        comparison = compare_strategies(args.origin, args.dest, risk_profile=risk)
        
        if 'error' in comparison:
            print(f"Error: {comparison['error']}")
        else:
            print(f"\nBest strategy: {comparison['best_strategy']}")
            print(f"\nRanking by Regret:")
            for r in comparison['ranking_by_regret']:
                print(f"  {r['rank']}. {r['strategy']}: ${r['avg_regret']:.2f} avg regret, "
                      f"{r['accuracy']:.1f}% accuracy")
    
    elif args.optimize:
        print(f"\nOptimizing threshold for {args.origin} → {args.dest}")
        print("=" * 60)
        
        result = optimize_percentile_threshold(args.origin, args.dest, risk_profile=risk)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nThreshold Results:")
            for r in result['threshold_results']:
                print(f"  P{r['threshold']}: ${r['avg_regret']:.2f} regret, {r['accuracy']:.1f}% accuracy")
            print(f"\nRecommendation: {result['recommendation']}")
    
    else:
        print(f"\nBacktesting {args.strategy} for {args.origin} → {args.dest}")
        print("=" * 60)
        
        result = backtest_strategy(
            args.origin, args.dest, args.strategy,
            risk_profile=risk
        )
        
        print(f"\nResults:")
        print(f"  Decisions tested: {result.total_decisions}")
        print(f"  Accuracy (within 5% of optimal): {result.accuracy}%")
        print(f"  Average regret: ${result.avg_regret_per_decision:.2f}")
        print(f"  Total regret: ${result.total_regret:.2f}")
        print(f"  Savings vs naive: ${result.total_savings_vs_naive:.2f}")
