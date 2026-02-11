"""
Dynamic intraday scheduling for Flight Price Tracker.

Implements adaptive collection frequency based on lead time proximity
to capture price movements near decision windows.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from threading import Lock
import time

import schedule

from database import (
    get_search_profiles,
    get_latest_combinations,
    update_profile_last_searched,
    insert_flight_combinations_batch,
)
from data_fetcher import (
    fetch_top_k_offers,
    test_api_connection,
    DEFAULT_TOP_K,
)

# Configure logging
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False
_shutdown_lock = Lock()


# =============================================================================
# DYNAMIC FREQUENCY CALCULATION
# =============================================================================

def get_collection_frequency_hours(lead_time_days: int) -> int:
    """
    Returns hours between collections based on decision window proximity.
    
    More frequent collection as departure approaches:
    - 0-3 days: Every 2 hours (urgent)
    - 4-7 days: Every 4 hours
    - 8-14 days: Every 6 hours
    - 15-30 days: Every 12 hours (2x/day)
    - 31+ days: Every 24 hours (1x/day)
    
    Args:
        lead_time_days: Days until departure
        
    Returns:
        Hours between collections
    """
    if lead_time_days <= 3:
        return 2   # Every 2 hours (urgent)
    elif lead_time_days <= 7:
        return 4   # Every 4 hours
    elif lead_time_days <= 14:
        return 6   # Every 6 hours
    elif lead_time_days <= 30:
        return 12  # 2x/day
    else:
        return 24  # 1x/day


def get_priority_combos(profile: dict) -> List[Dict[str, Any]]:
    """
    Get flight combinations that need priority collection based on lead time.
    
    Returns combinations sorted by urgency (shortest lead time first).
    """
    origin = profile['origin']
    dest = profile['destination']
    target_month = profile.get('target_departure_month')
    
    # Get latest combinations
    combos = get_latest_combinations(origin, dest, target_month)
    
    if not combos:
        return []
    
    today = date.today()
    priority_combos = []
    
    for combo in combos:
        dep_date = combo['departure_date']
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        
        lead_time = (dep_date - today).days
        
        if lead_time < 0:
            continue  # Skip past departures
        
        freq_hours = get_collection_frequency_hours(lead_time)
        
        priority_combos.append({
            'departure_date': dep_date,
            'return_date': combo['return_date'],
            'lead_time_days': lead_time,
            'frequency_hours': freq_hours,
            'current_price': combo.get('price'),
            'urgency': 'urgent' if lead_time <= 7 else 'normal',
        })
    
    # Sort by lead time (most urgent first)
    priority_combos.sort(key=lambda x: x['lead_time_days'])
    
    return priority_combos


def should_collect_now(
    departure_date: date,
    last_collected: Optional[datetime] = None
) -> bool:
    """
    Determine if a combination should be collected now based on frequency rules.
    
    Args:
        departure_date: Flight departure date
        last_collected: When this combo was last collected
        
    Returns:
        True if collection is due
    """
    today = date.today()
    lead_time = (departure_date - today).days
    
    if lead_time < 0:
        return False  # Past departure
    
    if last_collected is None:
        return True  # Never collected
    
    freq_hours = get_collection_frequency_hours(lead_time)
    hours_since_last = (datetime.now() - last_collected).total_seconds() / 3600
    
    return hours_since_last >= freq_hours


# =============================================================================
# INTRADAY COLLECTION ENGINE
# =============================================================================

class IntradayCollector:
    """
    Manages dynamic intraday price collection.
    
    Tracks which combinations need collection and schedules accordingly.
    """
    
    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.top_k = top_k
        self.progress_callback = progress_callback
        self.last_collection_times: Dict[str, datetime] = {}
        self._lock = Lock()
    
    def _combo_key(self, origin: str, dest: str, dep: date, ret: date) -> str:
        """Generate unique key for a combination."""
        return f"{origin}-{dest}:{dep}:{ret}"
    
    def _log(self, message: str) -> None:
        """Log and optionally callback."""
        logger.info(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def get_due_combinations(
        self,
        profile: dict
    ) -> List[Dict[str, Any]]:
        """
        Get combinations that are due for collection.
        
        Returns combinations where enough time has passed since last collection
        based on their lead time priority.
        """
        origin = profile['origin']
        dest = profile['destination']
        target_month = profile.get('target_departure_month')
        
        combos = get_latest_combinations(origin, dest, target_month)
        due_combos = []
        today = date.today()
        
        for combo in combos:
            dep_date = combo['departure_date']
            ret_date = combo['return_date']
            
            if isinstance(dep_date, str):
                dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
            if isinstance(ret_date, str):
                ret_date = datetime.strptime(ret_date, "%Y-%m-%d").date()
            
            lead_time = (dep_date - today).days
            if lead_time < 0:
                continue
            
            key = self._combo_key(origin, dest, dep_date, ret_date)
            last_collected = self.last_collection_times.get(key)
            
            if should_collect_now(dep_date, last_collected):
                due_combos.append({
                    'origin': origin,
                    'destination': dest,
                    'departure_date': dep_date,
                    'return_date': ret_date,
                    'lead_time_days': lead_time,
                    'frequency_hours': get_collection_frequency_hours(lead_time),
                })
        
        return due_combos
    
    def collect_combination(
        self,
        origin: str,
        dest: str,
        departure_date: date,
        return_date: date,
    ) -> Dict[str, Any]:
        """
        Collect fresh prices for a single combination.
        
        Returns collection result with status.
        """
        scrape_timestamp = datetime.now()
        key = self._combo_key(origin, dest, departure_date, return_date)
        
        try:
            offers = fetch_top_k_offers(
                origin=origin,
                destination=dest,
                departure_date=departure_date,
                return_date=return_date,
                top_k=self.top_k,
                scrape_timestamp=scrape_timestamp,
            )
            
            if offers:
                # Save to database
                combos_to_save = [o.to_dict() for o in offers]
                saved_count = insert_flight_combinations_batch(combos_to_save)
                
                # Update last collection time
                with self._lock:
                    self.last_collection_times[key] = scrape_timestamp
                
                return {
                    'success': True,
                    'offers_count': len(offers),
                    'saved_count': saved_count,
                    'prices': [o.price for o in offers],
                }
            else:
                return {
                    'success': True,
                    'offers_count': 0,
                    'message': 'No offers found',
                }
                
        except Exception as e:
            logger.error(f"Failed to collect {origin}-{dest} {departure_date}: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def run_collection_cycle(
        self,
        profiles: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single collection cycle for all due combinations.
        
        Args:
            profiles: List of search profiles to check, or None for all active
            
        Returns:
            Summary of collection results
        """
        if profiles is None:
            profiles = get_search_profiles(active_only=True)
        
        if not profiles:
            self._log("No active profiles to collect")
            return {'profiles': 0, 'combinations': 0}
        
        self._log(f"Starting intraday collection for {len(profiles)} profiles")
        
        total_collected = 0
        total_offers = 0
        errors = []
        
        for profile in profiles:
            due_combos = self.get_due_combinations(profile)
            
            if not due_combos:
                continue
            
            self._log(f"  {profile['origin']}-{profile['destination']}: {len(due_combos)} combos due")
            
            for combo in due_combos:
                result = self.collect_combination(
                    combo['origin'],
                    combo['destination'],
                    combo['departure_date'],
                    combo['return_date'],
                )
                
                if result['success']:
                    total_collected += 1
                    total_offers += result.get('offers_count', 0)
                else:
                    errors.append({
                        'combo': f"{combo['origin']}-{combo['destination']} {combo['departure_date']}",
                        'error': result.get('error'),
                    })
            
            # Update profile last searched
            update_profile_last_searched(profile['id'])
        
        summary = {
            'profiles_checked': len(profiles),
            'combinations_collected': total_collected,
            'total_offers': total_offers,
            'errors': len(errors),
            'error_details': errors[:5],  # First 5 errors
            'timestamp': datetime.now().isoformat(),
        }
        
        self._log(f"Collection complete: {total_collected} combos, {total_offers} offers")
        
        return summary


# =============================================================================
# SCHEDULED COLLECTION
# =============================================================================

def run_intraday_scheduler(
    check_interval_minutes: int = 30,
    run_immediately: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Start the intraday collection scheduler.
    
    Checks every `check_interval_minutes` for combinations that need collection
    based on their dynamic frequency.
    
    Args:
        check_interval_minutes: How often to check for due collections
        run_immediately: Run a collection cycle immediately on start
        progress_callback: Optional callback for status updates
    """
    global _shutdown_requested
    
    with _shutdown_lock:
        _shutdown_requested = False
    
    collector = IntradayCollector(progress_callback=progress_callback)
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    log("=" * 60)
    log("Intraday Flight Price Collector")
    log(f"Check interval: every {check_interval_minutes} minutes")
    log("Dynamic frequency based on lead time:")
    log("  0-3 days: every 2 hours")
    log("  4-7 days: every 4 hours")
    log("  8-14 days: every 6 hours")
    log("  15-30 days: every 12 hours")
    log("  31+ days: every 24 hours")
    log("=" * 60)
    
    # Test API connection
    api_status = test_api_connection()
    if not api_status.get('connected'):
        log(f"WARNING: API connection failed: {api_status.get('message')}")
    else:
        log("API connection verified")
    
    # Run immediately if requested
    if run_immediately:
        log("Running initial collection cycle...")
        collector.run_collection_cycle()
    
    # Schedule periodic checks
    schedule.every(check_interval_minutes).minutes.do(
        collector.run_collection_cycle
    )
    
    # Main loop
    while True:
        with _shutdown_lock:
            if _shutdown_requested:
                break
        
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            log(f"Scheduler error: {e}")
            time.sleep(60)
    
    log("Intraday scheduler stopped")


def stop_intraday_scheduler() -> None:
    """Signal the scheduler to stop."""
    global _shutdown_requested
    with _shutdown_lock:
        _shutdown_requested = True


def get_collection_schedule(profile: dict) -> Dict[str, Any]:
    """
    Get the collection schedule for a profile's combinations.
    
    Shows when each combination will next be collected.
    """
    priority_combos = get_priority_combos(profile)
    
    schedule_info = {
        'profile': f"{profile['origin']}-{profile['destination']}",
        'total_combinations': len(priority_combos),
        'urgent_count': sum(1 for c in priority_combos if c['urgency'] == 'urgent'),
        'schedule': [],
    }
    
    for combo in priority_combos[:20]:  # Show first 20
        schedule_info['schedule'].append({
            'departure': str(combo['departure_date']),
            'lead_time_days': combo['lead_time_days'],
            'collection_frequency': f"every {combo['frequency_hours']} hours",
            'urgency': combo['urgency'],
            'current_price': combo.get('current_price'),
        })
    
    return schedule_info


def estimate_daily_api_calls(profiles: Optional[List[dict]] = None) -> Dict[str, Any]:
    """
    Estimate total API calls per day based on current profiles and lead times.
    
    Helps with API quota planning.
    """
    if profiles is None:
        profiles = get_search_profiles(active_only=True)
    
    total_calls_per_day = 0
    breakdown = []
    
    for profile in profiles:
        priority_combos = get_priority_combos(profile)
        
        profile_calls = 0
        for combo in priority_combos:
            freq_hours = combo['frequency_hours']
            calls_per_day = 24 / freq_hours
            profile_calls += calls_per_day
        
        total_calls_per_day += profile_calls
        
        breakdown.append({
            'profile': f"{profile['origin']}-{profile['destination']}",
            'combinations': len(priority_combos),
            'urgent': sum(1 for c in priority_combos if c['urgency'] == 'urgent'),
            'estimated_calls_per_day': round(profile_calls, 1),
        })
    
    return {
        'total_estimated_calls_per_day': round(total_calls_per_day, 1),
        'profiles': len(profiles),
        'breakdown': breakdown,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intraday Flight Price Collector")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Check interval in minutes (default: 30)"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run collection immediately on start"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate daily API calls and exit"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Show collection schedule and exit"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.estimate:
        estimate = estimate_daily_api_calls()
        print(f"\nEstimated API calls per day: {estimate['total_estimated_calls_per_day']}")
        print(f"Active profiles: {estimate['profiles']}")
        print("\nBreakdown:")
        for p in estimate['breakdown']:
            print(f"  {p['profile']}: {p['estimated_calls_per_day']} calls/day "
                  f"({p['combinations']} combos, {p['urgent']} urgent)")
    
    elif args.schedule:
        profiles = get_search_profiles(active_only=True)
        for profile in profiles:
            sched = get_collection_schedule(profile)
            print(f"\n{sched['profile']} ({sched['total_combinations']} combos, {sched['urgent_count']} urgent)")
            for item in sched['schedule'][:10]:
                print(f"  {item['departure']}: {item['collection_frequency']} ({item['urgency']})")
    
    else:
        run_intraday_scheduler(
            check_interval_minutes=args.interval,
            run_immediately=args.run_now or not args.run_now,  # Default to immediate
        )
