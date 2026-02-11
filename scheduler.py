"""
Automated daily collection scheduler for Flight Price Tracker.

Handles scheduled and on-demand price collection with graceful shutdown.
Updated to work with search_profiles and comprehensive calendar searches.
"""

import logging
import signal
import sys
import time
from datetime import datetime, date
from typing import Optional, Callable, Dict, Any, List

import schedule

from config import DEFAULT_SCHEDULE_TIME
from database import (
    get_search_profiles,
    update_profile_last_searched,
    insert_flight_combinations_batch,
    get_latest_combinations,
    insert_price_alert,
    get_profile_by_id,
)
from data_fetcher import fetch_comprehensive_calendar, test_api_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger.info("Shutdown signal received. Finishing current task...")
    _shutdown_requested = True


def run_daily_collection(
    profile_ids: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive calendar collection for all active tracking profiles.
    
    Args:
        profile_ids: Optional list of specific profile IDs to collect.
                    If None, collects all active profiles.
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with collection statistics
    """
    global _shutdown_requested
    
    logger.info("=" * 60)
    logger.info("Starting daily price collection (calendar mode)")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Test API connection first
    api_status = test_api_connection()
    if not api_status.get("connected"):
        logger.error(f"API connection failed: {api_status.get('message')}")
        return {
            "success": False,
            "error": "API connection failed",
            "message": api_status.get('message'),
        }
    
    # Get profiles to collect
    if profile_ids:
        profiles = [get_profile_by_id(pid) for pid in profile_ids]
        profiles = [p for p in profiles if p]  # Filter None
    else:
        profiles = get_search_profiles(active_only=True)
    
    if not profiles:
        logger.warning("No active search profiles found")
        return {
            "success": True,
            "profiles_processed": 0,
            "total_combinations": 0,
            "message": "No profiles to collect",
        }
    
    # Collection statistics
    stats = {
        "success": True,
        "profiles_processed": 0,
        "profiles_failed": 0,
        "total_combinations": 0,
        "start_time": datetime.now().isoformat(),
        "profiles_detail": [],
        "alerts_generated": 0,
    }
    
    total_profiles = len(profiles)
    
    for idx, profile in enumerate(profiles):
        if _shutdown_requested:
            logger.info("Shutdown requested, stopping collection")
            break
        
        origin = profile['origin']
        destination = profile['destination']
        target_month = profile.get('target_departure_month') or date.today().strftime('%Y-%m')
        trip_duration = profile.get('trip_duration_days', 21)
        flexibility = profile.get('flexibility_days', 2)
        profile_id = profile['id']
        profile_name = profile.get('profile_name') or f"{origin}-{destination}"
        
        logger.info(f"Collecting [{idx+1}/{total_profiles}]: {profile_name} ({target_month})")
        
        # Update external progress
        if progress_callback:
            pct = int((idx / total_profiles) * 100)
            progress_callback(pct, f"Collecting {origin}-{destination}...")
        
        try:
            # Get previous prices for comparison
            previous_combos = get_latest_combinations(origin, destination, target_month)
            previous_prices = {
                (c['departure_date'], c['return_date']): c['price']
                for c in previous_combos
            }
            
            # Fetch comprehensive calendar
            results = fetch_comprehensive_calendar(
                origin=origin,
                destination=destination,
                target_month=target_month,
                trip_duration_days=trip_duration,
                flexibility_days=flexibility,
                progress_callback=None,  # Internal progress
            )
            
            if results['combinations']:
                # Convert to dicts for batch insert
                combos_to_save = [c.to_dict() for c in results['combinations']]
                
                # Insert to database
                count = insert_flight_combinations_batch(combos_to_save)
                
                stats["profiles_processed"] += 1
                stats["total_combinations"] += count
                
                # Check for price drops
                alert_count = _check_and_log_price_changes(
                    profile_id=profile_id,
                    origin=origin,
                    destination=destination,
                    new_combinations=results['combinations'],
                    previous_prices=previous_prices,
                    threshold_pct=profile.get('notify_on_drop_percent', 10.0),
                )
                stats["alerts_generated"] += alert_count
                
                # Update last searched timestamp
                update_profile_last_searched(profile_id)
                
                stats["profiles_detail"].append({
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "combinations": count,
                    "alerts": alert_count,
                    "status": "success",
                })
                
                logger.info(f"  ✓ {profile_name}: {count} combinations, {alert_count} alerts")
                
            else:
                stats["profiles_detail"].append({
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "combinations": 0,
                    "status": "no_data",
                })
                logger.warning(f"  ⚠ {profile_name}: No data returned")
                
        except Exception as e:
            stats["profiles_failed"] += 1
            stats["profiles_detail"].append({
                "profile_id": profile_id,
                "profile_name": profile_name,
                "combinations": 0,
                "status": "error",
                "error": str(e),
            })
            logger.error(f"  ✗ {profile_name}: {e}")
    
    # Finalize stats
    stats["end_time"] = datetime.now().isoformat()
    
    # Final progress update
    if progress_callback:
        progress_callback(100, "Collection complete")
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Collection complete!")
    logger.info(f"Profiles processed: {stats['profiles_processed']}")
    logger.info(f"Profiles failed: {stats['profiles_failed']}")
    logger.info(f"Total combinations: {stats['total_combinations']:,}")
    logger.info(f"Alerts generated: {stats['alerts_generated']}")
    logger.info("=" * 60)
    
    return stats


def _check_and_log_price_changes(
    profile_id: int,
    origin: str,
    destination: str,
    new_combinations: list,
    previous_prices: Dict[tuple, float],
    threshold_pct: float = 10.0,
) -> int:
    """
    Check for significant price changes and log alerts.
    
    Args:
        profile_id: The search profile ID
        origin: Origin airport code
        destination: Destination airport code
        new_combinations: List of new FlightCombination objects
        previous_prices: Dict mapping (departure_date, return_date) to previous price
        threshold_pct: Percentage drop to trigger alert (default 10%)
        
    Returns:
        Number of alerts generated
    """
    alerts_generated = 0
    
    for combo in new_combinations:
        key = (str(combo.departure_date), str(combo.return_date))
        
        if key in previous_prices:
            old_price = previous_prices[key]
            new_price = combo.price
            
            if old_price > 0:
                change_pct = ((old_price - new_price) / old_price) * 100
                
                # Check for significant drop
                if change_pct >= threshold_pct:
                    alert = {
                        'search_profile_id': profile_id,
                        'departure_date': str(combo.departure_date),
                        'return_date': str(combo.return_date),
                        'old_price': old_price,
                        'new_price': new_price,
                        'price_change_percent': round(change_pct, 1),
                        'alert_type': 'price_drop',
                    }
                    
                    insert_price_alert(alert)
                    alerts_generated += 1
                    
                    logger.info(
                        f"  🔔 Alert: {combo.departure_date} → {combo.return_date} "
                        f"dropped {change_pct:.0f}% (${old_price:.0f} → ${new_price:.0f})"
                    )
                
                # Also track significant increases (optional)
                elif change_pct <= -threshold_pct:
                    alert = {
                        'search_profile_id': profile_id,
                        'departure_date': str(combo.departure_date),
                        'return_date': str(combo.return_date),
                        'old_price': old_price,
                        'new_price': new_price,
                        'price_change_percent': round(change_pct, 1),
                        'alert_type': 'price_increase',
                    }
                    
                    insert_price_alert(alert)
                    # Don't count increases as alerts (just log)
    
    return alerts_generated


def run_single_profile(
    profile_id: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run collection for a single profile.
    
    Args:
        profile_id: Profile ID to collect
        progress_callback: Optional callback for progress updates
        
    Returns:
        Collection statistics
    """
    profile = get_profile_by_id(profile_id)
    
    if not profile:
        return {
            "success": False,
            "error": f"Profile {profile_id} not found",
        }
    
    return run_daily_collection(
        profile_ids=[profile_id],
        progress_callback=progress_callback,
    )


def run_calendar_search(
    origin: str,
    destination: str,
    target_month: str,
    trip_duration_days: int = 21,
    flexibility_days: int = 2,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run a one-off comprehensive calendar search (not tied to a profile).
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        target_month: Target month (YYYY-MM)
        trip_duration_days: Trip duration in days
        flexibility_days: Flexibility range (±days)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Search results including combinations and statistics
    """
    logger.info(f"Running calendar search: {origin} → {destination} ({target_month})")
    
    try:
        results = fetch_comprehensive_calendar(
            origin=origin,
            destination=destination,
            target_month=target_month,
            trip_duration_days=trip_duration_days,
            flexibility_days=flexibility_days,
            progress_callback=progress_callback,
        )
        
        if results['combinations']:
            # Save to database
            combos_to_save = [c.to_dict() for c in results['combinations']]
            count = insert_flight_combinations_batch(combos_to_save)
            
            return {
                "success": True,
                "route": f"{origin}-{destination}",
                "month": target_month,
                "combinations_found": len(results['combinations']),
                "combinations_saved": count,
                "failed_searches": len(results['failed']),
                "combinations": results['combinations'],
            }
        else:
            return {
                "success": True,
                "route": f"{origin}-{destination}",
                "month": target_month,
                "combinations_found": 0,
                "message": "No data returned",
            }
            
    except Exception as e:
        logger.error(f"Calendar search failed: {e}")
        return {
            "success": False,
            "route": f"{origin}-{destination}",
            "error": str(e),
        }


def start_scheduler(
    run_time: str = DEFAULT_SCHEDULE_TIME,
    run_immediately: bool = False,
) -> None:
    """
    Start the scheduling loop.
    
    Args:
        run_time: Time to run daily collection (HH:MM format)
        run_immediately: If True, run collection immediately on start
    """
    global _shutdown_requested
    _shutdown_requested = False
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    
    logger.info("=" * 60)
    logger.info("Flight Price Tracker Scheduler (Calendar Mode)")
    logger.info(f"Daily collection scheduled at: {run_time}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Log active profiles
    profiles = get_search_profiles(active_only=True)
    logger.info(f"Active tracking profiles: {len(profiles)}")
    for p in profiles:
        logger.info(f"  - {p['origin']} → {p['destination']} ({p.get('target_departure_month', 'any month')})")
    
    # Schedule daily job
    schedule.every().day.at(run_time).do(run_daily_collection)
    
    # Run immediately if requested
    if run_immediately:
        logger.info("Running immediate collection...")
        run_daily_collection()
    
    # Main loop
    while not _shutdown_requested:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)
    
    logger.info("Scheduler stopped")


def get_next_run_time() -> Optional[datetime]:
    """
    Get the next scheduled run time.
    
    Returns:
        Next run datetime or None if no jobs scheduled
    """
    jobs = schedule.get_jobs()
    
    if jobs:
        next_run = jobs[0].next_run
        return next_run
    
    return None


def get_scheduler_status() -> Dict[str, Any]:
    """
    Get current scheduler status.
    
    Returns:
        Dictionary with scheduler status information
    """
    next_run = get_next_run_time()
    profiles = get_search_profiles(active_only=True)
    
    return {
        "running": not _shutdown_requested,
        "jobs_scheduled": len(schedule.get_jobs()),
        "next_run": next_run.isoformat() if next_run else None,
        "current_time": datetime.now().isoformat(),
        "active_profiles": len(profiles),
        "profiles": [
            {
                "id": p['id'],
                "route": f"{p['origin']} → {p['destination']}",
                "month": p.get('target_departure_month'),
                "last_searched": p.get('last_searched'),
            }
            for p in profiles
        ],
    }


def estimate_next_collection_size() -> Dict[str, Any]:
    """
    Estimate the size of the next collection run.
    
    Returns:
        Dictionary with estimation details
    """
    from data_fetcher import estimate_search_time
    
    profiles = get_search_profiles(active_only=True)
    
    total_combinations = 0
    total_time = 0
    estimates = []
    
    for p in profiles:
        month = p.get('target_departure_month') or date.today().strftime('%Y-%m')
        duration = p.get('trip_duration_days', 21)
        flex = p.get('flexibility_days', 2)
        
        est = estimate_search_time(month, duration, flex)
        
        total_combinations += est['total_combinations']
        total_time += est['estimated_seconds']
        
        estimates.append({
            "profile_id": p['id'],
            "route": f"{p['origin']} → {p['destination']}",
            "combinations": est['total_combinations'],
            "estimated_seconds": est['estimated_seconds'],
        })
    
    return {
        "total_profiles": len(profiles),
        "total_combinations": total_combinations,
        "total_estimated_seconds": total_time,
        "total_estimated_minutes": round(total_time / 60, 1),
        "profiles": estimates,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flight Price Tracker Scheduler (Calendar Mode)")
    parser.add_argument(
        "--time", "-t",
        default=DEFAULT_SCHEDULE_TIME,
        help=f"Daily collection time (default: {DEFAULT_SCHEDULE_TIME})"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run collection immediately on start"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run collection once and exit (don't start scheduler)"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate next collection size and exit"
    )
    parser.add_argument(
        "--profile",
        type=int,
        help="Run collection for specific profile ID only"
    )
    
    args = parser.parse_args()
    
    if args.estimate:
        # Show estimation
        est = estimate_next_collection_size()
        print(f"Active profiles: {est['total_profiles']}")
        print(f"Total combinations: {est['total_combinations']:,}")
        print(f"Estimated time: {est['total_estimated_minutes']:.1f} minutes")
        print("\nBreakdown:")
        for p in est['profiles']:
            print(f"  {p['route']}: {p['combinations']} combos (~{p['estimated_seconds']:.0f}s)")
        sys.exit(0)
        
    elif args.once:
        # Single run mode
        if args.profile:
            result = run_single_profile(args.profile)
        else:
            result = run_daily_collection()
            
        print(f"Collection complete: {result.get('total_combinations', 0)} combinations")
        print(f"Alerts generated: {result.get('alerts_generated', 0)}")
        sys.exit(0 if result.get('success') else 1)
        
    else:
        # Scheduler mode
        start_scheduler(run_time=args.time, run_immediately=args.run_now)
