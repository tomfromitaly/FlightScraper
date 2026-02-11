"""
Database write operations for Flight Price Tracker.

Handles logging of flight snapshots and management of tracking targets.
"""

import logging
import sqlite3
from datetime import date, datetime
from typing import Optional

from database import get_db_connection, execute_query, update_last_scraped
from models import FlightSnapshot, TrackingTarget, PriceAlert

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# SNAPSHOT LOGGING
# =============================================================================

def log_snapshots(snapshots: list[FlightSnapshot]) -> int:
    """
    Bulk insert flight snapshots into the database.
    
    Uses INSERT OR REPLACE to handle duplicates gracefully.
    
    Args:
        snapshots: List of FlightSnapshot objects to insert
        
    Returns:
        Number of records successfully inserted/updated
        
    Example:
        >>> count = log_snapshots(snapshots)
        >>> print(f"Logged {count} snapshots")
    """
    if not snapshots:
        logger.warning("No snapshots to log")
        return 0
    
    insert_sql = """
        INSERT OR REPLACE INTO flight_snapshots (
            scrape_date, departure_airport, arrival_airport,
            departure_date, return_date, price, currency,
            airline, airline_code, stops, is_nonstop,
            duration_outbound_minutes, duration_return_minutes,
            lead_time_days, cabin_class, booking_token, other_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    inserted_count = 0
    failed_count = 0
    
    try:
        with get_db_connection() as conn:
            for snapshot in snapshots:
                try:
                    data = snapshot.to_dict()
                    values = (
                        data['scrape_date'],
                        data['departure_airport'],
                        data['arrival_airport'],
                        data['departure_date'],
                        data['return_date'],
                        data['price'],
                        data['currency'],
                        data['airline'],
                        data['airline_code'],
                        data['stops'],
                        data['is_nonstop'],
                        data['duration_outbound_minutes'],
                        data['duration_return_minutes'],
                        data['lead_time_days'],
                        data['cabin_class'],
                        data['booking_token'],
                        data['other_metadata'],
                    )
                    conn.execute(insert_sql, values)
                    inserted_count += 1
                    
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert snapshot: {e}")
                    failed_count += 1
            
            conn.commit()
        
        logger.info(f"Logged {inserted_count} snapshots ({failed_count} failures)")
        return inserted_count
        
    except sqlite3.Error as e:
        logger.error(f"Batch insert failed: {e}")
        raise


def log_snapshot_batch(
    origin: str,
    destination: str,
    snapshots: list[FlightSnapshot]
) -> int:
    """
    Log a batch of snapshots and update the tracking target's last_scraped date.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        snapshots: List of snapshots to log
        
    Returns:
        Number of snapshots logged
    """
    count = log_snapshots(snapshots)
    
    if count > 0:
        update_last_scraped(origin, destination, date.today())
    
    return count


# =============================================================================
# TRACKING TARGET MANAGEMENT
# =============================================================================

def add_tracking_target(
    origin: str,
    destination: str,
    duration: int = 7,
    target_month: Optional[str] = None,
    notify_price: Optional[float] = None,
) -> Optional[int]:
    """
    Add a new route to track.
    
    Args:
        origin: Origin airport IATA code
        destination: Destination airport IATA code
        duration: Trip duration in days
        target_month: Optional target month in 'YYYY-MM' format
        notify_price: Optional price threshold for alerts
        
    Returns:
        ID of the created tracking target, or None if creation failed
    """
    origin = origin.upper().strip()
    destination = destination.upper().strip()
    
    insert_sql = """
        INSERT OR IGNORE INTO tracking_targets (
            origin, destination, trip_duration_days, 
            target_month, notify_below_price, is_active
        ) VALUES (?, ?, ?, ?, ?, TRUE)
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                insert_sql,
                (origin, destination, duration, target_month, notify_price)
            )
            
            if cursor.rowcount == 0:
                # Already exists, get existing ID
                cursor = conn.execute(
                    "SELECT id FROM tracking_targets WHERE origin = ? AND destination = ? AND trip_duration_days = ?",
                    (origin, destination, duration)
                )
                row = cursor.fetchone()
                if row:
                    logger.info(f"Route {origin}-{destination} ({duration}d) already exists")
                    return row['id']
                return None
            
            target_id = cursor.lastrowid
            logger.info(f"Added tracking target: {origin}-{destination} ({duration}d) [ID: {target_id}]")
            return target_id
            
    except sqlite3.Error as e:
        logger.error(f"Failed to add tracking target: {e}")
        return None


def update_tracking_target(
    target_id: int,
    is_active: Optional[bool] = None,
    target_month: Optional[str] = None,
    notify_price: Optional[float] = None,
    trip_duration: Optional[int] = None,
) -> bool:
    """
    Update a tracking target's properties.
    
    Args:
        target_id: ID of the tracking target
        is_active: Optional new active status
        target_month: Optional new target month
        notify_price: Optional new price threshold
        trip_duration: Optional new trip duration
        
    Returns:
        True if update succeeded
    """
    updates = []
    params = []
    
    if is_active is not None:
        updates.append("is_active = ?")
        params.append(is_active)
    
    if target_month is not None:
        updates.append("target_month = ?")
        params.append(target_month)
    
    if notify_price is not None:
        updates.append("notify_below_price = ?")
        params.append(notify_price)
    
    if trip_duration is not None:
        updates.append("trip_duration_days = ?")
        params.append(trip_duration)
    
    if not updates:
        return True  # Nothing to update
    
    params.append(target_id)
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE tracking_targets SET {', '.join(updates)} WHERE id = ?",
                tuple(params)
            )
        logger.info(f"Updated tracking target {target_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Failed to update tracking target: {e}")
        return False


def deactivate_target(target_id: int) -> bool:
    """
    Deactivate a tracking target (stop tracking without deleting).
    
    Args:
        target_id: ID of the tracking target
        
    Returns:
        True if deactivation succeeded
    """
    return update_tracking_target(target_id, is_active=False)


def activate_target(target_id: int) -> bool:
    """
    Reactivate a previously deactivated tracking target.
    
    Args:
        target_id: ID of the tracking target
        
    Returns:
        True if activation succeeded
    """
    return update_tracking_target(target_id, is_active=True)


def delete_tracking_target(target_id: int) -> bool:
    """
    Permanently delete a tracking target.
    
    Args:
        target_id: ID of the tracking target
        
    Returns:
        True if deletion succeeded
    """
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM tracking_targets WHERE id = ?", (target_id,))
        logger.info(f"Deleted tracking target {target_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Failed to delete tracking target: {e}")
        return False


# =============================================================================
# PRICE HISTORY QUERIES
# =============================================================================

def get_price_history(
    origin: str,
    destination: str,
    days_back: int = 30,
    departure_date: Optional[date] = None,
) -> list[dict]:
    """
    Get price history for a route.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        days_back: Number of days of history to retrieve
        departure_date: Optional specific departure date to filter by
        
    Returns:
        List of price snapshot dictionaries
    """
    query = """
        SELECT * FROM flight_snapshots
        WHERE departure_airport = ? AND arrival_airport = ?
        AND scrape_date >= date('now', ?)
    """
    params = [origin.upper(), destination.upper(), f"-{days_back} days"]
    
    if departure_date:
        query += " AND departure_date = ?"
        params.append(str(departure_date))
    
    query += " ORDER BY scrape_date DESC, departure_date ASC"
    
    return execute_query(query, tuple(params), fetch="all") or []


def get_price_trend(
    origin: str,
    destination: str,
    departure_date: date,
) -> list[dict]:
    """
    Get the price trend over time for a specific departure date.
    
    Useful for seeing how prices changed as departure approached.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        departure_date: The departure date to track
        
    Returns:
        List of price points sorted by scrape date
    """
    query = """
        SELECT scrape_date, price, lead_time_days, airline
        FROM flight_snapshots
        WHERE departure_airport = ? AND arrival_airport = ?
        AND departure_date = ?
        ORDER BY scrape_date ASC
    """
    
    return execute_query(
        query,
        (origin.upper(), destination.upper(), str(departure_date)),
        fetch="all"
    ) or []


# =============================================================================
# PRICE ALERTS
# =============================================================================

def log_price_alert(alert: PriceAlert) -> Optional[int]:
    """
    Log a price drop alert.
    
    Args:
        alert: PriceAlert object
        
    Returns:
        ID of the created alert, or None if creation failed
    """
    insert_sql = """
        INSERT INTO price_alerts (
            origin, destination, departure_date, return_date,
            previous_price, new_price, price_drop_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                insert_sql,
                (
                    alert.origin,
                    alert.destination,
                    str(alert.departure_date),
                    str(alert.return_date),
                    alert.previous_price,
                    alert.new_price,
                    alert.price_drop_pct,
                )
            )
            return cursor.lastrowid
            
    except sqlite3.Error as e:
        logger.error(f"Failed to log price alert: {e}")
        return None


def get_unread_alerts(limit: int = 50) -> list[dict]:
    """
    Get unread price alerts.
    
    Args:
        limit: Maximum number of alerts to return
        
    Returns:
        List of unread alert dictionaries
    """
    query = """
        SELECT * FROM price_alerts
        WHERE is_read = FALSE
        ORDER BY alert_date DESC
        LIMIT ?
    """
    return execute_query(query, (limit,), fetch="all") or []


def mark_alerts_read(alert_ids: list[int]) -> bool:
    """
    Mark alerts as read.
    
    Args:
        alert_ids: List of alert IDs to mark as read
        
    Returns:
        True if update succeeded
    """
    if not alert_ids:
        return True
    
    placeholders = ','.join(['?'] * len(alert_ids))
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE price_alerts SET is_read = TRUE WHERE id IN ({placeholders})",
                tuple(alert_ids)
            )
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Failed to mark alerts read: {e}")
        return False


def check_price_drops(
    origin: str,
    destination: str,
    threshold_pct: float = 10.0,
) -> list[PriceAlert]:
    """
    Check for price drops compared to recent history.
    
    Compares latest prices to 7-day averages and creates alerts
    for significant drops.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        threshold_pct: Minimum percentage drop to trigger alert
        
    Returns:
        List of PriceAlert objects for detected drops
    """
    alerts = []
    
    # Get average prices from 7+ days ago
    history_query = """
        SELECT departure_date, return_date, AVG(price) as avg_price
        FROM flight_snapshots
        WHERE departure_airport = ? AND arrival_airport = ?
        AND scrape_date < date('now', '-7 days')
        GROUP BY departure_date, return_date
    """
    
    # Get latest prices
    latest_query = """
        SELECT fs.departure_date, fs.return_date, fs.price
        FROM flight_snapshots fs
        INNER JOIN (
            SELECT departure_date, return_date, MAX(scrape_date) as max_scrape
            FROM flight_snapshots
            WHERE departure_airport = ? AND arrival_airport = ?
            GROUP BY departure_date, return_date
        ) latest ON fs.departure_date = latest.departure_date 
            AND fs.return_date = latest.return_date
            AND fs.scrape_date = latest.max_scrape
        WHERE fs.departure_airport = ? AND fs.arrival_airport = ?
    """
    
    history = execute_query(
        history_query,
        (origin.upper(), destination.upper()),
        fetch="all"
    ) or []
    
    latest = execute_query(
        latest_query,
        (origin.upper(), destination.upper(), origin.upper(), destination.upper()),
        fetch="all"
    ) or []
    
    # Build lookup for historical averages
    history_lookup = {
        (h['departure_date'], h['return_date']): h['avg_price']
        for h in history
    }
    
    # Check for drops
    for price_point in latest:
        key = (price_point['departure_date'], price_point['return_date'])
        if key in history_lookup:
            avg_price = history_lookup[key]
            current_price = price_point['price']
            
            if avg_price > 0:
                drop_pct = ((avg_price - current_price) / avg_price) * 100
                
                if drop_pct >= threshold_pct:
                    # Handle both string and date object types
                    dep_date = price_point['departure_date']
                    ret_date = price_point['return_date']
                    if isinstance(dep_date, str):
                        dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
                    if isinstance(ret_date, str):
                        ret_date = datetime.strptime(ret_date, "%Y-%m-%d").date()
                    
                    alert = PriceAlert(
                        origin=origin.upper(),
                        destination=destination.upper(),
                        departure_date=dep_date,
                        return_date=ret_date,
                        previous_price=avg_price,
                        new_price=current_price,
                        price_drop_pct=drop_pct,
                    )
                    alerts.append(alert)
    
    return alerts


# =============================================================================
# DATA EXPORT
# =============================================================================

def export_route_data(
    origin: str,
    destination: str,
    output_path: str,
    format: str = "csv",
) -> bool:
    """
    Export route data to a file.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        output_path: Path for the output file
        format: Output format ('csv' or 'json')
        
    Returns:
        True if export succeeded
    """
    import csv
    import json
    
    data = get_price_history(origin, destination, days_back=365)
    
    if not data:
        logger.warning(f"No data to export for {origin}-{destination}")
        return False
    
    try:
        if format.lower() == "csv":
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(data)} records to {output_path}")
        return True
        
    except IOError as e:
        logger.error(f"Export failed: {e}")
        return False
