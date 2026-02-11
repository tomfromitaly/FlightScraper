"""
Amadeus API integration for Flight Price Tracker.

Handles all communication with the Amadeus Flight Offers Search API.
Enhanced architecture: Top-K offers, full timestamps, extended fields.
"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional, List, Callable, Dict, Any, Tuple
from threading import Lock

from amadeus import Client, ResponseError

from config import (
    AMADEUS_CLIENT_ID,
    AMADEUS_CLIENT_SECRET,
    MAX_RESULTS_PER_QUERY,
    RATE_LIMIT_DELAY,
    DEFAULT_CABIN_CLASS,
    CURRENCY,
    get_airline_name,
)
from models import FlightCombination
from utils import (
    retry_with_backoff,
    RateLimiter,
    calculate_lead_time,
    log_progress,
    parse_date_range,
    get_dates_in_range,
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rate limiter - thread-safe
rate_limiter = RateLimiter(min_delay=RATE_LIMIT_DELAY, max_per_minute=100)
_rate_lock = Lock()

# Maximum parallel API calls
MAX_PARALLEL_CALLS = 3

# Default number of offers to fetch per query (top-K)
DEFAULT_TOP_K = 3


# =============================================================================
# AMADEUS CLIENT INITIALIZATION
# =============================================================================

def get_amadeus_client() -> Client:
    """
    Initialize and return an Amadeus API client.
    
    Returns:
        Configured Amadeus Client instance
        
    Raises:
        ValueError: If credentials are missing
    """
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        raise ValueError(
            "Amadeus API credentials not configured. "
            "Please set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in .env"
        )
    
    return Client(
        client_id=AMADEUS_CLIENT_ID,
        client_secret=AMADEUS_CLIENT_SECRET,
        hostname='test',  # Use 'test' for sandbox, 'production' for live
        logger=logger,
    )


# Global client instance (lazy initialization)
_amadeus_client: Optional[Client] = None
_client_lock = Lock()


def _get_client() -> Client:
    """Get or create the global Amadeus client (thread-safe)."""
    global _amadeus_client
    with _client_lock:
        if _amadeus_client is None:
            _amadeus_client = get_amadeus_client()
    return _amadeus_client


# =============================================================================
# CORE FLIGHT SEARCH FUNCTIONS
# =============================================================================

def _wait_for_rate_limit():
    """Thread-safe rate limit waiting."""
    with _rate_lock:
        rate_limiter.wait()


@retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(ResponseError, Exception))
def fetch_single_combination(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    adults: int = 1,
    cabin_class: str = DEFAULT_CABIN_CLASS,
    max_results: int = 1,  # Default to 1 for backward compatibility
    nonstop_only: bool = False,
    scrape_timestamp: Optional[datetime] = None,
) -> Optional[FlightCombination]:
    """
    Fetch the cheapest flight offer for a single departure-return combination.
    
    Args:
        origin: Origin airport IATA code (e.g., 'JFK')
        destination: Destination airport IATA code (e.g., 'MEX')
        departure_date: Outbound flight date
        return_date: Return flight date
        adults: Number of adult passengers
        cabin_class: Cabin class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
        max_results: Maximum number of offers to return
        nonstop_only: If True, only return nonstop flights
        scrape_timestamp: Timestamp when this price was scraped (defaults to now)
        
    Returns:
        FlightCombination object or None if no results
    """
    client = _get_client()
    
    if scrape_timestamp is None:
        scrape_timestamp = datetime.now()
    
    # Respect rate limits (thread-safe)
    _wait_for_rate_limit()
    
    try:
        # Build search parameters
        search_params = {
            'originLocationCode': origin.upper(),
            'destinationLocationCode': destination.upper(),
            'departureDate': str(departure_date),
            'returnDate': str(return_date),
            'adults': adults,
            'currencyCode': CURRENCY,
            'max': max_results,
        }
        
        if nonstop_only:
            search_params['nonStop'] = 'true'
        
        if cabin_class and cabin_class != 'ECONOMY':
            search_params['travelClass'] = cabin_class
        
        # Make API request
        response = client.shopping.flight_offers_search.get(**search_params)
        
        # Parse the cheapest offer (first one returned by Amadeus)
        if response.data:
            return _parse_flight_offer_to_combination(
                response.data[0],
                origin,
                destination,
                departure_date,
                return_date,
                scrape_timestamp,
                cabin_class,
                offer_rank=1
            )
        
        return None
        
    except ResponseError as e:
        logger.warning(f"Amadeus API error for {origin}-{destination} on {departure_date}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching {origin}-{destination}: {e}")
        raise


@retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(ResponseError, Exception))
def fetch_top_k_offers(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    adults: int = 1,
    cabin_class: str = DEFAULT_CABIN_CLASS,
    top_k: int = DEFAULT_TOP_K,
    nonstop_only: bool = False,
    scrape_timestamp: Optional[datetime] = None,
) -> List[FlightCombination]:
    """
    Fetch top-K flight offers for a single departure-return combination.
    
    Returns multiple ranked offers for price dispersion analysis.
    
    Args:
        origin: Origin airport IATA code
        destination: Destination airport IATA code
        departure_date: Outbound flight date
        return_date: Return flight date
        adults: Number of adult passengers
        cabin_class: Cabin class
        top_k: Number of offers to fetch (default: 3)
        nonstop_only: If True, only return nonstop flights
        scrape_timestamp: Timestamp when scraped
        
    Returns:
        List of FlightCombination objects ranked by price (1=cheapest)
    """
    client = _get_client()
    
    if scrape_timestamp is None:
        scrape_timestamp = datetime.now()
    
    # Respect rate limits
    _wait_for_rate_limit()
    
    try:
        search_params = {
            'originLocationCode': origin.upper(),
            'destinationLocationCode': destination.upper(),
            'departureDate': str(departure_date),
            'returnDate': str(return_date),
            'adults': adults,
            'currencyCode': CURRENCY,
            'max': top_k,
        }
        
        if nonstop_only:
            search_params['nonStop'] = 'true'
        
        if cabin_class and cabin_class != 'ECONOMY':
            search_params['travelClass'] = cabin_class
        
        response = client.shopping.flight_offers_search.get(**search_params)
        
        combinations = []
        for rank, offer in enumerate(response.data, start=1):
            combo = _parse_flight_offer_to_combination(
                offer,
                origin,
                destination,
                departure_date,
                return_date,
                scrape_timestamp,
                cabin_class,
                offer_rank=rank
            )
            if combo:
                combinations.append(combo)
        
        return combinations
        
    except ResponseError as e:
        logger.warning(f"Amadeus API error for {origin}-{destination} on {departure_date}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching {origin}-{destination}: {e}")
        raise


def _parse_flight_offer_to_combination(
    offer: dict,
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    scrape_timestamp: datetime,
    cabin_class: str = "economy",
    offer_rank: int = 1
) -> Optional[FlightCombination]:
    """
    Parse a single flight offer from Amadeus API response into FlightCombination.
    
    Enhanced to extract:
    - Price breakdown (base fare, taxes)
    - Baggage info
    - Available seats
    - Last ticketing date
    - Overnight layovers
    - Connection airports
    """
    try:
        # Extract price breakdown
        price_data = offer.get('price')
        if not price_data or 'total' not in price_data:
            logger.warning("Missing price data in flight offer")
            return None
        
        price = float(price_data['total'])
        currency = price_data.get('currency', CURRENCY)
        
        # Base fare and taxes
        base_fare = None
        taxes_fees = None
        if 'base' in price_data:
            base_fare = float(price_data['base'])
            taxes_fees = price - base_fare
        elif 'grandTotal' in price_data and 'base' in price_data:
            base_fare = float(price_data.get('base', 0))
            taxes_fees = float(price_data.get('grandTotal', price)) - base_fare
        
        # Extract availability signals
        available_seats = offer.get('numberOfBookableSeats')
        last_ticketing_date_str = offer.get('lastTicketingDate')
        last_ticketing_date = None
        if last_ticketing_date_str:
            try:
                last_ticketing_date = datetime.strptime(last_ticketing_date_str, '%Y-%m-%d').date()
            except ValueError:
                pass
        
        # Extract baggage info from travelerPricings
        included_bags = 0
        checked_bag_price = None
        carry_on_allowed = True
        
        traveler_pricings = offer.get('travelerPricings', [])
        if traveler_pricings:
            first_traveler = traveler_pricings[0]
            fare_details = first_traveler.get('fareDetailsBySegment', [])
            if fare_details:
                first_segment_fare = fare_details[0]
                
                # Included checked bags
                included_checked = first_segment_fare.get('includedCheckedBags', {})
                if included_checked:
                    included_bags = included_checked.get('quantity', 0) or 0
                    if included_checked.get('weight'):
                        # If weight is specified, assume at least 1 bag
                        included_bags = max(1, included_bags)
                
                # Amenities (carry-on info)
                amenities = first_segment_fare.get('amenities', [])
                for amenity in amenities:
                    if amenity.get('amenityType') == 'CARRY_ON':
                        carry_on_allowed = amenity.get('isChargeable', False) is False
        
        # Extract itinerary details
        itineraries = offer.get('itineraries', [])
        if len(itineraries) < 2:
            return None
        
        outbound = itineraries[0]
        return_flight = itineraries[1]
        
        # Count stops and extract connection airports
        outbound_segments = outbound.get('segments', [])
        return_segments = return_flight.get('segments', [])
        stops_outbound = len(outbound_segments) - 1
        stops_return = len(return_segments) - 1
        
        # Extract connection airports
        connection_airports = []
        for seg in outbound_segments[:-1]:  # All segments except last
            arrival = seg.get('arrival', {}).get('iataCode')
            if arrival:
                connection_airports.append(arrival)
        for seg in return_segments[:-1]:
            arrival = seg.get('arrival', {}).get('iataCode')
            if arrival:
                connection_airports.append(arrival)
        
        connection_airports_json = json.dumps(connection_airports) if connection_airports else None
        
        # Detect overnight layovers
        overnight_layover = _detect_overnight_layover(outbound_segments) or \
                          _detect_overnight_layover(return_segments)
        
        # Parse durations
        outbound_duration = _parse_duration(outbound.get('duration', ''))
        return_duration = _parse_duration(return_flight.get('duration', ''))
        
        # Get airline info for outbound
        airline_outbound = None
        if outbound_segments:
            first_segment = outbound_segments[0]
            airline_code = first_segment.get('operating', {}).get('carrierCode') or \
                          first_segment.get('carrierCode')
            if airline_code:
                airline_outbound = get_airline_name(airline_code)
        
        # Get airline info for return
        airline_return = None
        if return_segments:
            first_segment = return_segments[0]
            airline_code = first_segment.get('operating', {}).get('carrierCode') or \
                          first_segment.get('carrierCode')
            if airline_code:
                airline_return = get_airline_name(airline_code)
        
        # Calculate analytics
        analytics = FlightCombination.calculate_analytics(
            departure_date, return_date, scrape_timestamp.date()
        )
        
        # Create FlightCombination with all enhanced fields
        return FlightCombination(
            scrape_timestamp=scrape_timestamp,
            departure_airport=origin.upper(),
            arrival_airport=destination.upper(),
            departure_date=departure_date,
            return_date=return_date,
            trip_duration_days=analytics['trip_duration_days'],
            offer_rank=offer_rank,
            price=price,
            currency=currency,
            base_fare=base_fare,
            taxes_fees=taxes_fees,
            included_bags=included_bags,
            checked_bag_price=checked_bag_price,
            carry_on_allowed=carry_on_allowed,
            available_seats=available_seats,
            last_ticketing_date=last_ticketing_date,
            airline_outbound=airline_outbound,
            airline_return=airline_return,
            stops_outbound=stops_outbound,
            stops_return=stops_return,
            duration_outbound_minutes=outbound_duration,
            duration_return_minutes=return_duration,
            cabin_class=cabin_class.lower() if cabin_class else "economy",
            overnight_layover=overnight_layover,
            connection_airports=connection_airports_json,
            lead_time_days=analytics['lead_time_days'],
            departure_day_of_week=analytics['departure_day_of_week'],
            return_day_of_week=analytics['return_day_of_week'],
            is_weekend_departure=analytics['is_weekend_departure'],
            is_weekend_return=analytics['is_weekend_return'],
            booking_token=offer.get('id'),
        )
        
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse flight offer: {e}")
        return None


def _detect_overnight_layover(segments: List[dict]) -> bool:
    """
    Detect if there's an overnight layover in the segments.
    
    Overnight = arrival on one day, departure on a different day.
    """
    if len(segments) <= 1:
        return False
    
    for i in range(len(segments) - 1):
        try:
            arrival_str = segments[i].get('arrival', {}).get('at', '')
            departure_str = segments[i + 1].get('departure', {}).get('at', '')
            
            if arrival_str and departure_str:
                arrival_dt = datetime.fromisoformat(arrival_str.replace('Z', '+00:00'))
                departure_dt = datetime.fromisoformat(departure_str.replace('Z', '+00:00'))
                
                # If departure is on a different calendar day than arrival
                if arrival_dt.date() != departure_dt.date():
                    return True
        except (ValueError, AttributeError):
            continue
    
    return False


def _parse_duration(duration_str: str) -> Optional[int]:
    """Parse ISO 8601 duration string to minutes."""
    if not duration_str or not duration_str.startswith('PT'):
        return None
    
    try:
        duration_str = duration_str[2:]  # Remove 'PT' prefix
        hours = 0
        minutes = 0
        
        if 'H' in duration_str:
            h_idx = duration_str.index('H')
            hours = int(duration_str[:h_idx])
            duration_str = duration_str[h_idx + 1:]
        
        if 'M' in duration_str:
            m_idx = duration_str.index('M')
            minutes = int(duration_str[:m_idx])
        
        return hours * 60 + minutes
        
    except (ValueError, IndexError):
        return None


# =============================================================================
# COMPREHENSIVE CALENDAR SEARCH (ENHANCED)
# =============================================================================

def generate_search_combinations(
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2
) -> List[Tuple[date, date]]:
    """
    Generate all departure-return date combinations for a target month.
    
    Args:
        target_month: Target month in 'YYYY-MM' format (e.g., '2026-10')
        trip_duration_days: Base trip duration in days
        flexibility_days: Number of days flexibility (±) on trip duration
        
    Returns:
        List of (departure_date, return_date) tuples
    """
    start_date, end_date = parse_date_range(target_month)
    departure_dates = get_dates_in_range(start_date, end_date)
    
    combinations = []
    
    for departure_date in departure_dates:
        # Generate return dates for each flexibility option
        for delta in range(-flexibility_days, flexibility_days + 1):
            actual_duration = trip_duration_days + delta
            if actual_duration < 1:  # Skip invalid durations
                continue
            return_date = departure_date + timedelta(days=actual_duration)
            combinations.append((departure_date, return_date))
    
    return combinations


def _fetch_topk_with_tracking(
    params: Dict[str, Any],
    completed_count: List[int],
    total_count: int,
    progress_callback: Optional[Callable],
    lock: Lock,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[Tuple[date, date], List[FlightCombination], Optional[str]]:
    """
    Fetch top-K offers for a single combination with progress tracking.
    
    Returns:
        Tuple of ((departure_date, return_date), list_of_results, error_message_or_none)
    """
    departure_date = params['departure_date']
    return_date = params['return_date']
    
    try:
        results = fetch_top_k_offers(
            origin=params['origin'],
            destination=params['destination'],
            departure_date=departure_date,
            return_date=return_date,
            cabin_class=params.get('cabin_class', 'ECONOMY'),
            top_k=top_k,
            scrape_timestamp=params.get('scrape_timestamp'),
        )
        
        # Update progress
        with lock:
            completed_count[0] += 1
            if progress_callback:
                pct = int((completed_count[0] / total_count) * 100)
                msg = f"Searching {departure_date}: {completed_count[0]}/{total_count} combinations"
                progress_callback(pct, msg)
        
        return ((departure_date, return_date), results, None)
        
    except Exception as e:
        # Update progress even on failure
        with lock:
            completed_count[0] += 1
            if progress_callback:
                pct = int((completed_count[0] / total_count) * 100)
                msg = f"Searching {departure_date}: {completed_count[0]}/{total_count} combinations"
                progress_callback(pct, msg)
        
        return ((departure_date, return_date), [], str(e))


def _fetch_single_with_tracking(
    params: Dict[str, Any],
    completed_count: List[int],
    total_count: int,
    progress_callback: Optional[Callable],
    lock: Lock
) -> Tuple[Tuple[date, date], Optional[FlightCombination], Optional[str]]:
    """
    Fetch a single combination with progress tracking.
    Legacy function - returns only cheapest offer.
    
    Returns:
        Tuple of ((departure_date, return_date), result_or_none, error_message_or_none)
    """
    departure_date = params['departure_date']
    return_date = params['return_date']
    
    try:
        result = fetch_single_combination(
            origin=params['origin'],
            destination=params['destination'],
            departure_date=departure_date,
            return_date=return_date,
            cabin_class=params.get('cabin_class', 'ECONOMY'),
            scrape_timestamp=params.get('scrape_timestamp'),
        )
        
        # Update progress
        with lock:
            completed_count[0] += 1
            if progress_callback:
                pct = int((completed_count[0] / total_count) * 100)
                msg = f"Searching {departure_date}: {completed_count[0]}/{total_count} combinations"
                progress_callback(pct, msg)
        
        return ((departure_date, return_date), result, None)
        
    except Exception as e:
        # Update progress even on failure
        with lock:
            completed_count[0] += 1
            if progress_callback:
                pct = int((completed_count[0] / total_count) * 100)
                msg = f"Searching {departure_date}: {completed_count[0]}/{total_count} combinations"
                progress_callback(pct, msg)
        
        return ((departure_date, return_date), None, str(e))


def fetch_comprehensive_calendar(
    origin: str,
    destination: str,
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    max_stops: int = 2,
    cabin_class: str = 'ECONOMY',
    progress_callback: Optional[Callable[[int, str], None]] = None,
    max_workers: int = MAX_PARALLEL_CALLS,
    fetch_top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Fetch ALL possible flight combinations for a route within a target month.
    
    Enhanced to fetch top-K offers per combination for price dispersion analysis.
    
    For target_month='2026-10' with trip_duration_days=21 and flexibility_days=2:
    - Departures: Oct 1-31 (31 dates)
    - Returns per departure: duration-2, duration-1, duration, duration+1, duration+2
    - Total: 31 × 5 = 155 API calls
    - With top_k=3: Up to 155 × 3 = 465 offer records
    
    Args:
        origin: Origin airport IATA code (e.g., 'JFK')
        destination: Destination airport IATA code (e.g., 'MEX')
        target_month: Target month in 'YYYY-MM' format
        trip_duration_days: Base trip duration in days
        flexibility_days: Days of flexibility (±) on trip duration
        max_stops: Maximum stops allowed (not currently enforced in API)
        cabin_class: Cabin class to search
        progress_callback: Optional callback(percentage, status_message)
        max_workers: Maximum parallel API calls (default: 3)
        fetch_top_k: Number of offers to fetch per combination (default: 3)
        
    Returns:
        Dictionary with:
        - combinations: List[FlightCombination] - all ranked offers
        - failed: List[Tuple[date, date, str]] - failed combinations with error
        - total_searched: int
        - success_count: int - number of date combos with results
        - total_offers: int - total number of offers fetched
        - search_query_id: str - unique ID for this search
    """
    scrape_timestamp = datetime.now()
    search_query_id = str(uuid.uuid4())[:8]
    
    # Generate all combinations to search
    date_combinations = generate_search_combinations(
        target_month, trip_duration_days, flexibility_days
    )
    total_count = len(date_combinations)
    
    logger.info(
        f"Starting comprehensive calendar search for {origin}-{destination} "
        f"({target_month}, {trip_duration_days}±{flexibility_days} days): "
        f"{total_count} combinations, fetching top-{fetch_top_k} offers each"
    )
    
    if progress_callback:
        progress_callback(0, f"Starting search: 0/{total_count} combinations")
    
    # Prepare search parameters for each combination
    search_params_list = [
        {
            'origin': origin,
            'destination': destination,
            'departure_date': dep_date,
            'return_date': ret_date,
            'cabin_class': cabin_class,
            'scrape_timestamp': scrape_timestamp,
        }
        for dep_date, ret_date in date_combinations
    ]
    
    # Results storage
    all_combinations = []
    failed = []
    completed_count = [0]  # Use list for mutable reference in threads
    progress_lock = Lock()
    success_combos = 0
    
    # Execute in parallel with limited workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_topk_with_tracking,
                params,
                completed_count,
                total_count,
                progress_callback,
                progress_lock,
                fetch_top_k
            ): params
            for params in search_params_list
        }
        
        for future in as_completed(futures):
            try:
                (dep_date, ret_date), results, error = future.result()
                
                if results:
                    # Add search query ID to all results
                    for combo in results:
                        combo.search_query_id = search_query_id
                    all_combinations.extend(results)
                    success_combos += 1
                elif error:
                    failed.append((dep_date, ret_date, error))
                else:
                    # No result but no error (no flights found)
                    failed.append((dep_date, ret_date, "No flights found"))
                    
            except Exception as e:
                params = futures[future]
                failed.append((
                    params['departure_date'],
                    params['return_date'],
                    str(e)
                ))
    
    logger.info(
        f"Completed {origin}-{destination}: "
        f"{success_combos}/{total_count} combinations found, "
        f"{len(all_combinations)} total offers "
        f"({len(failed)} failures)"
    )
    
    if progress_callback:
        progress_callback(100, f"Complete: {success_combos}/{total_count} combinations, {len(all_combinations)} offers")
    
    return {
        'combinations': all_combinations,
        'failed': failed,
        'total_searched': total_count,
        'success_count': success_combos,
        'total_offers': len(all_combinations),
        'search_query_id': search_query_id,
        'origin': origin,
        'destination': destination,
        'target_month': target_month,
        'trip_duration_days': trip_duration_days,
        'flexibility_days': flexibility_days,
        'scrape_timestamp': scrape_timestamp,
        'top_k': fetch_top_k,
    }


def fetch_specific_dates(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    flexibility_days: int = 2,
    cabin_class: str = 'ECONOMY',
    progress_callback: Optional[Callable[[int, str], None]] = None,
    fetch_top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Fetch prices for a specific base combination plus flexibility options.
    
    Useful for "I want to travel Oct 15-Nov 5, but show me ±2 days options"
    
    Args:
        origin: Origin airport code
        destination: Destination airport code  
        departure_date: Base departure date
        return_date: Base return date
        flexibility_days: Days of flexibility on both departure and return
        cabin_class: Cabin class to search
        progress_callback: Optional callback(percentage, status_message)
        fetch_top_k: Number of offers per combination
        
    Returns:
        Dictionary with combinations and base_combination identified
    """
    scrape_timestamp = datetime.now()
    search_query_id = str(uuid.uuid4())[:8]
    
    # Generate combinations: vary both departure AND return by ±flexibility_days
    date_combinations = []
    for dep_delta in range(-flexibility_days, flexibility_days + 1):
        for ret_delta in range(-flexibility_days, flexibility_days + 1):
            dep = departure_date + timedelta(days=dep_delta)
            ret = return_date + timedelta(days=ret_delta)
            if dep < ret:  # Valid combination
                date_combinations.append((dep, ret))
    
    total_count = len(date_combinations)
    
    logger.info(
        f"Fetching specific dates with flexibility for {origin}-{destination}: "
        f"{total_count} combinations, top-{fetch_top_k} offers each"
    )
    
    if progress_callback:
        progress_callback(0, f"Starting: 0/{total_count}")
    
    all_combinations = []
    failed = []
    base_combination = None
    
    for i, (dep, ret) in enumerate(date_combinations, 1):
        try:
            results = fetch_top_k_offers(
                origin=origin,
                destination=destination,
                departure_date=dep,
                return_date=ret,
                cabin_class=cabin_class,
                top_k=fetch_top_k,
                scrape_timestamp=scrape_timestamp,
            )
            
            for combo in results:
                combo.search_query_id = search_query_id
            all_combinations.extend(results)
            
            # Identify base combination (cheapest offer for exact dates)
            if dep == departure_date and ret == return_date and results:
                base_combination = results[0]  # First is cheapest
                    
        except Exception as e:
            failed.append((dep, ret, str(e)))
        
        if progress_callback:
            pct = int((i / total_count) * 100)
            progress_callback(pct, f"Searching: {i}/{total_count}")
    
    return {
        'combinations': all_combinations,
        'base_combination': base_combination,
        'failed': failed,
        'total_searched': total_count,
        'success_count': len([c for c in all_combinations if c.offer_rank == 1]),
        'total_offers': len(all_combinations),
        'search_query_id': search_query_id,
    }


# =============================================================================
# LEGACY SUPPORT FUNCTIONS
# =============================================================================

def fetch_single_route(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: date,
    adults: int = 1,
    cabin_class: str = DEFAULT_CABIN_CLASS,
    max_results: int = MAX_RESULTS_PER_QUERY,
    nonstop_only: bool = False,
) -> List[FlightCombination]:
    """
    Fetch flight offers for a single route and date combination.
    
    Legacy compatibility function - returns list of FlightCombination.
    """
    client = _get_client()
    scrape_timestamp = datetime.now()
    
    _wait_for_rate_limit()
    
    try:
        search_params = {
            'originLocationCode': origin.upper(),
            'destinationLocationCode': destination.upper(),
            'departureDate': str(departure_date),
            'returnDate': str(return_date),
            'adults': adults,
            'currencyCode': CURRENCY,
            'max': max_results,
        }
        
        if nonstop_only:
            search_params['nonStop'] = 'true'
        
        if cabin_class and cabin_class != 'ECONOMY':
            search_params['travelClass'] = cabin_class
        
        response = client.shopping.flight_offers_search.get(**search_params)
        
        combinations = []
        for rank, offer in enumerate(response.data, start=1):
            combo = _parse_flight_offer_to_combination(
                offer, origin, destination, departure_date, return_date, 
                scrape_timestamp, cabin_class, offer_rank=rank
            )
            if combo:
                combinations.append(combo)
        
        return combinations
        
    except ResponseError as e:
        logger.error(f"Amadeus API error for {origin}-{destination} on {departure_date}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching {origin}-{destination}: {e}")
        raise


def fetch_date_range(
    origin: str,
    destination: str,
    start_date: date,
    end_date: date,
    trip_duration: int = 7,
) -> List[FlightCombination]:
    """
    Fetch flight prices for a specific date range.
    Legacy compatibility function.
    """
    all_combinations = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    
    logger.info(f"Fetching {origin}-{destination} from {start_date} to {end_date}")
    
    day_count = 0
    while current_date <= end_date:
        return_date = current_date + timedelta(days=trip_duration)
        day_count += 1
        
        try:
            result = fetch_single_combination(
                origin=origin,
                destination=destination,
                departure_date=current_date,
                return_date=return_date,
            )
            if result:
                all_combinations.append(result)
                
        except Exception as e:
            logger.warning(f"Failed to fetch {current_date}: {e}")
        
        log_progress(day_count, total_days, f"Fetching {origin}-{destination}")
        current_date += timedelta(days=1)
    
    return all_combinations


# =============================================================================
# API STATUS AND UTILITIES
# =============================================================================

def test_api_connection() -> dict:
    """
    Test the Amadeus API connection and return status.
    """
    try:
        client = _get_client()
        
        tomorrow = date.today() + timedelta(days=1)
        next_week = tomorrow + timedelta(days=7)
        
        response = client.shopping.flight_offers_search.get(
            originLocationCode='JFK',
            destinationLocationCode='LAX',
            departureDate=str(tomorrow),
            returnDate=str(next_week),
            adults=1,
            max=1,
        )
        
        return {
            "connected": True,
            "test_results": len(response.data),
            "message": "API connection successful",
        }
        
    except ResponseError as e:
        return {
            "connected": False,
            "error_code": getattr(e, 'code', 'UNKNOWN'),
            "message": str(e),
        }
    except Exception as e:
        return {
            "connected": False,
            "error_code": 'CONNECTION_ERROR',
            "message": str(e),
        }


def get_api_remaining_quota() -> dict:
    """Get information about API rate limits and remaining quota."""
    remaining = rate_limiter.get_remaining()
    
    return {
        "requests_remaining_this_minute": remaining,
        "max_per_minute": rate_limiter.max_per_minute,
        "min_delay_seconds": rate_limiter.min_delay,
        "estimated_capacity": f"~{int(60 / rate_limiter.min_delay)} requests/minute",
        "max_parallel_calls": MAX_PARALLEL_CALLS,
        "default_top_k": DEFAULT_TOP_K,
    }


def reset_client() -> None:
    """Reset the global Amadeus client (useful for re-authentication)."""
    global _amadeus_client
    with _client_lock:
        _amadeus_client = None
    logger.info("Amadeus client reset")


def estimate_search_time(
    target_month: str,
    trip_duration_days: int,
    flexibility_days: int = 2,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """
    Estimate how long a comprehensive search will take.
    
    Returns:
        Dictionary with estimated time and search parameters
    """
    combinations = generate_search_combinations(
        target_month, trip_duration_days, flexibility_days
    )
    total_combos = len(combinations)
    
    # With parallel execution and rate limiting
    # Each batch of MAX_PARALLEL_CALLS takes ~RATE_LIMIT_DELAY seconds
    batches = (total_combos + MAX_PARALLEL_CALLS - 1) // MAX_PARALLEL_CALLS
    estimated_seconds = batches * RATE_LIMIT_DELAY * 1.5  # Add buffer for processing
    
    return {
        'total_combinations': total_combos,
        'estimated_offers': total_combos * top_k,
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_seconds / 60,
        'max_parallel_calls': MAX_PARALLEL_CALLS,
        'rate_limit_delay': RATE_LIMIT_DELAY,
        'top_k': top_k,
    }
