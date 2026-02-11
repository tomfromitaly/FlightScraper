"""
Data models and schemas for Flight Price Tracker.

Uses Pydantic for validation and serialization.
Enhanced architecture: Top-K offers, decision engine, fair value distributions.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, Any, List, Dict, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json


# =============================================================================
# ENUMS
# =============================================================================

class RiskProfile(str, Enum):
    """User risk tolerance for buy/wait decisions."""
    AGGRESSIVE = "aggressive"      # Accept more risk for potential savings
    MODERATE = "moderate"          # Balanced approach
    CONSERVATIVE = "conservative"  # Minimize risk of price increase


class Decision(str, Enum):
    """Decision engine output."""
    BUY = "buy"
    WAIT = "wait"
    WATCH_CLOSELY = "watch_closely"


class PriceRegime(str, Enum):
    """Current pricing regime for a route."""
    PROMO = "promo"           # Z-score < -1.5 (anomaly low)
    NORMAL = "normal"         # -1.5 <= Z <= 1.5
    ELEVATED = "elevated"     # Z > 1.5, inventory pressure
    SCARCITY = "scarcity"     # Z > 2.5, likely sold out soon


class RouteArchetype(str, Enum):
    """Route behavior classification."""
    BUSINESS_HEAVY = "business"   # Mon/Fri peaks, last-minute premium
    LEISURE_HEAVY = "leisure"     # Weekend departure preference, advance booking
    MIXED = "mixed"


# =============================================================================
# CORE DATA MODELS
# =============================================================================

class FlightCombination(BaseModel):
    """
    Represents a single flight combination (departure + return).
    
    Enhanced with:
    - Full timestamp (not just date)
    - Offer rank for top-K storage
    - Extended pricing (base fare, taxes)
    - Baggage info
    - Availability signals
    - Overnight layover detection
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    
    # Timestamp (full datetime)
    scrape_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Route
    departure_airport: str = Field(..., min_length=3, max_length=3)
    arrival_airport: str = Field(..., min_length=3, max_length=3)
    
    # Exact travel dates
    departure_date: date
    return_date: date
    trip_duration_days: int = Field(..., ge=1)
    
    # Offer ranking (1=cheapest, 2=second, 3=third)
    offer_rank: int = Field(default=1, ge=1)
    
    # Pricing breakdown
    price: float = Field(..., gt=0)
    currency: str = Field(default="USD", max_length=3)
    base_fare: Optional[float] = None
    taxes_fees: Optional[float] = None
    
    # Baggage info
    included_bags: int = Field(default=0, ge=0)
    checked_bag_price: Optional[float] = None
    carry_on_allowed: bool = Field(default=True)
    
    # Availability signals
    available_seats: Optional[int] = None
    last_ticketing_date: Optional[date] = None
    
    # Flight details
    airline_outbound: Optional[str] = None
    airline_return: Optional[str] = None
    stops_outbound: int = Field(default=0, ge=0)
    stops_return: int = Field(default=0, ge=0)
    duration_outbound_minutes: Optional[int] = None
    duration_return_minutes: Optional[int] = None
    cabin_class: str = Field(default="economy")
    
    # Generalized cost components
    overnight_layover: bool = Field(default=False)
    connection_airports: Optional[str] = None  # JSON array
    
    # Analytics fields
    lead_time_days: int = Field(..., ge=0)
    departure_day_of_week: Optional[str] = None
    return_day_of_week: Optional[str] = None
    is_weekend_departure: Optional[bool] = None
    is_weekend_return: Optional[bool] = None
    
    # Booking reference
    booking_token: Optional[str] = None
    deep_link_url: Optional[str] = None
    
    # Metadata
    search_query_id: Optional[str] = None
    other_metadata: Optional[str] = None
    
    @field_validator('departure_airport', 'arrival_airport')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        """Ensure airport codes are uppercase."""
        return v.upper().strip()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for database insertion."""
        data = self.model_dump()
        # Convert datetime/date objects to strings for SQLite
        if data.get('scrape_timestamp'):
            data['scrape_timestamp'] = data['scrape_timestamp'].isoformat()
        for field in ['departure_date', 'return_date', 'last_ticketing_date']:
            if data.get(field):
                data[field] = str(data[field])
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlightCombination":
        """Create model from dictionary (e.g., database row)."""
        # Handle scrape_timestamp
        if 'scrape_timestamp' in data and isinstance(data['scrape_timestamp'], str):
            try:
                data['scrape_timestamp'] = datetime.fromisoformat(data['scrape_timestamp'])
            except ValueError:
                data['scrape_timestamp'] = datetime.strptime(data['scrape_timestamp'], "%Y-%m-%d %H:%M:%S")
        
        # Handle date fields
        for field in ['departure_date', 'return_date', 'last_ticketing_date']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.strptime(data[field], "%Y-%m-%d").date()
        
        return cls(**data)
    
    @property
    def route(self) -> str:
        """Get route string (e.g., 'JFK-MEX')."""
        return f"{self.departure_airport}-{self.arrival_airport}"
    
    @property
    def total_stops(self) -> int:
        """Total stops across both legs."""
        return self.stops_outbound + self.stops_return
    
    def get_connection_airports_list(self) -> List[str]:
        """Parse connection airports JSON to list."""
        if self.connection_airports:
            try:
                return json.loads(self.connection_airports)
            except json.JSONDecodeError:
                return []
        return []
    
    @classmethod
    def calculate_analytics(cls, departure_date: date, return_date: date, scrape_date: date) -> dict:
        """Calculate analytics fields from dates."""
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        dep_weekday = departure_date.weekday()
        ret_weekday = return_date.weekday()
        
        return {
            'trip_duration_days': (return_date - departure_date).days,
            'lead_time_days': (departure_date - scrape_date).days,
            'departure_day_of_week': weekday_names[dep_weekday],
            'return_day_of_week': weekday_names[ret_weekday],
            'is_weekend_departure': dep_weekday >= 5,  # Saturday or Sunday
            'is_weekend_return': ret_weekday >= 5,
        }


class SearchProfile(BaseModel):
    """
    Represents a user's search preferences for tracking.
    
    Enhanced with risk profile for decision engine.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    profile_name: Optional[str] = None
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    target_departure_month: Optional[str] = None  # Format: '2026-10'
    trip_duration_days: int = Field(..., ge=1, le=60)
    flexibility_days: int = Field(default=2, ge=0, le=7)
    max_stops: int = Field(default=2, ge=0)
    preferred_airlines: Optional[str] = None  # JSON array
    max_price_threshold: Optional[float] = None
    notify_on_drop_percent: float = Field(default=10.0, ge=0)
    is_active: bool = Field(default=True)
    created_date: Optional[date] = None
    last_searched: Optional[datetime] = None
    
    # Risk profile for decision engine
    risk_profile: RiskProfile = Field(default=RiskProfile.MODERATE)
    
    @field_validator('origin', 'destination')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        """Ensure airport codes are uppercase."""
        return v.upper().strip()
    
    @field_validator('target_departure_month')
    @classmethod
    def validate_target_month(cls, v: Optional[str]) -> Optional[str]:
        """Validate month format YYYY-MM."""
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m")
            except ValueError:
                raise ValueError("target_departure_month must be in YYYY-MM format")
        return v
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for database insertion."""
        data = self.model_dump()
        for field in ['created_date']:
            if data.get(field):
                data[field] = str(data[field])
        if data.get('last_searched'):
            data['last_searched'] = data['last_searched'].isoformat()
        if data.get('risk_profile'):
            data['risk_profile'] = data['risk_profile'].value if isinstance(data['risk_profile'], RiskProfile) else data['risk_profile']
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchProfile":
        """Create model from dictionary (e.g., database row)."""
        if 'created_date' in data and isinstance(data['created_date'], str):
            data['created_date'] = datetime.strptime(data['created_date'], "%Y-%m-%d").date()
        if 'last_searched' in data and isinstance(data['last_searched'], str):
            try:
                data['last_searched'] = datetime.fromisoformat(data['last_searched'])
            except ValueError:
                data['last_searched'] = datetime.strptime(data['last_searched'], "%Y-%m-%d %H:%M:%S")
        return cls(**data)
    
    @property
    def route(self) -> str:
        """Get route string (e.g., 'JFK-MEX')."""
        return f"{self.origin}-{self.destination}"
    
    def get_preferred_airlines_list(self) -> List[str]:
        """Parse preferred airlines JSON to list."""
        if self.preferred_airlines:
            try:
                return json.loads(self.preferred_airlines)
            except json.JSONDecodeError:
                return []
        return []


class PriceAlert(BaseModel):
    """
    Represents a price change alert.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    search_profile_id: Optional[int] = None
    alert_date: datetime = Field(default_factory=datetime.now)
    departure_date: Optional[date] = None
    return_date: Optional[date] = None
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    price_change_percent: Optional[float] = None
    alert_type: Optional[str] = None  # 'price_drop', 'price_increase', 'deal'
    is_read: bool = Field(default=False)
    
    @property
    def savings(self) -> float:
        """Calculate absolute savings (positive = cheaper)."""
        if self.old_price and self.new_price:
            return self.old_price - self.new_price
        return 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for database insertion."""
        data = self.model_dump()
        if data.get('alert_date'):
            data['alert_date'] = data['alert_date'].isoformat()
        for field in ['departure_date', 'return_date']:
            if data.get(field):
                data[field] = str(data[field])
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriceAlert":
        """Create model from dictionary."""
        if 'alert_date' in data and isinstance(data['alert_date'], str):
            try:
                data['alert_date'] = datetime.fromisoformat(data['alert_date'])
            except ValueError:
                data['alert_date'] = datetime.strptime(data['alert_date'], "%Y-%m-%d %H:%M:%S")
        for field in ['departure_date', 'return_date']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.strptime(data[field], "%Y-%m-%d").date()
        return cls(**data)


# =============================================================================
# FAIR VALUE & DECISION ENGINE MODELS
# =============================================================================

class FairValueDistribution(BaseModel):
    """
    Route-specific price distribution for an itinerary pattern.
    
    Used for quantile-based deal detection and decision making.
    """
    origin: str
    destination: str
    
    # Quantile thresholds
    p10: float  # Great deal threshold
    p25: float  # Good deal threshold
    p50: float  # Fair value (median)
    p75: float  # Above average
    p90: float  # Expensive threshold
    
    # Statistics
    mean: float
    std_dev: float
    sample_count: int
    
    # Context
    month: Optional[int] = None
    day_of_week: Optional[str] = None
    lead_time_bucket: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    def get_percentile_label(self, price: float) -> str:
        """Get a label for where a price falls in the distribution."""
        if price <= self.p10:
            return "Great deal (below P10)"
        elif price <= self.p25:
            return "Good deal (P10-P25)"
        elif price <= self.p50:
            return "Below average (P25-P50)"
        elif price <= self.p75:
            return "Above average (P50-P75)"
        elif price <= self.p90:
            return "Expensive (P75-P90)"
        else:
            return "Very expensive (above P90)"
    
    def calculate_z_score(self, price: float) -> float:
        """Calculate z-score for a price."""
        if self.std_dev == 0:
            return 0.0
        return (price - self.mean) / self.std_dev
    
    def get_percentile(self, price: float) -> float:
        """Estimate percentile for a price (linear interpolation)."""
        if price <= self.p10:
            return 10 * (price / self.p10) if self.p10 > 0 else 0
        elif price <= self.p25:
            return 10 + 15 * (price - self.p10) / (self.p25 - self.p10) if self.p25 > self.p10 else 10
        elif price <= self.p50:
            return 25 + 25 * (price - self.p25) / (self.p50 - self.p25) if self.p50 > self.p25 else 25
        elif price <= self.p75:
            return 50 + 25 * (price - self.p50) / (self.p75 - self.p50) if self.p75 > self.p50 else 50
        elif price <= self.p90:
            return 75 + 15 * (price - self.p75) / (self.p90 - self.p75) if self.p90 > self.p75 else 75
        else:
            return min(99, 90 + 10 * (price - self.p90) / self.p90) if self.p90 > 0 else 99


class DecisionResult(BaseModel):
    """
    Output from the decision engine.
    """
    decision: Decision
    confidence: float = Field(..., ge=0, le=1)  # 0-1
    deadline: date  # Latest recommended action date
    
    # Expected value analysis
    expected_savings_if_wait: float = 0.0
    rebound_risk: float = 0.0  # 0-1 probability
    
    # Context
    current_price: float
    fair_value: float
    percentile: float
    regime: PriceRegime
    
    # Reasoning
    reasoning: List[str] = Field(default_factory=list)
    
    def get_recommendation_text(self) -> str:
        """Generate human-readable recommendation."""
        if self.decision == Decision.BUY:
            return f"BUY NOW (confidence: {self.confidence:.0%}) - Price is good, book before {self.deadline}"
        elif self.decision == Decision.WAIT:
            return f"WAIT (confidence: {self.confidence:.0%}) - Expected savings: ${self.expected_savings_if_wait:.0f}"
        else:
            return f"WATCH CLOSELY (confidence: {self.confidence:.0%}) - Check again soon, deadline: {self.deadline}"


class BookingOutcome(BaseModel):
    """
    Tracks actual booking outcomes for feedback loop.
    """
    id: Optional[int] = None
    search_profile_id: Optional[int] = None
    departure_date: date
    return_date: date
    
    # Decision made
    decision_date: datetime
    decision_made: Decision
    price_at_decision: float
    fair_value_at_decision: Optional[float] = None
    regime_at_decision: Optional[PriceRegime] = None
    
    # Actual outcome
    booked_date: Optional[datetime] = None
    booked_price: Optional[float] = None
    final_lowest_price: Optional[float] = None  # Lowest price before departure
    
    # Percentile tracking
    booked_price_percentile: Optional[float] = None
    
    # Regret analysis
    regret_vs_optimal: Optional[float] = None  # booked_price - final_lowest_price
    decision_was_correct: Optional[bool] = None
    
    def calculate_regret(self) -> Optional[float]:
        """Calculate regret vs optimal price."""
        if self.booked_price is not None and self.final_lowest_price is not None:
            self.regret_vs_optimal = self.booked_price - self.final_lowest_price
            return self.regret_vs_optimal
        return None


class BacktestResult(BaseModel):
    """
    Results from backtesting a decision strategy.
    """
    strategy_name: str
    route: str
    date_range_start: date
    date_range_end: date
    
    # Decision counts
    total_decisions: int = 0
    buy_decisions: int = 0
    wait_decisions: int = 0
    watch_decisions: int = 0
    
    # Accuracy
    correct_buy_decisions: int = 0
    correct_wait_decisions: int = 0
    accuracy: float = 0.0
    
    # Regret analysis
    total_regret: float = 0.0
    avg_regret_per_decision: float = 0.0
    regret_vs_ex_post_best: float = 0.0  # vs always buying at the lowest
    
    # Savings
    total_savings_vs_naive: float = 0.0  # vs always buying immediately
    avg_savings_per_booking: float = 0.0


# =============================================================================
# ANALYSIS RESULT MODELS
# =============================================================================

class PriceMatrixCell(BaseModel):
    """Single cell in a price matrix."""
    departure_date: date
    return_date: date
    price: Optional[float] = None
    airline: Optional[str] = None
    stops: Optional[int] = None
    duration_minutes: Optional[int] = None


class PriceMatrix(BaseModel):
    """
    2D price matrix for calendar visualization.
    
    Rows: Departure dates
    Columns: Return dates (based on trip duration variations)
    """
    origin: str
    destination: str
    target_month: str
    base_trip_duration: int
    flexibility_days: int
    
    # Matrix data
    departure_dates: List[date] = Field(default_factory=list)
    return_date_offsets: List[int] = Field(default_factory=list)  # e.g., [-2, -1, 0, 1, 2]
    prices: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)  # dep_date -> {offset -> price}
    
    # Statistics
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    avg_price: Optional[float] = None
    
    def get_price(self, departure_date: date, duration_offset: int) -> Optional[float]:
        """Get price for specific departure date and duration offset."""
        dep_str = str(departure_date)
        offset_str = str(duration_offset)
        return self.prices.get(dep_str, {}).get(offset_str)
    
    def set_price(self, departure_date: date, duration_offset: int, price: float) -> None:
        """Set price for specific departure date and duration offset."""
        dep_str = str(departure_date)
        offset_str = str(duration_offset)
        if dep_str not in self.prices:
            self.prices[dep_str] = {}
        self.prices[dep_str][offset_str] = price


class FlexibilitySavings(BaseModel):
    """
    Analysis of potential savings from flexible dates.
    """
    base_departure: date
    base_return: date
    base_price: float
    
    # Alternatives
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    # Each alternative: {departure, return, price, savings, recommendation}
    
    # Best alternative
    best_alternative: Optional[Dict[str, Any]] = None
    max_savings: float = 0.0
    avg_savings_if_flexible: float = 0.0
    
    def add_alternative(
        self,
        departure: date,
        return_date: date,
        price: float
    ) -> None:
        """Add an alternative date combination."""
        savings = self.base_price - price
        
        # Generate recommendation text
        dep_diff = (departure - self.base_departure).days
        ret_diff = (return_date - self.base_return).days
        
        parts = []
        if dep_diff != 0:
            direction = "later" if dep_diff > 0 else "earlier"
            parts.append(f"Leave {abs(dep_diff)} day{'s' if abs(dep_diff) > 1 else ''} {direction}")
        if ret_diff != 0:
            direction = "later" if ret_diff > 0 else "earlier"
            parts.append(f"Return {abs(ret_diff)} day{'s' if abs(ret_diff) > 1 else ''} {direction}")
        
        recommendation = ", ".join(parts) if parts else "Same dates"
        if savings > 0:
            recommendation += f": Save ${savings:.2f}"
        elif savings < 0:
            recommendation += f": Costs ${abs(savings):.2f} more"
        
        alternative = {
            'departure': str(departure),
            'return': str(return_date),
            'price': price,
            'savings': savings,
            'recommendation': recommendation,
            'departure_diff': dep_diff,
            'return_diff': ret_diff,
        }
        
        self.alternatives.append(alternative)
        
        # Update best alternative
        if savings > self.max_savings:
            self.max_savings = savings
            self.best_alternative = alternative
    
    def calculate_avg_savings(self) -> float:
        """Calculate average savings across alternatives with lower prices."""
        cheaper = [a['savings'] for a in self.alternatives if a['savings'] > 0]
        if cheaper:
            self.avg_savings_if_flexible = sum(cheaper) / len(cheaper)
        return self.avg_savings_if_flexible


class OptimalBuyDate(BaseModel):
    """
    Recommendation for when to buy tickets.
    """
    target_departure: date
    target_return: date
    
    # Recommendation
    optimal_lead_days: int
    recommended_buy_date: date
    expected_price: Optional[float] = None
    
    # Confidence
    confidence: str = "low"  # 'low', 'medium', 'high'
    data_points: int = 0
    
    # Price trend
    price_trend: str = "stable"  # 'stable', 'increasing', 'decreasing'
    
    # Historical context
    historical_min: Optional[float] = None
    historical_max: Optional[float] = None
    historical_avg: Optional[float] = None


class CombinationRanking(BaseModel):
    """
    Ranked flight combination for top deals display.
    """
    rank: int
    departure_date: date
    return_date: date
    trip_duration: int
    price: float
    
    # Flight details
    airline_outbound: Optional[str] = None
    airline_return: Optional[str] = None
    stops_outbound: int = 0
    stops_return: int = 0
    
    # Comparison to average
    vs_avg_percent: Optional[float] = None  # Negative = cheaper than avg
    
    # Fair value context
    percentile: Optional[float] = None
    fair_value: Optional[float] = None
    
    # Booking recommendation
    recommended_buy_date: Optional[date] = None
    booking_url: Optional[str] = None
    
    @property
    def is_deal(self) -> bool:
        """Check if this is a deal (>10% below average or below P25)."""
        if self.percentile is not None:
            return self.percentile <= 25
        return self.vs_avg_percent is not None and self.vs_avg_percent < -10


# =============================================================================
# COMPREHENSIVE ANALYSIS RESULT
# =============================================================================

class WeekdayAnalysis(BaseModel):
    """Analysis results for a specific weekday."""
    weekday: str
    weekday_num: int = Field(..., ge=0, le=6)
    avg_price: float
    min_price: float
    max_price: float
    sample_count: int
    savings_vs_expensive: float = 0.0  # Savings vs most expensive day


class LeadTimeBucket(BaseModel):
    """Analysis results for a lead time bucket."""
    bucket_name: str
    min_days: int
    max_days: int
    avg_price: float
    min_price: float
    sample_count: int


class ComprehensiveAnalysisResult(BaseModel):
    """
    Complete analysis result for a route's calendar search.
    
    Contains all insights from comprehensive price analysis.
    """
    # Route info
    route: str
    origin: str
    destination: str
    target_month: str
    trip_duration: int
    flexibility_days: int
    analysis_date: datetime = Field(default_factory=datetime.now)
    
    # Best combination
    best_combo: Optional[Dict[str, Any]] = None
    # {departure, return, price, buy_date, percentile, regime}
    
    # Top 10 combinations
    top_10_combos: List[CombinationRanking] = Field(default_factory=list)
    
    # Fair value distribution
    fair_value: Optional[FairValueDistribution] = None
    current_regime: Optional[PriceRegime] = None
    
    # Flexibility insights
    flexibility_insights: Dict[str, Any] = Field(default_factory=dict)
    # {avg_savings_if_flexible, examples: [...]}
    
    # Weekday analysis
    weekday_analysis: Dict[str, Any] = Field(default_factory=dict)
    # {cheapest_departure_day, cheapest_return_day, most_expensive_day, avg_weekend_premium}
    weekday_details: List[WeekdayAnalysis] = Field(default_factory=list)
    
    # Booking windows
    booking_windows: Dict[str, Any] = Field(default_factory=dict)
    # {0-14_days: {avg, count}, 15-30_days: {...}, optimal_window, optimal_lead_days}
    lead_time_details: List[LeadTimeBucket] = Field(default_factory=list)
    
    # Price statistics
    price_stats: Dict[str, Any] = Field(default_factory=dict)
    # {min, max, avg, median, std_dev, p10, p25, p50, p75, p90}
    
    # Current deals
    current_deals: List[Dict[str, Any]] = Field(default_factory=list)
    # [{departure, return, price, percentile, vs_avg, status}]
    
    # Data quality metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # {total_combinations, data_coverage_percent, oldest_scrape, newest_scrape, confidence}
    
    def get_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Route Analysis: {self.route}",
            f"Travel Month: {self.target_month}",
            f"Trip Duration: {self.trip_duration} days (±{self.flexibility_days})",
            ""
        ]
        
        if self.current_regime:
            lines.append(f"Current Regime: {self.current_regime.value.upper()}")
        
        if self.best_combo:
            lines.extend([
                f"Best Price: ${self.best_combo.get('price', 'N/A')}",
                f"Best Dates: {self.best_combo.get('departure')} → {self.best_combo.get('return')}",
                f"Recommended Buy Date: {self.best_combo.get('buy_date', 'N/A')}",
                ""
            ])
        
        if self.price_stats:
            lines.extend([
                f"Price Range: ${self.price_stats.get('min', 0):.2f} - ${self.price_stats.get('max', 0):.2f}",
                f"Fair Value (P50): ${self.price_stats.get('p50', self.price_stats.get('avg', 0)):.2f}",
                ""
            ])
        
        if self.weekday_analysis:
            lines.append(f"Cheapest Departure Day: {self.weekday_analysis.get('cheapest_departure_day', 'N/A')}")
        
        if self.booking_windows:
            lines.append(f"Optimal Booking Window: {self.booking_windows.get('optimal_window', 'N/A')}")
        
        return "\n".join(lines)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Alias for backward compatibility
FlightSnapshot = FlightCombination
TrackingTarget = SearchProfile
