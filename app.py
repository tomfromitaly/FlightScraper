"""
Streamlit Dashboard for Flight Price Tracker.

A modern, interactive web interface for comprehensive flight calendar search,
intelligent buy/wait decisions, and backtesting analysis.

New architecture: Calendar search, tracking dashboard, decision engine, and insights pages.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Flight Price Calendar",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import our modules after page config
from config import (
    get_all_airport_codes,
    get_airport_name,
    validate_iata_code,
    COLORS,
)
from database import (
    init_db,
    verify_db,
    get_database_stats,
    get_search_profiles,
    get_search_profile,
    create_search_profile,
    delete_search_profile,
    toggle_profile_active,
    get_latest_combinations,
    get_price_alerts,
    mark_all_alerts_read,
    insert_flight_combinations_batch,
    get_combinations_for_calendar,
)
from analyzer import (
    analyze_comprehensive_calendar,
    detect_price_patterns,
    generate_recommendations,
)
from calendar_builder import (
    build_price_matrix,
    find_cheapest_combinations,
    calculate_flexibility_savings,
    find_optimal_buy_date,
    generate_calendar_heatmap_data,
    find_deals,
    compare_duration_options,
)
from data_fetcher import (
    fetch_comprehensive_calendar,
    test_api_connection,
    estimate_search_time,
)
from visualizer import (
    plot_full_calendar_heatmap,
    plot_2d_price_matrix,
    plot_flexibility_comparison,
    plot_booking_timeline,
    plot_combo_comparison_grid,
    plot_weekday_heatmap,
    plot_lead_time_curve,
    create_dashboard_charts,
)
from utils import parse_date_range, format_currency
from models import RiskProfile, Decision, PriceRegime
from decision_engine import make_decision, get_recommendations_for_profile
from quantile_model import (
    get_fair_value_distribution,
    get_regime_with_context,
    find_promo_opportunities,
    get_complete_price_analysis,
)
from route_classifier import get_route_classification_details, get_booking_strategy
from backtester import compare_strategies, backtest_strategy
from outcome_tracker import get_feedback_summary
from baseline_builder import get_baseline_coverage, build_route_baselines


# =============================================================================
# INITIALIZATION
# =============================================================================

@st.cache_resource
def initialize_database():
    """Initialize database once per session."""
    return init_db()

initialize_database()


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .hero-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        color: #10b981;
    }
    
    .deal-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .deal-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .alert-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .profile-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .profile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .profile-route {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
    }
    
    .profile-status {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-active {
        background: #d1fae5;
        color: #059669;
    }
    
    .status-paused {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .recommendation-card {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .stProgress > div > div {
        background-color: #667eea;
    }
    
    .decision-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .decision-wait {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .decision-watch {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .regime-promo {
        background: #dcfce7;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .regime-normal {
        background: #f3f4f6;
        border-left: 4px solid #6b7280;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .regime-elevated {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .regime-scarcity {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .fair-value-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .backtest-winner {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## Flight Calendar")
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["Calendar Search", "Decision Dashboard", "Tracking Dashboard", "Backtesting", "Insights & Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Database stats
    db_stats = get_database_stats()
    st.markdown("### Database Stats")
    st.metric("Total Combinations", f"{db_stats['total_combinations']:,}")
    st.metric("Active Profiles", db_stats['active_profiles'])
    st.metric("Unread Alerts", db_stats['unread_alerts'])
    st.metric("Route Baselines", db_stats.get('route_baselines_count', 0))
    
    st.markdown("---")
    
    # API Status
    st.markdown("### API Status")
    api_status = test_api_connection()
    if api_status.get('connected'):
        st.success("Connected")
    else:
        st.error(f"Error: {api_status.get('message', 'Unknown')}")


# =============================================================================
# PAGE 1: CALENDAR SEARCH
# =============================================================================

def page_calendar_search():
    """Main calendar search page."""
    
    st.markdown("""
    <div class="hero-header">
        <h1>Flight Price Calendar</h1>
        <p>Search all possible flight combinations and find the best deals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search form
    col1, col2 = st.columns(2)
    
    with col1:
        airports = get_all_airport_codes()
        origin = st.selectbox(
            "From",
            airports,
            index=airports.index("JFK") if "JFK" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}"
        )
        
        dest = st.selectbox(
            "To",
            airports,
            index=airports.index("MEX") if "MEX" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}"
        )
    
    with col2:
        # Target month
        today = date.today()
        next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        
        target_month = st.date_input(
            "Travel Month",
            value=next_month,
            min_value=today,
        )
        target_month_str = target_month.strftime('%Y-%m')
        
        col2a, col2b = st.columns(2)
        with col2a:
            trip_duration = st.slider("Trip Duration (days)", 1, 60, 21)
        with col2b:
            flexibility = st.slider("Flexibility (±days)", 0, 7, 2)
    
    # Estimate search time
    estimate = estimate_search_time(target_month_str, trip_duration, flexibility)
    
    st.info(
        f"This will search **{estimate['total_combinations']} combinations** "
        f"(~{estimate['estimated_seconds']:.0f} seconds)"
    )
    
    # Search button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        search_clicked = st.button("Search All Combinations", type="primary", use_container_width=True)
    
    with col_btn2:
        use_cached = st.checkbox("Use cached data", value=True)
    
    with col_btn3:
        add_to_tracking = st.checkbox("Add to tracking")
    
    # Execute search
    if search_clicked:
        if origin == dest:
            st.error("Origin and destination must be different!")
            return
        
        # Check for cached data first
        if use_cached:
            cached = get_latest_combinations(origin, dest, target_month_str)
            if cached:
                st.success(f"Using {len(cached)} cached combinations from database")
                display_search_results(origin, dest, target_month_str, trip_duration, flexibility)
                
                if add_to_tracking:
                    add_profile_to_tracking(origin, dest, target_month_str, trip_duration, flexibility)
                return
        
        # Run new search
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, msg):
            progress_bar.progress(pct / 100)
            status_text.text(msg)
        
        with st.spinner("Searching..."):
            results = fetch_comprehensive_calendar(
                origin=origin,
                destination=dest,
                target_month=target_month_str,
                trip_duration_days=trip_duration,
                flexibility_days=flexibility,
                progress_callback=update_progress,
            )
        
        progress_bar.empty()
        status_text.empty()
        
        # Save to database
        if results['combinations']:
            combos_to_save = [c.to_dict() for c in results['combinations']]
            saved = insert_flight_combinations_batch(combos_to_save)
            st.success(f"Found {results['success_count']} combinations, saved {saved} to database")
        else:
            st.warning("No flight combinations found. Try different dates or route.")
            return
        
        # Add to tracking if requested
        if add_to_tracking:
            add_profile_to_tracking(origin, dest, target_month_str, trip_duration, flexibility)
        
        # Display results
        display_search_results(origin, dest, target_month_str, trip_duration, flexibility)
    
    # Show existing data if available
    else:
        cached = get_latest_combinations(origin, dest, target_month_str)
        if cached:
            st.markdown("---")
            st.markdown(f"### Showing cached data for {origin} → {dest} ({target_month_str})")
            display_search_results(origin, dest, target_month_str, trip_duration, flexibility)


def display_search_results(origin: str, dest: str, target_month: str, trip_duration: int, flexibility: int):
    """Display comprehensive search results."""
    
    # Get analysis
    analysis = analyze_comprehensive_calendar(origin, dest, target_month, trip_duration, flexibility)
    
    if not analysis.best_combo:
        st.warning("No analysis data available")
        return
    
    # Hero metrics
    st.markdown("### Best Deal Found")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Price</div>
            <div class="metric-value" style="color: {COLORS['cheap']}">${analysis.best_combo['price']:.0f}</div>
            <div class="metric-delta">Save ${analysis.price_stats['avg'] - analysis.best_combo['price']:.0f} vs avg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Dates</div>
            <div class="metric-value" style="font-size: 1rem">{analysis.best_combo['departure']} → {analysis.best_combo['return']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Book By</div>
            <div class="metric-value" style="font-size: 1rem">{analysis.best_combo.get('buy_date', 'ASAP')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        coverage = analysis.metadata.get('data_coverage_percent', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Coverage</div>
            <div class="metric-value">{coverage:.0f}%</div>
            <div class="metric-delta">{analysis.metadata.get('total_combinations_searched', 0)} combos</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Calendar View", "Price Matrix", "Flexibility Savings", "Top Deals", "Patterns"
    ])
    
    with tab1:
        # Calendar heatmap
        heatmap_data = generate_calendar_heatmap_data(
            origin, dest, target_month, trip_duration, flexibility
        )
        fig = plot_full_calendar_heatmap(origin, dest, target_month, trip_duration, heatmap_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 2D price matrix
        matrix_data = get_combinations_for_calendar(origin, dest, target_month, trip_duration, flexibility)
        fig = plot_2d_price_matrix(origin, dest, target_month, matrix_data, trip_duration, flexibility)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Flexibility savings
        if analysis.flexibility_insights:
            flex = analysis.flexibility_insights
            
            if flex.get('max_savings', 0) > 0:
                st.markdown(f"""
                <div class="deal-card">
                    <span class="deal-badge">FLEXIBILITY BONUS</span>
                    <h3 style="margin: 1rem 0 0.5rem">Save up to ${flex['max_savings']:.0f}</h3>
                    <p style="margin: 0; opacity: 0.9">Average savings: ${flex.get('avg_savings_if_flexible', 0):.0f} if flexible</p>
                </div>
                """, unsafe_allow_html=True)
            
            if flex.get('examples'):
                st.markdown("### Savings Opportunities")
                for example in flex['examples'][:5]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        {example}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Flexibility comparison chart
            if flex.get('best_alternative'):
                base_combo = {'base_price': analysis.price_stats['avg']}
                alternatives = [
                    alt for alt in (analysis.flexibility_insights.get('alternatives') or [])
                ] if 'alternatives' in str(analysis.flexibility_insights) else []
                
                if alternatives:
                    fig = plot_flexibility_comparison(base_combo, alternatives)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Top deals table
        if analysis.top_10_combos:
            fig = plot_combo_comparison_grid(analysis.top_10_combos, origin, dest)
            st.plotly_chart(fig, use_container_width=True)
            
            # Also show current deals
            deals = analysis.current_deals
            if deals:
                st.markdown("### Hot Deals (>15% below average)")
                for deal in deals[:5]:
                    st.markdown(f"""
                    <div class="deal-card">
                        <span class="deal-badge">DEAL - {abs(deal['vs_avg']):.0f}% OFF</span>
                        <h4 style="margin: 0.5rem 0">{deal['departure']} → {deal['return']}</h4>
                        <p style="margin: 0; font-size: 1.5rem; font-weight: bold">${deal['price']:.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab5:
        # Patterns and insights
        patterns = detect_price_patterns(origin, dest, target_month)
        
        if patterns.get('patterns'):
            st.markdown("### Detected Patterns")
            for pattern in patterns['patterns']:
                impact_color = COLORS['expensive'] if pattern['impact'] == 'high' else COLORS['moderate']
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>{pattern['type'].replace('_', ' ').title()}</strong><br>
                    {pattern['description']}
                    {f"<br><em>{pattern.get('recommendation', '')}</em>" if pattern.get('recommendation') else ""}
                </div>
                """, unsafe_allow_html=True)
        
        # Weekday analysis
        if analysis.weekday_details:
            st.markdown("### Best Days to Fly")
            weekday_data = [
                {
                    'weekday': w.weekday,
                    'avg_price': w.avg_price,
                    'min_price': w.min_price,
                    'sample_count': w.sample_count,
                }
                for w in analysis.weekday_details
            ]
            fig = plot_weekday_heatmap(weekday_data, origin, dest)
            st.plotly_chart(fig, use_container_width=True)
        
        # Booking window analysis
        if analysis.lead_time_details:
            st.markdown("### Best Time to Book")
            lead_data = [
                {
                    'bucket_name': b.bucket_name,
                    'avg_price': b.avg_price,
                    'min_price': b.min_price,
                    'sample_count': b.sample_count,
                    'min_days': b.min_days,
                }
                for b in analysis.lead_time_details
            ]
            fig = plot_lead_time_curve(lead_data, origin, dest)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    recommendations = generate_recommendations(analysis)
    if recommendations:
        st.markdown("---")
        st.markdown("### Recommendations")
        for rec in recommendations[:5]:
            st.markdown(f"""
            <div class="recommendation-card">
                {rec}
            </div>
            """, unsafe_allow_html=True)


def add_profile_to_tracking(origin: str, dest: str, target_month: str, trip_duration: int, flexibility: int):
    """Add a search profile to tracking."""
    profile = {
        'profile_name': f"{origin}-{dest} {target_month}",
        'origin': origin,
        'destination': dest,
        'target_departure_month': target_month,
        'trip_duration_days': trip_duration,
        'flexibility_days': flexibility,
        'is_active': True,
    }
    
    result = create_search_profile(profile)
    if result:
        st.success(f"Added {origin} → {dest} to tracking!")
    else:
        st.info("Profile already exists or could not be created")


# =============================================================================
# PAGE 2: TRACKING DASHBOARD
# =============================================================================

def page_tracking_dashboard():
    """Tracking dashboard page."""
    
    st.markdown("""
    <div class="hero-header">
        <h1>Tracking Dashboard</h1>
        <p>Monitor your saved searches and get price alerts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get profiles
    profiles = get_search_profiles(active_only=False)
    
    if not profiles:
        st.info("No search profiles yet. Start a calendar search and enable tracking!")
        return
    
    # Price alerts section
    alerts = get_price_alerts(unread_only=True, limit=10)
    
    if alerts:
        st.markdown("### Recent Price Alerts")
        for alert in alerts:
            alert_type = alert.get('alert_type', 'change')
            if alert_type == 'drop':
                st.markdown(f"""
                <div class="deal-card">
                    <span class="deal-badge">PRICE DROP</span>
                    <p style="margin: 0.5rem 0">
                        Price dropped {abs(alert.get('price_change_percent', 0)):.0f}% to ${alert.get('new_price', 0):.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("Mark All as Read"):
            mark_all_alerts_read()
            st.rerun()
        
        st.markdown("---")
    
    # Active profiles
    st.markdown("### Active Search Profiles")
    
    for profile in profiles:
        is_active = profile.get('is_active', True)
        status_class = "status-active" if is_active else "status-paused"
        status_text = "Active" if is_active else "Paused"
        
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-header">
                        <span class="profile-route">{profile['origin']} → {profile['destination']}</span>
                        <span class="profile-status {status_class}">{status_text}</span>
                    </div>
                    <p style="margin: 0; color: #6b7280">
                        {profile.get('target_departure_month', 'Any month')} | 
                        {profile.get('trip_duration_days', 7)} days ±{profile.get('flexibility_days', 2)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Get latest price
                combos = get_latest_combinations(
                    profile['origin'], 
                    profile['destination'],
                    profile.get('target_departure_month')
                )
                if combos:
                    min_price = min(c['price'] for c in combos)
                    st.metric("Best Price", f"${min_price:.0f}")
                else:
                    st.metric("Best Price", "No data")
            
            with col3:
                if st.button("View", key=f"view_{profile['id']}"):
                    st.session_state['selected_profile'] = profile['id']
                    st.session_state['view_origin'] = profile['origin']
                    st.session_state['view_dest'] = profile['destination']
                    st.session_state['view_month'] = profile.get('target_departure_month')
            
            with col4:
                if st.button("Toggle", key=f"toggle_{profile['id']}"):
                    toggle_profile_active(profile['id'], not is_active)
                    st.rerun()
    
    # Show selected profile details
    if st.session_state.get('selected_profile'):
        st.markdown("---")
        profile_id = st.session_state['selected_profile']
        profile = get_search_profile(profile_id)
        
        if profile:
            st.markdown(f"### Details: {profile['origin']} → {profile['destination']}")
            
            display_search_results(
                profile['origin'],
                profile['destination'],
                profile.get('target_departure_month', date.today().strftime('%Y-%m')),
                profile.get('trip_duration_days', 7),
                profile.get('flexibility_days', 2)
            )
            
            if st.button("Delete Profile", type="secondary"):
                delete_search_profile(profile_id)
                st.session_state['selected_profile'] = None
                st.rerun()


# =============================================================================
# PAGE 3: DECISION DASHBOARD
# =============================================================================

def page_decision_dashboard():
    """Decision engine dashboard with buy/wait recommendations."""
    
    st.markdown("""
    <div class="hero-header">
        <h1>Decision Dashboard</h1>
        <p>Intelligent buy/wait recommendations powered by historical analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route selection
    col1, col2 = st.columns(2)
    
    with col1:
        airports = get_all_airport_codes()
        origin = st.selectbox(
            "From",
            airports,
            index=airports.index("JFK") if "JFK" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}",
            key="decision_origin"
        )
        
        dest = st.selectbox(
            "To",
            airports,
            index=airports.index("MEX") if "MEX" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}",
            key="decision_dest"
        )
    
    with col2:
        today = date.today()
        departure_date = st.date_input(
            "Departure Date",
            value=today + timedelta(days=60),
            min_value=today,
            key="decision_depart"
        )
        
        return_date = st.date_input(
            "Return Date",
            value=departure_date + timedelta(days=21),
            min_value=departure_date,
            key="decision_return"
        )
        
        risk_profile = st.selectbox(
            "Risk Profile",
            [RiskProfile.CONSERVATIVE, RiskProfile.MODERATE, RiskProfile.AGGRESSIVE],
            index=1,
            format_func=lambda x: x.value.capitalize()
        )
    
    # Get decision
    if st.button("Get Recommendation", type="primary", use_container_width=True):
        if origin == dest:
            st.error("Origin and destination must be different!")
            return
        
        with st.spinner("Analyzing..."):
            result = make_decision(
                origin=origin,
                destination=dest,
                departure_date=departure_date,
                return_date=return_date,
                risk_profile=risk_profile,
            )
        
        # Display decision
        decision_class = {
            Decision.BUY: "decision-buy",
            Decision.WAIT: "decision-wait",
            Decision.WATCH_CLOSELY: "decision-watch",
        }.get(result.decision, "decision-watch")
        
        decision_emoji = {
            Decision.BUY: "✅",
            Decision.WAIT: "⏳",
            Decision.WATCH_CLOSELY: "👀",
        }.get(result.decision, "❓")
        
        st.markdown(f"""
        <div class="{decision_class}">
            <h1 style="margin: 0; font-size: 3rem">{decision_emoji} {result.decision.value.upper().replace('_', ' ')}</h1>
            <p style="margin: 0.5rem 0 0; font-size: 1.25rem; opacity: 0.9">
                Confidence: {result.confidence:.0%} | Deadline: {result.deadline}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fair value and current price
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${result.current_price:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fair Value (P50)</div>
                <div class="metric-value">${result.fair_value:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price Percentile</div>
                <div class="metric-value">P{result.percentile:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rebound Risk</div>
                <div class="metric-value">{result.rebound_risk:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Regime indicator
        regime_classes = {
            PriceRegime.PROMO: ("regime-promo", "🎉 PROMOTIONAL PRICING"),
            PriceRegime.NORMAL: ("regime-normal", "📊 NORMAL MARKET"),
            PriceRegime.ELEVATED: ("regime-elevated", "⚠️ ELEVATED PRICES"),
            PriceRegime.SCARCITY: ("regime-scarcity", "🔴 SCARCITY PRICING"),
        }
        regime_class, regime_text = regime_classes.get(
            result.regime, 
            ("regime-normal", "📊 UNKNOWN")
        )
        
        st.markdown(f"""
        <div class="{regime_class}">
            <strong>{regime_text}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Reasoning
        st.markdown("### Analysis Reasoning")
        for reason in result.reasoning:
            st.markdown(f"""
            <div class="recommendation-card">
                {reason}
            </div>
            """, unsafe_allow_html=True)
        
        # Expected savings
        if result.expected_savings_if_wait > 0:
            st.info(f"💰 Expected savings if you wait: **${result.expected_savings_if_wait:.0f}** (but with risk)")
        
        # Fair value distribution visualization
        st.markdown("### Fair Value Distribution")
        
        try:
            fv = get_fair_value_distribution(
                origin, dest, departure_date, return_date
            )
            
            if fv and fv.p50 > 0:
                fig = go.Figure()
                
                # Add distribution bars
                percentiles = ['P10', 'P25', 'P50', 'P75', 'P90']
                values = [fv.p10, fv.p25, fv.p50, fv.p75, fv.p90]
                colors = ['#22c55e', '#86efac', '#6b7280', '#fbbf24', '#ef4444']
                
                fig.add_trace(go.Bar(
                    x=percentiles,
                    y=values,
                    marker_color=colors,
                    text=[f'${v:.0f}' for v in values],
                    textposition='outside',
                ))
                
                # Add current price line
                fig.add_hline(
                    y=result.current_price,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Current: ${result.current_price:.0f}",
                )
                
                fig.update_layout(
                    title="Historical Price Distribution",
                    xaxis_title="Percentile",
                    yaxis_title="Price ($)",
                    showlegend=False,
                    height=350,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient historical data to show distribution")
        except Exception as e:
            st.warning(f"Could not load fair value distribution: {e}")
        
        # Route classification
        st.markdown("### Route Classification")
        details = get_route_classification_details(origin, dest)
        
        archetype_colors = {
            'business': '#3b82f6',
            'leisure': '#22c55e',
            'mixed': '#f59e0b',
        }
        color = archetype_colors.get(details['archetype'], '#6b7280')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                <h3 style="margin: 0">{details['archetype_name'].upper()}</h3>
                <p style="margin: 0.5rem 0 0; opacity: 0.9">{origin} → {dest}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            for char in details['characteristics'][:3]:
                st.markdown(f"• {char}")
            st.markdown(f"**Recommendation:** {details['recommendation']}")
    
    # Show profile-based recommendations
    st.markdown("---")
    st.markdown("### Quick Recommendations for Tracked Profiles")
    
    profiles = get_search_profiles(active_only=True)
    
    if profiles:
        for profile in profiles[:5]:
            try:
                recs = get_recommendations_for_profile(
                    profile['id'],
                    limit=3,
                    risk_profile=RiskProfile.MODERATE,
                )
                
                if recs:
                    st.markdown(f"**{profile['origin']} → {profile['destination']}** ({profile.get('target_departure_month', 'Any')})")
                    
                    for rec in recs:
                        decision_emoji = {"buy": "✅", "wait": "⏳", "watch_closely": "👀"}.get(rec.decision.value, "❓")
                        st.markdown(f"""
                        <div class="recommendation-card">
                            {decision_emoji} <strong>{rec.departure_date} → {rec.return_date}</strong>: 
                            ${rec.current_price:.0f} (P{rec.percentile:.0f}) - {rec.decision.value.upper().replace('_', ' ')}
                        </div>
                        """, unsafe_allow_html=True)
            except Exception:
                continue
    else:
        st.info("Add some search profiles to see recommendations here!")


# =============================================================================
# PAGE 4: BACKTESTING
# =============================================================================

def page_backtesting():
    """Backtesting and strategy comparison page."""
    
    st.markdown("""
    <div class="hero-header">
        <h1>Backtesting Engine</h1>
        <p>Evaluate decision strategies against historical data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route selection
    col1, col2 = st.columns(2)
    
    with col1:
        airports = get_all_airport_codes()
        origin = st.selectbox(
            "Origin",
            airports,
            index=airports.index("JFK") if "JFK" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}",
            key="backtest_origin"
        )
        
        dest = st.selectbox(
            "Destination",
            airports,
            index=airports.index("MEX") if "MEX" in airports else 0,
            format_func=lambda x: f"{x} - {get_airport_name(x)}",
            key="backtest_dest"
        )
    
    with col2:
        risk_profile = st.selectbox(
            "Risk Profile",
            [RiskProfile.CONSERVATIVE, RiskProfile.MODERATE, RiskProfile.AGGRESSIVE],
            index=1,
            format_func=lambda x: x.value.capitalize(),
            key="backtest_risk"
        )
        
        run_backtest = st.button("Compare All Strategies", type="primary", use_container_width=True)
    
    if run_backtest:
        if origin == dest:
            st.error("Origin and destination must be different!")
            return
        
        with st.spinner("Running backtests... This may take a moment."):
            result = compare_strategies(origin, dest, risk_profile=risk_profile)
        
        if 'error' in result:
            st.warning(f"Could not run backtest: {result['error']}")
            return
        
        # Best strategy winner
        st.markdown(f"""
        <div class="backtest-winner">
            <h2 style="margin: 0">🏆 Best Strategy: {result['best_strategy']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Results table
        st.markdown("### Strategy Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart of average regret
            if result.get('results'):
                strategies = []
                avg_regrets = []
                accuracies = []
                
                for name, data in result['results'].items():
                    strategies.append(name)
                    avg_regrets.append(data.avg_regret_per_decision)
                    accuracies.append(data.accuracy)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Avg Regret ($)',
                    x=strategies,
                    y=avg_regrets,
                    marker_color='#ef4444',
                ))
                
                fig.update_layout(
                    title='Average Regret by Strategy (Lower is Better)',
                    xaxis_title='Strategy',
                    yaxis_title='Average Regret ($)',
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Accuracy chart
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    name='Accuracy (%)',
                    x=strategies,
                    y=accuracies,
                    marker_color='#22c55e',
                ))
                
                fig2.update_layout(
                    title='Decision Accuracy by Strategy (Higher is Better)',
                    xaxis_title='Strategy',
                    yaxis_title='Accuracy (%)',
                    height=400,
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown("### Rankings")
            
            # By regret
            st.markdown("**By Lowest Regret:**")
            for r in result.get('ranking_by_regret', []):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r['rank'], f"{r['rank']}.")
                st.markdown(f"{medal} {r['strategy']}: ${r['avg_regret']:.0f}")
            
            st.markdown("---")
            
            # By accuracy
            st.markdown("**By Highest Accuracy:**")
            for r in result.get('ranking_by_accuracy', []):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r['rank'], f"{r['rank']}.")
                st.markdown(f"{medal} {r['strategy']}: {r['accuracy']:.0f}%")
        
        # Detailed results
        st.markdown("### Detailed Results")
        
        for name, data in result.get('results', {}).items():
            with st.expander(f"{name} - {data.total_decisions} decisions"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Decisions", data.total_decisions)
                with col2:
                    st.metric("Accuracy", f"{data.accuracy:.1f}%")
                with col3:
                    st.metric("Avg Regret", f"${data.avg_regret_per_decision:.0f}")
                with col4:
                    st.metric("Total Regret", f"${data.total_regret:.0f}")
                
                st.metric("Savings vs Naive", f"${data.total_savings_vs_naive:.0f}")
    
    # Feedback summary section
    st.markdown("---")
    st.markdown("### Feedback Loop Summary")
    
    try:
        feedback = get_feedback_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            grade_color = {
                'A': '#22c55e',
                'B': '#86efac',
                'C': '#fbbf24',
                'D': '#f87171',
                'F': '#ef4444',
            }.get(feedback.get('grade', 'N/A'), '#6b7280')
            
            st.markdown(f"""
            <div style="background: {grade_color}; color: white; padding: 2rem; border-radius: 12px; text-align: center;">
                <h1 style="margin: 0; font-size: 4rem">{feedback.get('grade', 'N/A')}</h1>
                <p style="margin: 0.5rem 0 0">Score: {feedback.get('score', 0):.0f}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Assessment:** {feedback.get('assessment', 'No data')}")
        
        with col2:
            if feedback.get('score_factors'):
                st.markdown("**Score Factors:**")
                for factor in feedback['score_factors']:
                    st.markdown(f"• {factor}")
            
            if feedback.get('accuracy', {}).get('overall_accuracy'):
                st.metric("Overall Accuracy", f"{feedback['accuracy']['overall_accuracy']}%")
            
            if feedback.get('regret', {}).get('avg_regret_per_decision') is not None:
                st.metric("Avg Regret/Decision", f"${feedback['regret']['avg_regret_per_decision']:.0f}")
    except Exception as e:
        st.info(f"No feedback data available yet. Start tracking decisions to see performance metrics.")


# =============================================================================
# PAGE 5: INSIGHTS & ANALYTICS
# =============================================================================

def page_insights():
    """Insights and analytics page."""
    
    st.markdown("""
    <div class="hero-header">
        <h1>Insights & Analytics</h1>
        <p>Deep analysis of flight price trends and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Select route
    profiles = get_search_profiles(active_only=False)
    
    if not profiles:
        st.info("Add some search profiles first to see insights!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        profile_options = {
            p['id']: f"{p['origin']} → {p['destination']} ({p.get('target_departure_month', 'Any')})"
            for p in profiles
        }
        selected_id = st.selectbox(
            "Select Route",
            list(profile_options.keys()),
            format_func=lambda x: profile_options[x]
        )
    
    profile = get_search_profile(selected_id)
    
    if not profile:
        return
    
    origin = profile['origin']
    dest = profile['destination']
    target_month = profile.get('target_departure_month', date.today().strftime('%Y-%m'))
    trip_duration = profile.get('trip_duration_days', 7)
    flexibility = profile.get('flexibility_days', 2)
    
    # Get analysis
    analysis = analyze_comprehensive_calendar(origin, dest, target_month, trip_duration, flexibility)
    
    if not analysis.metadata.get('total_combinations_searched'):
        st.warning("No data available for this route. Run a calendar search first!")
        return
    
    # Price statistics
    st.markdown("### Price Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Minimum", f"${analysis.price_stats.get('min', 0):.0f}")
    with col2:
        st.metric("Maximum", f"${analysis.price_stats.get('max', 0):.0f}")
    with col3:
        st.metric("Average", f"${analysis.price_stats.get('avg', 0):.0f}")
    with col4:
        st.metric("Median", f"${analysis.price_stats.get('median', 0):.0f}")
    with col5:
        st.metric("Std Dev", f"${analysis.price_stats.get('std_dev', 0):.0f}")
    
    st.markdown("---")
    
    # Booking window analysis
    st.markdown("### Best Time to Book")
    
    if analysis.booking_windows:
        bw = analysis.booking_windows
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Optimal Booking Window</div>
                <div class="metric-value" style="font-size: 1.25rem">{bw.get('optimal_window', 'N/A')}</div>
                <div class="metric-delta">Avg price: ${bw.get('optimal_avg_price', 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if analysis.lead_time_details:
                lead_data = [
                    {
                        'bucket_name': b.bucket_name,
                        'avg_price': b.avg_price,
                        'min_price': b.min_price,
                        'sample_count': b.sample_count,
                        'min_days': b.min_days,
                    }
                    for b in analysis.lead_time_details
                ]
                fig = plot_lead_time_curve(lead_data, origin, dest)
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Weekday analysis
    st.markdown("### Best Days to Travel")
    
    if analysis.weekday_analysis:
        wa = analysis.weekday_analysis
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cheapest Day", wa.get('cheapest_departure_day', 'N/A'))
        with col2:
            st.metric("Most Expensive", wa.get('most_expensive_departure_day', 'N/A'))
        with col3:
            premium = wa.get('avg_weekend_premium', 0)
            st.metric("Weekend Premium", f"${premium:.0f}" if premium > 0 else "No premium")
        
        if analysis.weekday_details:
            weekday_data = [
                {
                    'weekday': w.weekday,
                    'avg_price': w.avg_price,
                    'min_price': w.min_price,
                    'sample_count': w.sample_count,
                }
                for w in analysis.weekday_details
            ]
            fig = plot_weekday_heatmap(weekday_data, origin, dest)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Fair Value & Promo Analysis
    st.markdown("### Fair Value & Promotional Pricing")
    
    try:
        promos = find_promo_opportunities(origin, dest, target_month=target_month, z_threshold=-1.0)
        
        if promos:
            st.success(f"Found {len(promos)} promotional opportunities!")
            
            promo_df = pd.DataFrame(promos[:10])
            if not promo_df.empty and 'price' in promo_df.columns:
                fig = px.scatter(
                    promo_df,
                    x='departure_date',
                    y='price',
                    size='savings_pct',
                    color='percentile',
                    hover_data=['return_date', 'savings', 'fair_value'],
                    title='Promotional Pricing Opportunities',
                    color_continuous_scale='RdYlGn_r',
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No promotional pricing detected at this time.")
    except Exception:
        st.info("Insufficient baseline data for promo detection. Build baselines first.")
    
    st.markdown("---")
    
    # Duration comparison
    st.markdown("### Trip Duration Comparison")
    
    durations = [7, 14, 21, 28]
    comparison = compare_duration_options(origin, dest, target_month, durations)
    
    if comparison.get('durations'):
        df = pd.DataFrame(comparison['durations'])
        df = df[df['avg_price'].notna()]
        
        if not df.empty:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    df[['duration', 'min_price', 'avg_price', 'sample_count']].rename(columns={
                        'duration': 'Days',
                        'min_price': 'Min ($)',
                        'avg_price': 'Avg ($)',
                        'sample_count': 'Samples'
                    }),
                    hide_index=True
                )
            
            with col2:
                fig = px.bar(
                    df, x='duration', y='avg_price',
                    title='Average Price by Trip Duration',
                    labels={'duration': 'Trip Duration (days)', 'avg_price': 'Average Price ($)'},
                    color='avg_price',
                    color_continuous_scale=[[0, COLORS['cheap']], [1, COLORS['expensive']]]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if comparison.get('insights'):
            for insight in comparison['insights']:
                st.markdown(f"""
                <div class="recommendation-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Baseline coverage
    st.markdown("### Baseline Coverage")
    
    coverage = get_baseline_coverage(origin, dest)
    
    if coverage['has_baselines']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Baselines", "1" if coverage['has_overall'] else "0")
        with col2:
            st.metric("By Month", coverage['by_month_count'])
        with col3:
            st.metric("By Lead Time", coverage['by_lead_time_count'])
        
        if not coverage['has_overall']:
            if st.button("Build Baselines Now"):
                with st.spinner("Building baselines..."):
                    result = build_route_baselines(origin, dest, min_sample_count=10)
                st.success(f"Created {result.get('baselines_created', 0)} baselines!")
                st.rerun()
    else:
        st.warning("No baselines exist for this route. Build them to enable fair value analysis.")
        if st.button("Build Baselines"):
            with st.spinner("Building baselines..."):
                result = build_route_baselines(origin, dest, min_sample_count=10)
            st.success(f"Created {result.get('baselines_created', 0)} baselines!")
            st.rerun()
    
    st.markdown("---")
    
    # Pattern detection
    st.markdown("### Detected Patterns")
    
    patterns = detect_price_patterns(origin, dest, target_month)
    
    if patterns.get('patterns'):
        for pattern in patterns['patterns']:
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{pattern['type'].replace('_', ' ').title()}</strong> 
                (Impact: {pattern['impact'].upper()})<br>
                {pattern['description']}
                {f"<br><em>Tip: {pattern.get('recommendation', '')}</em>" if pattern.get('recommendation') else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to detect patterns yet. Keep tracking!")


# =============================================================================
# MAIN
# =============================================================================

# Initialize session state
if 'selected_profile' not in st.session_state:
    st.session_state['selected_profile'] = None

# Route to correct page
if page == "Calendar Search":
    page_calendar_search()
elif page == "Decision Dashboard":
    page_decision_dashboard()
elif page == "Tracking Dashboard":
    page_tracking_dashboard()
elif page == "Backtesting":
    page_backtesting()
elif page == "Insights & Analytics":
    page_insights()
