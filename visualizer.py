"""
Chart generation for Flight Price Tracker.

Creates visualizations using matplotlib (static) and plotly (interactive).
New architecture: Calendar heatmaps, price matrices, and flexibility comparisons.
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any
import io
import base64

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLORS
from utils import format_currency, get_weekday_name, parse_date_range, get_dates_in_range

# Configure logging
logger = logging.getLogger(__name__)

# Set matplotlib style (with fallback for older versions)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def get_price_color(price: float, min_price: float, max_price: float) -> str:
    """Get a color based on price relative to min/max range."""
    if max_price == min_price:
        return COLORS["moderate"]
    
    ratio = (price - min_price) / (max_price - min_price)
    
    if ratio < 0.33:
        return COLORS["cheap"]
    elif ratio < 0.66:
        return COLORS["moderate"]
    else:
        return COLORS["expensive"]


def get_color_scale() -> list:
    """Get a custom color scale for heatmaps (green=cheap, red=expensive)."""
    return [
        [0.0, COLORS["cheap"]],
        [0.5, COLORS["moderate"]],
        [1.0, COLORS["expensive"]],
    ]


# =============================================================================
# NEW CALENDAR VISUALIZATIONS
# =============================================================================

def plot_full_calendar_heatmap(
    origin: str,
    dest: str,
    year_month: str,
    trip_duration_days: int,
    heatmap_data: Dict[str, Any]
) -> go.Figure:
    """
    Interactive calendar heatmap where each cell = departure date.
    Color = minimum price for that departure.
    Hover shows price details.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        year_month: Target month (YYYY-MM)
        trip_duration_days: Trip duration
        heatmap_data: Data from generate_calendar_heatmap_data()
        
    Returns:
        Plotly Figure object
    """
    if not heatmap_data or not heatmap_data.get('dates'):
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this month",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    dates = heatmap_data['dates']
    prices = heatmap_data['prices']
    
    # Parse dates and organize by week
    date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    
    # Create calendar layout
    # Rows = weeks, Columns = days of week (Mon-Sun)
    weeks_data = {}
    for i, (d, price) in enumerate(zip(date_objs, prices)):
        week_num = d.isocalendar()[1]
        day_of_week = d.weekday()  # 0=Monday
        
        if week_num not in weeks_data:
            weeks_data[week_num] = [None] * 7
        
        weeks_data[week_num][day_of_week] = {
            'date': d,
            'price': price,
            'day': d.day
        }
    
    # Build heatmap data
    sorted_weeks = sorted(weeks_data.keys())
    z_data = []
    text_data = []
    hover_data = []
    
    for week in sorted_weeks:
        z_row = []
        text_row = []
        hover_row = []
        
        for day_idx in range(7):
            cell = weeks_data[week][day_idx]
            if cell:
                price = cell['price']
                z_row.append(price if price else np.nan)
                text_row.append(f"{cell['day']}")
                if price:
                    hover_row.append(
                        f"{cell['date'].strftime('%B %d, %Y')}<br>"
                        f"${price:.0f}<br>"
                        f"{trip_duration_days}-day trip"
                    )
                else:
                    hover_row.append(f"{cell['date'].strftime('%B %d, %Y')}<br>No data")
            else:
                z_row.append(np.nan)
                text_row.append("")
                hover_row.append("")
        
        z_data.append(z_row)
        text_data.append(text_row)
        hover_data.append(hover_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=[f'Week {w}' for w in sorted_weeks],
        text=text_data,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertext=hover_data,
        hoverinfo='text',
        colorscale=[[0, COLORS["cheap"]], [0.5, COLORS["moderate"]], [1, COLORS["expensive"]]],
        colorbar=dict(
            title="Price ($)",
            tickprefix="$"
        ),
        showscale=True,
    ))
    
    # Add min/max annotations
    min_price = heatmap_data.get('min_price')
    max_price = heatmap_data.get('max_price')
    avg_price = heatmap_data.get('avg_price')
    
    title_text = f"Flight Calendar: {origin} → {dest} ({year_month})"
    if min_price and max_price:
        title_text += f"<br><sup>Min: ${min_price:.0f} | Max: ${max_price:.0f} | Avg: ${avg_price:.0f}</sup>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        xaxis_title="Day of Week",
        yaxis_title="",
        yaxis=dict(autorange='reversed'),
        height=400,
    )
    
    return fig


def plot_2d_price_matrix(
    origin: str,
    dest: str,
    target_month: str,
    matrix_data: List[Dict[str, Any]],
    trip_duration_days: int,
    flexibility_days: int = 2
) -> go.Figure:
    """
    2D heatmap with departure dates on X-axis, duration offsets on Y-axis.
    Color indicates price.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        target_month: Target month
        matrix_data: List of combination dicts from database
        trip_duration_days: Base trip duration
        flexibility_days: Flexibility range
        
    Returns:
        Plotly Figure object
    """
    if not matrix_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Group data by departure date and duration offset
    start_date, end_date = parse_date_range(target_month)
    departure_dates = get_dates_in_range(start_date, end_date)
    offsets = list(range(-flexibility_days, flexibility_days + 1))
    
    # Build price matrix
    price_dict = {}
    for combo in matrix_data:
        dep = combo['departure_date']
        if isinstance(dep, str):
            dep = datetime.strptime(dep, '%Y-%m-%d').date()
        
        duration = combo['trip_duration_days']
        offset = duration - trip_duration_days
        
        if abs(offset) <= flexibility_days:
            key = (str(dep), offset)
            price_dict[key] = combo['price']
    
    # Create z-data matrix
    z_data = []
    hover_data = []
    
    for offset in offsets:
        z_row = []
        hover_row = []
        
        for dep_date in departure_dates:
            key = (str(dep_date), offset)
            price = price_dict.get(key)
            
            z_row.append(price if price else np.nan)
            
            duration = trip_duration_days + offset
            if price:
                ret_date = dep_date + pd.Timedelta(days=duration)
                hover_row.append(
                    f"Depart: {dep_date.strftime('%b %d')}<br>"
                    f"Return: {ret_date.strftime('%b %d')}<br>"
                    f"Duration: {duration} days<br>"
                    f"Price: ${price:.0f}"
                )
            else:
                hover_row.append(f"No data")
        
        z_data.append(z_row)
        hover_data.append(hover_row)
    
    # Format x-axis labels (departure dates)
    x_labels = [d.strftime('%b %d') for d in departure_dates]
    
    # Format y-axis labels (duration offsets)
    y_labels = [
        f"{trip_duration_days + off} days ({'+' if off > 0 else ''}{off})"
        for off in offsets
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        hovertext=hover_data,
        hoverinfo='text',
        colorscale=[[0, COLORS["cheap"]], [0.5, COLORS["moderate"]], [1, COLORS["expensive"]]],
        colorbar=dict(
            title="Price ($)",
            tickprefix="$"
        ),
    ))
    
    # Find and mark cheapest combination
    if price_dict:
        min_price = min(price_dict.values())
        for (dep_str, offset), price in price_dict.items():
            if price == min_price:
                # Add star marker for cheapest
                dep_idx = x_labels.index(
                    datetime.strptime(dep_str, '%Y-%m-%d').strftime('%b %d')
                ) if datetime.strptime(dep_str, '%Y-%m-%d').strftime('%b %d') in x_labels else -1
                offset_idx = offsets.index(offset) if offset in offsets else -1
                
                if dep_idx >= 0 and offset_idx >= 0:
                    fig.add_annotation(
                        x=x_labels[dep_idx],
                        y=y_labels[offset_idx],
                        text="★",
                        showarrow=False,
                        font=dict(size=20, color="white"),
                    )
                break
    
    fig.update_layout(
        title=dict(
            text=f"Price Matrix: {origin} → {dest} ({target_month})<br>"
                 f"<sup>★ = Best deal</sup>",
            x=0.5
        ),
        xaxis_title="Departure Date",
        yaxis_title="Trip Duration",
        height=300 + (len(offsets) * 30),
    )
    
    return fig


def plot_flexibility_comparison(
    base_combo: Dict[str, Any],
    alternatives: List[Dict[str, Any]]
) -> go.Figure:
    """
    Bar chart comparing base travel dates vs shifted alternatives.
    
    Args:
        base_combo: Base combination dict with price
        alternatives: List of alternative combinations with savings
        
    Returns:
        Plotly Figure object
    """
    if not alternatives:
        fig = go.Figure()
        fig.add_annotation(text="No alternatives available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Prepare data
    base_price = base_combo.get('base_price', 0)
    
    # Sort by savings
    sorted_alts = sorted(alternatives, key=lambda x: x.get('savings', 0), reverse=True)[:10]
    
    labels = ['Your dates']
    prices = [base_price]
    savings = [0]
    colors = [COLORS["primary"]]
    
    for alt in sorted_alts:
        label = alt.get('recommendation', '').split(':')[0] if alt.get('recommendation') else 'Alternative'
        labels.append(label[:25])  # Truncate long labels
        prices.append(alt.get('price', base_price))
        savings.append(alt.get('savings', 0))
        
        if alt.get('savings', 0) > 0:
            colors.append(COLORS["cheap"])
        elif alt.get('savings', 0) < 0:
            colors.append(COLORS["expensive"])
        else:
            colors.append(COLORS["moderate"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=prices,
        marker_color=colors,
        text=[f"${p:.0f}" for p in prices],
        textposition='outside',
        hovertemplate="%{x}<br>Price: $%{y:.0f}<extra></extra>",
    ))
    
    # Add savings annotations
    annotations = []
    for i, (save, price) in enumerate(zip(savings, prices)):
        if save > 0:
            annotations.append(dict(
                x=labels[i],
                y=price,
                text=f"Save ${save:.0f}",
                showarrow=False,
                yshift=-20,
                font=dict(color=COLORS["cheap"], size=10),
            ))
        elif save < 0:
            annotations.append(dict(
                x=labels[i],
                y=price,
                text=f"+${abs(save):.0f}",
                showarrow=False,
                yshift=-20,
                font=dict(color=COLORS["expensive"], size=10),
            ))
    
    fig.update_layout(
        title="Flexible Date Savings Comparison",
        xaxis_title="",
        yaxis_title="Price ($)",
        xaxis=dict(tickangle=45),
        annotations=annotations,
        height=400,
    )
    
    return fig


def plot_booking_timeline(
    origin: str,
    dest: str,
    price_history: List[Dict[str, Any]],
    optimal_buy_date: Optional[date] = None
) -> go.Figure:
    """
    Line chart showing price history for a specific combo over time.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        price_history: List of dicts with scrape_date and price
        optimal_buy_date: Optional date to mark as optimal buy point
        
    Returns:
        Plotly Figure object
    """
    if not price_history:
        fig = go.Figure()
        fig.add_annotation(text="No price history available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df = pd.DataFrame(price_history)
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    df = df.sort_values('scrape_date')
    
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=df['scrape_date'],
        y=df['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.0f}<extra></extra>",
    ))
    
    # Add average line
    avg_price = df['price'].mean()
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"Avg: ${avg_price:.0f}",
    )
    
    # Mark minimum price
    min_idx = df['price'].idxmin()
    min_row = df.loc[min_idx]
    
    fig.add_trace(go.Scatter(
        x=[min_row['scrape_date']],
        y=[min_row['price']],
        mode='markers',
        name='Lowest Price',
        marker=dict(size=15, color=COLORS["cheap"], symbol='star'),
        hovertemplate=f"Lowest: ${min_row['price']:.0f}<extra></extra>",
    ))
    
    # Mark optimal buy date if provided
    if optimal_buy_date:
        fig.add_vline(
            x=optimal_buy_date,
            line_dash="dot",
            line_color=COLORS["cheap"],
            annotation_text="Optimal Buy Date",
            annotation_position="top right",
        )
    
    fig.update_layout(
        title=f"Price History: {origin} → {dest}",
        xaxis_title="Date Checked",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=350,
    )
    
    return fig


def plot_combo_comparison_grid(
    top_combos: List[Dict[str, Any]],
    origin: str,
    dest: str
) -> go.Figure:
    """
    Table/grid showing top N combinations with details.
    
    Args:
        top_combos: List of top combinations (CombinationRanking or dicts)
        origin: Origin airport code
        dest: Destination airport code
        
    Returns:
        Plotly Figure object
    """
    if not top_combos:
        fig = go.Figure()
        fig.add_annotation(text="No combinations available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Prepare table data
    ranks = []
    departures = []
    returns = []
    durations = []
    prices = []
    vs_avgs = []
    airlines = []
    
    for combo in top_combos[:10]:
        if hasattr(combo, 'rank'):
            # CombinationRanking object
            ranks.append(f"#{combo.rank}")
            departures.append(str(combo.departure_date))
            returns.append(str(combo.return_date))
            durations.append(f"{combo.trip_duration}d")
            prices.append(f"${combo.price:.0f}")
            vs_avgs.append(f"{combo.vs_avg_percent:+.0f}%" if combo.vs_avg_percent else "-")
            airlines.append(combo.airline_outbound or "-")
        else:
            # Dict
            ranks.append(f"#{combo.get('rank', '-')}")
            departures.append(str(combo.get('departure_date', '-')))
            returns.append(str(combo.get('return_date', '-')))
            durations.append(f"{combo.get('trip_duration', '-')}d")
            prices.append(f"${combo.get('price', 0):.0f}")
            vs_avgs.append(f"{combo.get('vs_avg_percent', 0):+.0f}%" if combo.get('vs_avg_percent') else "-")
            airlines.append(combo.get('airline_outbound') or "-")
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Depart', 'Return', 'Duration', 'Price', 'vs Avg', 'Airline'],
            fill_color=COLORS["primary"],
            font=dict(color='white', size=12),
            align='center',
        ),
        cells=dict(
            values=[ranks, departures, returns, durations, prices, vs_avgs, airlines],
            fill_color=[
                ['white'] * len(ranks),
                ['white'] * len(ranks),
                ['white'] * len(ranks),
                ['white'] * len(ranks),
                [COLORS["cheap"] if i == 0 else 'white' for i in range(len(ranks))],
                [
                    COLORS["cheap"] if float(v.replace('%', '').replace('+', '')) < -10 
                    else COLORS["expensive"] if float(v.replace('%', '').replace('+', '')) > 10 
                    else 'white' 
                    for v in vs_avgs
                ] if all(v != '-' for v in vs_avgs) else ['white'] * len(ranks),
                ['white'] * len(ranks),
            ],
            align='center',
            font=dict(size=11),
            height=30,
        )
    )])
    
    fig.update_layout(
        title=f"Top 10 Deals: {origin} → {dest}",
        height=400,
    )
    
    return fig


def plot_weekday_heatmap(
    weekday_data: List[Dict[str, Any]],
    origin: str,
    dest: str
) -> go.Figure:
    """
    Enhanced weekday comparison with heatmap style.
    
    Args:
        weekday_data: List of WeekdayAnalysis dicts
        origin: Origin airport
        dest: Destination airport
        
    Returns:
        Plotly Figure object
    """
    if not weekday_data:
        fig = go.Figure()
        fig.add_annotation(text="No weekday data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df = pd.DataFrame(weekday_data)
    
    # Ensure correct order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
    df = df.sort_values('weekday')
    
    min_price = df['avg_price'].min()
    max_price = df['avg_price'].max()
    
    colors = [get_price_color(p, min_price, max_price) for p in df['avg_price']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['weekday'],
        y=df['avg_price'],
        marker_color=colors,
        text=[f"${p:.0f}" for p in df['avg_price']],
        textposition='outside',
        hovertemplate=(
            "%{x}<br>"
            "Avg: $%{y:.0f}<br>"
            "Min: $%{customdata[0]:.0f}<br>"
            "Samples: %{customdata[1]}<extra></extra>"
        ),
        customdata=list(zip(df['min_price'], df['sample_count'])),
    ))
    
    # Highlight cheapest day
    cheapest_idx = df['avg_price'].idxmin()
    cheapest_day = df.loc[cheapest_idx, 'weekday']
    cheapest_price = df.loc[cheapest_idx, 'avg_price']
    
    fig.add_annotation(
        x=cheapest_day,
        y=cheapest_price,
        text="Best Day!",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["cheap"],
        font=dict(color=COLORS["cheap"]),
        yshift=30,
    )
    
    fig.update_layout(
        title=f"Best Days to Fly: {origin} → {dest}",
        xaxis_title="Departure Day",
        yaxis_title="Average Price ($)",
        height=350,
    )
    
    return fig


def plot_lead_time_curve(
    lead_time_data: List[Dict[str, Any]],
    origin: str,
    dest: str
) -> go.Figure:
    """
    Enhanced lead time analysis with optimal booking window highlight.
    
    Args:
        lead_time_data: List of LeadTimeBucket dicts
        origin: Origin airport
        dest: Destination airport
        
    Returns:
        Plotly Figure object
    """
    if not lead_time_data:
        fig = go.Figure()
        fig.add_annotation(text="No lead time data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df = pd.DataFrame(lead_time_data)
    
    # Sort by min_days
    df = df.sort_values('min_days')
    
    # Find optimal bucket
    optimal_idx = df['avg_price'].idxmin()
    
    colors = [
        COLORS["cheap"] if i == optimal_idx else COLORS["primary"]
        for i in df.index
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['bucket_name'],
        y=df['avg_price'],
        marker_color=colors,
        text=[f"${p:.0f}" for p in df['avg_price']],
        textposition='outside',
        hovertemplate=(
            "%{x}<br>"
            "Avg: $%{y:.0f}<br>"
            "Min: $%{customdata[0]:.0f}<br>"
            "Samples: %{customdata[1]}<extra></extra>"
        ),
        customdata=list(zip(df['min_price'], df['sample_count'])),
    ))
    
    # Add annotation for optimal window
    optimal_bucket = df.loc[optimal_idx, 'bucket_name']
    optimal_price = df.loc[optimal_idx, 'avg_price']
    
    fig.add_annotation(
        x=optimal_bucket,
        y=optimal_price,
        text="Best Time to Book!",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["cheap"],
        font=dict(color=COLORS["cheap"]),
        yshift=30,
    )
    
    fig.update_layout(
        title=f"When to Book: {origin} → {dest}",
        xaxis_title="Days Before Departure",
        yaxis_title="Average Price ($)",
        xaxis=dict(tickangle=45),
        height=400,
    )
    
    return fig


# =============================================================================
# LEGACY CHARTS (kept for compatibility)
# =============================================================================

def plot_price_over_time(
    data: list,
    title: str = "Price Over Time",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a line chart of prices over time."""
    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    df = pd.DataFrame(data)
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    df = df.sort_values('scrape_date')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['scrape_date'], df['price'], color=COLORS["primary"],
            linewidth=2, marker='o', markersize=4)
    
    avg_price = df['price'].mean()
    ax.axhline(y=avg_price, color=COLORS["secondary"], linestyle='--', 
               label=f'Average: ${avg_price:.2f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weekday_comparison(
    weekday_data: list,
    title: str = "Average Price by Day of Week",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a bar chart comparing prices by weekday."""
    if not weekday_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        return fig
    
    df = pd.DataFrame(weekday_data)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
    df = df.sort_values('weekday')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    min_price = df['avg_price'].min()
    max_price = df['avg_price'].max()
    colors = [get_price_color(p, min_price, max_price) for p in df['avg_price']]
    
    bars = ax.bar(df['weekday'], df['avg_price'], color=colors, edgecolor='white', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Average Price ($)', fontsize=12)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_interactive_price_trend(data: list, title: str = "Price Trend") -> go.Figure:
    """Create an interactive line chart with hover details."""
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df = pd.DataFrame(data)
    
    date_col = 'departure_date' if 'departure_date' in df.columns else 'scrape_date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
    ))
    
    avg_price = df['price'].mean()
    fig.add_hline(y=avg_price, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text=f"Avg: ${avg_price:.0f}")
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)",
                      hovermode="x unified")
    
    return fig


def create_dashboard_charts(analysis_result) -> dict:
    """Create all charts needed for the Streamlit dashboard."""
    charts = {}
    
    # Weekday comparison
    if hasattr(analysis_result, 'weekday_details') and analysis_result.weekday_details:
        weekday_data = [
            {
                'weekday': wa.weekday,
                'avg_price': wa.avg_price,
                'min_price': wa.min_price,
                'sample_count': wa.sample_count,
            }
            for wa in analysis_result.weekday_details
        ]
        charts['weekday'] = plot_weekday_heatmap(
            weekday_data, analysis_result.origin, analysis_result.destination
        )
    
    # Lead time analysis
    if hasattr(analysis_result, 'lead_time_details') and analysis_result.lead_time_details:
        lead_data = [
            {
                'bucket_name': b.bucket_name,
                'avg_price': b.avg_price,
                'min_price': b.min_price,
                'sample_count': b.sample_count,
                'min_days': b.min_days,
            }
            for b in analysis_result.lead_time_details
        ]
        charts['lead_time'] = plot_lead_time_curve(
            lead_data, analysis_result.origin, analysis_result.destination
        )
    
    return charts


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to base64 string for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def save_charts_as_pdf(charts: list, output_path: str, title: str = "Flight Analysis Report") -> bool:
    """Save multiple charts to a PDF file."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    try:
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, title, ha='center', va='center', fontsize=24, fontweight='bold')
            fig.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            
            for chart in charts:
                pdf.savefig(chart, bbox_inches='tight')
        
        logger.info(f"Saved PDF report to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        return False
