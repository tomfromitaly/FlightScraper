#!/usr/bin/env python3
"""
Flight Price Tracker CLI

Command-line interface for comprehensive flight calendar search, tracking, and
intelligent buy/wait decisions.

Calendar & Tracking Commands:
    python main.py search-calendar JFK MEX --duration 21 --flex 2 --month 2026-10
    python main.py track JFK MEX --duration 21 --month 2026-10 --alert-drop 10
    python main.py collect-tracked
    python main.py analyze-combo JFK MEX --depart 2026-10-15 --return 2026-11-05
    python main.py find-savings JFK MEX --depart 2026-10-15 --return 2026-11-05
    python main.py export-calendar JFK MEX --month 2026-10

Decision Engine Commands:
    python main.py decision JFK MEX --depart 2026-10-15 --return 2026-11-05 --risk moderate
    python main.py baseline-build JFK MEX
    python main.py baseline-build --all
    python main.py backtest JFK MEX --strategy decision_engine
    python main.py backtest JFK MEX --compare
    python main.py promos JFK MEX --month 2026-10
    python main.py classify-route JFK MEX
    python main.py feedback

General Commands:
    python main.py init
    python main.py stats
    python main.py status
    python main.py dashboard
"""

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


# =============================================================================
# NEW CALENDAR COMMANDS
# =============================================================================

def cmd_search_calendar(args):
    """Comprehensive calendar search for all combinations."""
    from config import validate_iata_code
    from data_fetcher import fetch_comprehensive_calendar, estimate_search_time
    from database import insert_flight_combinations_batch
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    # Validate airports
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    # Estimate time
    estimate = estimate_search_time(args.month, args.duration, args.flex)
    console.print(f"\n[cyan]Searching {origin} → {dest} for {args.month}[/cyan]")
    console.print(f"Duration: {args.duration} days (±{args.flex})")
    console.print(f"Total combinations: {estimate['total_combinations']}")
    console.print(f"Estimated time: {estimate['estimated_seconds']:.0f} seconds\n")
    
    # Confirm
    if not args.yes:
        confirm = input("Continue? [y/N] ")
        if confirm.lower() != 'y':
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    # Run search with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=100)
        
        def update_progress(pct, msg):
            progress.update(task, completed=pct, description=msg)
        
        results = fetch_comprehensive_calendar(
            origin=origin,
            destination=dest,
            target_month=args.month,
            trip_duration_days=args.duration,
            flexibility_days=args.flex,
            progress_callback=update_progress,
        )
    
    # Save to database
    if results['combinations']:
        combos_to_save = [c.to_dict() for c in results['combinations']]
        saved = insert_flight_combinations_batch(combos_to_save)
        
        console.print(f"\n[bold green]✓ Found {results['success_count']} combinations[/bold green]")
        console.print(f"Saved {saved} to database")
        
        # Show summary
        prices = [c.price for c in results['combinations']]
        
        table = Table(title="Search Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Combinations found", str(len(prices)))
        table.add_row("Min price", format_currency(min(prices)))
        table.add_row("Max price", format_currency(max(prices)))
        table.add_row("Avg price", format_currency(sum(prices)/len(prices)))
        table.add_row("Failed searches", str(len(results['failed'])))
        console.print(table)
        
        # Show best deals
        sorted_combos = sorted(results['combinations'], key=lambda c: c.price)[:5]
        console.print("\n[bold]Top 5 Deals:[/bold]")
        for i, combo in enumerate(sorted_combos, 1):
            console.print(
                f"  {i}. {combo.departure_date} → {combo.return_date}: "
                f"[green]{format_currency(combo.price)}[/green]"
            )
    else:
        console.print("[bold yellow]⚠ No combinations found[/bold yellow]")


def cmd_track(args):
    """Add a search profile to tracking."""
    from config import validate_iata_code
    from database import create_search_profile
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    profile = {
        'profile_name': args.name or f"{origin}-{dest} {args.month}",
        'origin': origin,
        'destination': dest,
        'target_departure_month': args.month,
        'trip_duration_days': args.duration,
        'flexibility_days': args.flex,
        'max_price_threshold': args.max_price,
        'notify_on_drop_percent': args.alert_drop or 10.0,
        'is_active': True,
    }
    
    result = create_search_profile(profile)
    
    if result:
        console.print(
            f"[bold green]✓ Created tracking profile: {origin} → {dest}[/bold green]"
        )
        console.print(f"  Month: {args.month}")
        console.print(f"  Duration: {args.duration} days (±{args.flex})")
        if args.max_price:
            console.print(f"  Max price threshold: ${args.max_price:.2f}")
        console.print(f"  Alert on: {args.alert_drop or 10}% price drop")
    else:
        console.print("[bold red]✗ Failed to create profile (may already exist)[/bold red]")
        sys.exit(1)


def cmd_collect_tracked(args):
    """Run collection for all active tracking profiles."""
    from database import get_search_profiles, update_profile_last_searched, insert_flight_combinations_batch
    from data_fetcher import fetch_comprehensive_calendar, test_api_connection
    
    # Test API
    console.print("[yellow]Testing API connection...[/yellow]")
    api_status = test_api_connection()
    
    if not api_status.get("connected"):
        console.print(f"[bold red]✗ API error: {api_status.get('message')}[/bold red]")
        sys.exit(1)
    
    # Get profiles
    profiles = get_search_profiles(active_only=True)
    
    if not profiles:
        console.print("[bold yellow]No active tracking profiles. Add some with 'track'[/bold yellow]")
        return
    
    console.print(f"\n[bold]Collecting for {len(profiles)} profiles...[/bold]")
    
    total_combos = 0
    
    for i, profile in enumerate(profiles, 1):
        origin = profile['origin']
        dest = profile['destination']
        month = profile.get('target_departure_month', date.today().strftime('%Y-%m'))
        duration = profile.get('trip_duration_days', 7)
        flex = profile.get('flexibility_days', 2)
        
        console.print(f"\n[cyan]({i}/{len(profiles)}) {origin} → {dest} ({month})[/cyan]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Searching", total=100)
                
                def update_progress(pct, msg):
                    progress.update(task, completed=pct)
                
                results = fetch_comprehensive_calendar(
                    origin=origin,
                    destination=dest,
                    target_month=month,
                    trip_duration_days=duration,
                    flexibility_days=flex,
                    progress_callback=update_progress,
                )
            
            if results['combinations']:
                combos_to_save = [c.to_dict() for c in results['combinations']]
                saved = insert_flight_combinations_batch(combos_to_save)
                total_combos += saved
                console.print(f"  [green]✓ {saved} combinations saved[/green]")
                
                # Update last searched
                update_profile_last_searched(profile['id'])
            else:
                console.print(f"  [yellow]⚠ No data found[/yellow]")
                
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
    
    console.print(f"\n[bold green]✓ Collection complete: {total_combos} total combinations[/bold green]")


def cmd_analyze_combo(args):
    """Analyze a specific departure-return combination."""
    from config import validate_iata_code
    from calendar_builder import find_optimal_buy_date
    from database import get_price_history_for_combo
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    depart = datetime.strptime(args.depart, '%Y-%m-%d').date()
    ret = datetime.strptime(args.return_date, '%Y-%m-%d').date()
    
    console.print(f"\n[cyan]Analyzing: {origin} → {dest}[/cyan]")
    console.print(f"Dates: {depart} → {ret} ({(ret - depart).days} days)\n")
    
    # Get price history
    history = get_price_history_for_combo(origin, dest, depart, ret)
    
    if history:
        prices = [h['price'] for h in history]
        
        table = Table(title="Price History")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Data points", str(len(history)))
        table.add_row("Min price seen", format_currency(min(prices)))
        table.add_row("Max price seen", format_currency(max(prices)))
        table.add_row("Current/Latest", format_currency(prices[-1]))
        console.print(table)
    else:
        console.print("[yellow]No price history for this specific combination[/yellow]")
    
    # Get optimal buy date
    optimal = find_optimal_buy_date(origin, dest, depart, ret)
    
    console.print(Panel.fit(
        f"[bold]Optimal Booking Window[/bold]\n"
        f"Book around: {optimal.recommended_buy_date}\n"
        f"Lead time: ~{optimal.optimal_lead_days} days before departure\n"
        f"Confidence: {optimal.confidence}\n"
        f"Price trend: {optimal.price_trend}",
        title="When to Book",
        border_style="green",
    ))


def cmd_find_savings(args):
    """Find flexibility savings for specific dates."""
    from config import validate_iata_code
    from calendar_builder import calculate_flexibility_savings, get_flexibility_recommendations
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    depart = datetime.strptime(args.depart, '%Y-%m-%d').date()
    ret = datetime.strptime(args.return_date, '%Y-%m-%d').date()
    
    console.print(f"\n[cyan]Finding savings for: {origin} → {dest}[/cyan]")
    console.print(f"Base dates: {depart} → {ret}\n")
    
    # Calculate savings
    savings = calculate_flexibility_savings(origin, dest, depart, ret, args.flex)
    
    if savings.base_price == 0:
        console.print("[yellow]No price data available for these dates[/yellow]")
        return
    
    console.print(f"[bold]Base price: {format_currency(savings.base_price)}[/bold]\n")
    
    if savings.max_savings > 0:
        console.print(Panel.fit(
            f"[bold green]Maximum savings: {format_currency(savings.max_savings)}[/bold green]\n"
            f"Average savings if flexible: {format_currency(savings.avg_savings_if_flexible)}",
            title="Flexibility Bonus",
            border_style="green",
        ))
    
    # Show alternatives
    if savings.alternatives:
        table = Table(title="Alternative Dates")
        table.add_column("Departure", style="cyan")
        table.add_column("Return", style="cyan")
        table.add_column("Price", style="green")
        table.add_column("Savings", style="yellow")
        
        sorted_alts = sorted(savings.alternatives, key=lambda x: x['savings'], reverse=True)
        for alt in sorted_alts[:10]:
            savings_str = f"${alt['savings']:.0f}" if alt['savings'] > 0 else f"-${abs(alt['savings']):.0f}"
            color = "green" if alt['savings'] > 0 else "red"
            table.add_row(
                alt['departure'],
                alt['return'],
                format_currency(alt['price']),
                f"[{color}]{savings_str}[/{color}]"
            )
        
        console.print(table)
    
    # Recommendations
    recs = get_flexibility_recommendations(savings)
    if recs:
        console.print("\n[bold]💡 Recommendations:[/bold]")
        for rec in recs:
            console.print(f"  • {rec}")


def cmd_export_calendar(args):
    """Export calendar data to CSV."""
    from config import validate_iata_code
    from database import get_latest_combinations
    import csv
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    output_path = args.output or f"{origin}_{dest}_{args.month}_calendar.csv"
    
    console.print(f"[cyan]Exporting calendar for {origin} → {dest} ({args.month})...[/cyan]")
    
    combos = get_latest_combinations(origin, dest, args.month)
    
    if not combos:
        console.print("[yellow]No data to export[/yellow]")
        return
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'departure_date', 'return_date', 'trip_duration_days', 'price',
            'airline_outbound', 'airline_return', 'stops_outbound', 'stops_return',
            'departure_day_of_week', 'lead_time_days'
        ])
        writer.writeheader()
        for combo in combos:
            writer.writerow({
                'departure_date': combo['departure_date'],
                'return_date': combo['return_date'],
                'trip_duration_days': combo['trip_duration_days'],
                'price': combo['price'],
                'airline_outbound': combo.get('airline_outbound', ''),
                'airline_return': combo.get('airline_return', ''),
                'stops_outbound': combo.get('stops_outbound', 0),
                'stops_return': combo.get('stops_return', 0),
                'departure_day_of_week': combo.get('departure_day_of_week', ''),
                'lead_time_days': combo.get('lead_time_days', ''),
            })
    
    console.print(f"[bold green]✓ Exported {len(combos)} combinations to {output_path}[/bold green]")


def cmd_stats(args):
    """Show comprehensive database statistics."""
    from database import get_database_stats
    from utils import format_currency
    
    stats = get_database_stats()
    
    console.print(Panel.fit(
        "[bold]Flight Tracker Database Statistics[/bold]",
        border_style="blue",
    ))
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Combinations", f"{stats['total_combinations']:,}")
    table.add_row("Total Profiles", str(stats['total_profiles']))
    table.add_row("Active Profiles", str(stats['active_profiles']))
    table.add_row("Total Alerts", str(stats['total_alerts']))
    table.add_row("Unread Alerts", str(stats['unread_alerts']))
    table.add_row("Routes with Data", str(stats['routes_with_data']))
    
    if stats['date_range']['earliest']:
        table.add_row("Earliest Scrape", stats['date_range']['earliest'])
        table.add_row("Latest Scrape", stats['date_range']['latest'])
    
    console.print(table)
    
    if stats['months_with_data']:
        console.print(f"\n[cyan]Months with data:[/cyan] {', '.join(stats['months_with_data'][:6])}")


# =============================================================================
# LEGACY COMMANDS
# =============================================================================

def cmd_init(args):
    """Initialize the database."""
    from database import init_db, verify_db
    
    console.print("[bold blue]Initializing Flight Tracker database...[/bold blue]")
    
    success = init_db()
    
    if success:
        console.print("[bold green]✓ Database initialized successfully![/bold green]")
        stats = verify_db()
        
        table = Table(title="Database Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            if key != "tables":
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
    else:
        console.print("[bold red]✗ Database initialization failed![/bold red]")
        sys.exit(1)


def cmd_analyze(args):
    """Analyze a route."""
    from config import validate_iata_code
    from analyzer import analyze_comprehensive_calendar, generate_recommendations
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    month = args.month or date.today().strftime('%Y-%m')
    
    console.print(f"[cyan]Analyzing {origin} → {dest} for {month}[/cyan]\n")
    
    result = analyze_comprehensive_calendar(origin, dest, month, args.duration, args.flex)
    
    if not result.metadata.get('total_combinations_searched'):
        console.print("[bold yellow]No data available. Run 'search-calendar' first.[/bold yellow]")
        return
    
    # Price statistics
    console.print(Panel.fit(
        f"[bold]Analysis: {origin} → {dest}[/bold]\n"
        f"Month: {month} | Duration: {args.duration}d (±{args.flex})\n"
        f"Data coverage: {result.metadata.get('data_coverage_percent', 0):.0f}%",
        border_style="blue",
    ))
    
    stats_table = Table(title="Price Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Min Price", format_currency(result.price_stats.get('min', 0)))
    stats_table.add_row("Max Price", format_currency(result.price_stats.get('max', 0)))
    stats_table.add_row("Avg Price", format_currency(result.price_stats.get('avg', 0)))
    stats_table.add_row("Median", format_currency(result.price_stats.get('median', 0)))
    console.print(stats_table)
    
    # Best combo
    if result.best_combo:
        console.print(Panel.fit(
            f"[bold green]Best Deal[/bold green]\n"
            f"Dates: {result.best_combo['departure']} → {result.best_combo['return']}\n"
            f"Price: {format_currency(result.best_combo['price'])}\n"
            f"Book by: {result.best_combo.get('buy_date', 'ASAP')}",
            title="Recommended",
            border_style="green",
        ))
    
    # Recommendations
    recs = generate_recommendations(result)
    if recs:
        console.print("\n[bold]💡 Recommendations:[/bold]")
        for rec in recs[:5]:
            console.print(f"  • {rec}")


def cmd_list_profiles(args):
    """List all tracking profiles."""
    from database import get_search_profiles, get_latest_combinations
    from utils import format_currency
    
    profiles = get_search_profiles(active_only=not args.all)
    
    if not profiles:
        console.print("[bold yellow]No tracking profiles. Add some with 'track'[/bold yellow]")
        return
    
    table = Table(title="Tracking Profiles")
    table.add_column("ID", style="dim")
    table.add_column("Route", style="cyan")
    table.add_column("Month", style="yellow")
    table.add_column("Duration", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Best Price", style="green")
    
    for profile in profiles:
        route = f"{profile['origin']} → {profile['destination']}"
        status = "[green]Active[/green]" if profile['is_active'] else "[dim]Paused[/dim]"
        
        # Get best price
        combos = get_latest_combinations(
            profile['origin'],
            profile['destination'],
            profile.get('target_departure_month')
        )
        best_price = format_currency(min(c['price'] for c in combos)) if combos else "N/A"
        
        table.add_row(
            str(profile['id']),
            route,
            profile.get('target_departure_month') or "Any",
            f"{profile.get('trip_duration_days', 7)}d (±{profile.get('flexibility_days', 2)})",
            status,
            best_price,
        )
    
    console.print(table)


def cmd_status(args):
    """Show system status."""
    from database import verify_db, get_database_stats
    from data_fetcher import test_api_connection, get_api_remaining_quota
    
    console.print("[bold]Flight Tracker Status[/bold]\n")
    
    # Database
    console.print("[cyan]Database:[/cyan]")
    stats = get_database_stats()
    
    console.print(f"  Combinations: {stats['total_combinations']:,}")
    console.print(f"  Active profiles: {stats['active_profiles']}")
    console.print(f"  Unread alerts: {stats['unread_alerts']}")
    
    # API
    console.print("\n[cyan]Amadeus API:[/cyan]")
    api_status = test_api_connection()
    
    if api_status.get("connected"):
        console.print(f"  [green]✓ Connected[/green]")
        quota = get_api_remaining_quota()
        console.print(f"  Max parallel: {quota.get('max_parallel_calls', 3)} calls")
    else:
        console.print(f"  [red]✗ Error: {api_status.get('message')}[/red]")


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    console.print("[bold blue]Launching Flight Calendar Dashboard...[/bold blue]")
    
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        console.print("[bold red]✗ Dashboard app not found[/bold red]")
        sys.exit(1)
    
    try:
        subprocess.run(
            ["streamlit", "run", str(app_path), "--server.headless", "true"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]✗ Failed to launch dashboard: {e}[/bold red]")
        sys.exit(1)
    except FileNotFoundError:
        console.print("[bold red]✗ Streamlit not installed. Run: pip install streamlit[/bold red]")
        sys.exit(1)


# =============================================================================
# DECISION ENGINE COMMANDS
# =============================================================================

def cmd_decision(args):
    """Get BUY/WAIT/WATCH recommendation for a specific flight."""
    from config import validate_iata_code
    from decision_engine import make_decision
    from models import RiskProfile
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    depart = datetime.strptime(args.depart, '%Y-%m-%d').date()
    ret = datetime.strptime(args.return_date, '%Y-%m-%d').date()
    
    risk_profile = RiskProfile(args.risk)
    
    console.print(f"\n[cyan]Analyzing: {origin} → {dest}[/cyan]")
    console.print(f"Dates: {depart} → {ret}")
    console.print(f"Risk profile: {risk_profile.value}\n")
    
    result = make_decision(
        origin=origin,
        destination=dest,
        departure_date=depart,
        return_date=ret,
        current_price=args.price,
        risk_profile=risk_profile,
    )
    
    # Decision display
    decision_colors = {
        'buy': 'green',
        'wait': 'yellow',
        'watch_closely': 'blue',
    }
    decision_color = decision_colors.get(result.decision.value, 'white')
    
    console.print(Panel.fit(
        f"[bold {decision_color}]{result.decision.value.upper().replace('_', ' ')}[/bold {decision_color}]\n\n"
        f"Confidence: {result.confidence:.0%}\n"
        f"Deadline: {result.deadline}",
        title="RECOMMENDATION",
        border_style=decision_color,
    ))
    
    # Details
    table = Table(title="Analysis Details")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Current Price", format_currency(result.current_price))
    table.add_row("Fair Value (P50)", format_currency(result.fair_value))
    table.add_row("Percentile", f"P{result.percentile:.0f}")
    table.add_row("Regime", result.regime.value.upper())
    table.add_row("Rebound Risk", f"{result.rebound_risk:.0%}")
    
    if result.expected_savings_if_wait > 0:
        table.add_row("Expected Savings if Wait", format_currency(result.expected_savings_if_wait))
    
    console.print(table)
    
    # Reasoning
    console.print("\n[bold]Reasoning:[/bold]")
    for reason in result.reasoning:
        console.print(f"  • {reason}")


def cmd_baseline_build(args):
    """Build route baselines from historical data."""
    from config import validate_iata_code
    from baseline_builder import build_route_baselines, rebuild_all_baselines, get_baseline_coverage
    
    if args.all:
        console.print("[cyan]Rebuilding baselines for all routes...[/cyan]\n")
        
        result = rebuild_all_baselines(min_sample_count=args.min_samples)
        
        console.print(f"[bold green]✓ Created {result['total_baselines_created']} baselines[/bold green]")
        console.print(f"Routes processed: {result['routes_processed']}")
    else:
        origin = args.origin.upper()
        dest = args.dest.upper()
        
        if not validate_iata_code(origin) or not validate_iata_code(dest):
            console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
            sys.exit(1)
        
        console.print(f"[cyan]Building baselines for {origin} → {dest}...[/cyan]\n")
        
        result = build_route_baselines(
            origin, dest,
            min_sample_count=args.min_samples,
            rebuild=args.rebuild,
        )
        
        if result.get('error'):
            console.print(f"[bold red]✗ {result['error']}[/bold red]")
            return
        
        console.print(f"[bold green]✓ Created {result['baselines_created']} baselines[/bold green]")
        console.print(f"From {result['total_data_points']} data points")
        
        # Show coverage
        coverage = get_baseline_coverage(origin, dest)
        if coverage['has_baselines']:
            console.print(f"\n[cyan]Coverage:[/cyan]")
            console.print(f"  By month: {coverage['by_month_count']} baselines")
            console.print(f"  By lead time: {coverage['by_lead_time_count']} baselines")
            console.print(f"  By month+lead: {coverage['by_month_lead_count']} baselines")


def cmd_backtest(args):
    """Run backtesting for a strategy."""
    from config import validate_iata_code
    from backtester import backtest_strategy, compare_strategies
    from models import RiskProfile
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    risk_profile = RiskProfile(args.risk)
    
    if args.compare:
        console.print(f"\n[cyan]Comparing strategies for {origin} → {dest}[/cyan]\n")
        
        result = compare_strategies(origin, dest, risk_profile=risk_profile)
        
        if 'error' in result:
            console.print(f"[bold red]✗ {result['error']}[/bold red]")
            return
        
        console.print(f"[bold green]Best Strategy: {result['best_strategy']}[/bold green]\n")
        
        table = Table(title="Strategy Comparison (by Regret)")
        table.add_column("Rank", style="dim")
        table.add_column("Strategy", style="cyan")
        table.add_column("Avg Regret", style="yellow")
        table.add_column("Accuracy", style="green")
        table.add_column("Decisions", style="dim")
        
        for r in result['ranking_by_regret']:
            table.add_row(
                str(r['rank']),
                r['strategy'],
                format_currency(r['avg_regret']),
                f"{r['accuracy']:.1f}%",
                str(r['decisions'])
            )
        
        console.print(table)
    else:
        console.print(f"\n[cyan]Backtesting '{args.strategy}' for {origin} → {dest}[/cyan]\n")
        
        result = backtest_strategy(
            origin, dest,
            strategy_name=args.strategy,
            risk_profile=risk_profile,
        )
        
        if result.total_decisions == 0:
            console.print("[bold yellow]No data available for backtesting[/bold yellow]")
            return
        
        console.print(Panel.fit(
            f"[bold]Backtest Results: {args.strategy}[/bold]\n\n"
            f"Decisions tested: {result.total_decisions}\n"
            f"Accuracy: {result.accuracy}%\n"
            f"Average regret: {format_currency(result.avg_regret_per_decision)}\n"
            f"Total regret: {format_currency(result.total_regret)}\n"
            f"Savings vs naive: {format_currency(result.total_savings_vs_naive)}",
            border_style="blue",
        ))


def cmd_promos(args):
    """Find promotional pricing opportunities."""
    from config import validate_iata_code
    from quantile_model import find_promo_opportunities
    from utils import format_currency
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    console.print(f"\n[cyan]Finding promotional opportunities: {origin} → {dest}[/cyan]\n")
    
    opportunities = find_promo_opportunities(
        origin, dest,
        target_month=args.month,
        z_threshold=-1.0,
    )
    
    if not opportunities:
        console.print("[yellow]No promotional opportunities found[/yellow]")
        console.print("This may mean prices are currently normal or there's insufficient baseline data.")
        return
    
    console.print(f"[bold green]Found {len(opportunities)} deals![/bold green]\n")
    
    table = Table(title="Promotional Opportunities")
    table.add_column("Dates", style="cyan")
    table.add_column("Price", style="green")
    table.add_column("Fair Value", style="dim")
    table.add_column("Savings", style="yellow")
    table.add_column("Percentile", style="magenta")
    
    for opp in opportunities[:10]:
        table.add_row(
            f"{opp['departure_date']} → {opp['return_date']}",
            format_currency(opp['price']),
            format_currency(opp['fair_value']),
            f"{format_currency(opp['savings'])} ({opp['savings_pct']:.0f}%)",
            f"P{opp['percentile']:.0f}",
        )
    
    console.print(table)


def cmd_classify_route(args):
    """Classify a route as business/leisure/mixed."""
    from config import validate_iata_code
    from route_classifier import get_route_classification_details
    
    origin = args.origin.upper()
    dest = args.dest.upper()
    
    if not validate_iata_code(origin) or not validate_iata_code(dest):
        console.print("[bold red]✗ Invalid airport code(s)[/bold red]")
        sys.exit(1)
    
    details = get_route_classification_details(origin, dest)
    
    archetype_colors = {
        'business': 'blue',
        'leisure': 'green',
        'mixed': 'yellow',
    }
    color = archetype_colors.get(details['archetype'], 'white')
    
    console.print(Panel.fit(
        f"[bold {color}]{details['archetype_name'].upper()}[/bold {color}]\n\n"
        f"Route: {origin} → {dest}",
        title="Route Classification",
        border_style=color,
    ))
    
    console.print("\n[bold]Characteristics:[/bold]")
    for char in details['characteristics']:
        console.print(f"  • {char}")
    
    console.print(f"\n[bold]Recommendation:[/bold]")
    console.print(f"  {details['recommendation']}")
    
    if details.get('seasonal_analysis') and details['seasonal_analysis'].get('has_seasonal_data'):
        seasonal = details['seasonal_analysis']
        console.print(f"\n[cyan]Seasonal Insights:[/cyan]")
        console.print(f"  Cheapest month: {seasonal['cheapest_month']} (${seasonal['cheapest_month_median']:.0f})")
        console.print(f"  Most expensive: {seasonal['most_expensive_month']} (${seasonal['expensive_month_median']:.0f})")


def cmd_feedback(args):
    """Show feedback and outcome statistics."""
    from outcome_tracker import get_feedback_summary, get_decision_accuracy, get_regret_analysis
    from utils import format_currency
    
    summary = get_feedback_summary(args.profile)
    
    console.print(Panel.fit(
        f"[bold]Feedback Score: {summary['score']:.0f}/100[/bold]\n"
        f"Grade: {summary['grade']}\n\n"
        f"{summary['assessment']}",
        title="Performance Summary",
        border_style="blue",
    ))
    
    if summary['score_factors']:
        console.print("\n[cyan]Score Factors:[/cyan]")
        for factor in summary['score_factors']:
            console.print(f"  • {factor}")
    
    if summary['accuracy'].get('overall_accuracy'):
        acc = summary['accuracy']
        console.print(f"\n[cyan]Decision Accuracy:[/cyan]")
        console.print(f"  Overall: {acc['overall_accuracy']}%")
        if 'buy_decisions' in acc:
            console.print(f"  BUY decisions: {acc['buy_decisions']['accuracy']}%")
        if 'wait_decisions' in acc:
            console.print(f"  WAIT decisions: {acc['wait_decisions']['accuracy']}%")
    
    if summary['regret'].get('avg_regret_per_decision') is not None:
        reg = summary['regret']
        console.print(f"\n[cyan]Regret Analysis:[/cyan]")
        console.print(f"  Average regret: {format_currency(reg['avg_regret_per_decision'])}")
        console.print(f"  Total regret: {format_currency(reg['total_regret'])}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flight Price Tracker CLI - Comprehensive Calendar Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py search-calendar JFK MEX --duration 21 --month 2026-10
  python main.py track JFK MEX --duration 21 --month 2026-10
  python main.py collect-tracked
  python main.py analyze JFK MEX --month 2026-10
  python main.py find-savings JFK MEX --depart 2026-10-15 --return 2026-11-05
  python main.py dashboard
        """
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # search-calendar
    search_parser = subparsers.add_parser("search-calendar", help="Comprehensive calendar search")
    search_parser.add_argument("origin", help="Origin airport code")
    search_parser.add_argument("dest", help="Destination airport code")
    search_parser.add_argument("--duration", "-d", type=int, default=21, help="Trip duration (default: 21)")
    search_parser.add_argument("--flex", "-f", type=int, default=2, help="Flexibility ±days (default: 2)")
    search_parser.add_argument("--month", "-m", required=True, help="Target month (YYYY-MM)")
    search_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    search_parser.set_defaults(func=cmd_search_calendar)
    
    # track
    track_parser = subparsers.add_parser("track", help="Add tracking profile")
    track_parser.add_argument("origin", help="Origin airport code")
    track_parser.add_argument("dest", help="Destination airport code")
    track_parser.add_argument("--duration", "-d", type=int, default=21, help="Trip duration")
    track_parser.add_argument("--flex", "-f", type=int, default=2, help="Flexibility ±days")
    track_parser.add_argument("--month", "-m", required=True, help="Target month (YYYY-MM)")
    track_parser.add_argument("--name", "-n", help="Profile name")
    track_parser.add_argument("--max-price", type=float, help="Max price threshold")
    track_parser.add_argument("--alert-drop", type=float, help="Alert on % price drop")
    track_parser.set_defaults(func=cmd_track)
    
    # collect-tracked
    collect_tracked_parser = subparsers.add_parser("collect-tracked", help="Collect for all tracked profiles")
    collect_tracked_parser.set_defaults(func=cmd_collect_tracked)
    
    # analyze-combo
    analyze_combo_parser = subparsers.add_parser("analyze-combo", help="Analyze specific combination")
    analyze_combo_parser.add_argument("origin", help="Origin airport code")
    analyze_combo_parser.add_argument("dest", help="Destination airport code")
    analyze_combo_parser.add_argument("--depart", required=True, help="Departure date (YYYY-MM-DD)")
    analyze_combo_parser.add_argument("--return", dest="return_date", required=True, help="Return date (YYYY-MM-DD)")
    analyze_combo_parser.set_defaults(func=cmd_analyze_combo)
    
    # find-savings
    savings_parser = subparsers.add_parser("find-savings", help="Find flexibility savings")
    savings_parser.add_argument("origin", help="Origin airport code")
    savings_parser.add_argument("dest", help="Destination airport code")
    savings_parser.add_argument("--depart", required=True, help="Base departure date")
    savings_parser.add_argument("--return", dest="return_date", required=True, help="Base return date")
    savings_parser.add_argument("--flex", "-f", type=int, default=2, help="Flexibility range")
    savings_parser.set_defaults(func=cmd_find_savings)
    
    # export-calendar
    export_parser = subparsers.add_parser("export-calendar", help="Export calendar to CSV")
    export_parser.add_argument("origin", help="Origin airport code")
    export_parser.add_argument("dest", help="Destination airport code")
    export_parser.add_argument("--month", "-m", required=True, help="Target month")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export_calendar)
    
    # stats
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)
    
    # init
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.set_defaults(func=cmd_init)
    
    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a route")
    analyze_parser.add_argument("origin", help="Origin airport code")
    analyze_parser.add_argument("dest", help="Destination airport code")
    analyze_parser.add_argument("--month", "-m", help="Target month")
    analyze_parser.add_argument("--duration", "-d", type=int, default=21, help="Trip duration")
    analyze_parser.add_argument("--flex", "-f", type=int, default=2, help="Flexibility")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # list-profiles
    list_parser = subparsers.add_parser("list-profiles", help="List tracking profiles")
    list_parser.add_argument("--all", "-a", action="store_true", help="Include inactive")
    list_parser.set_defaults(func=cmd_list_profiles)
    
    # status
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)
    
    # dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dashboard_parser.set_defaults(func=cmd_dashboard)
    
    # =================================================================
    # DECISION ENGINE COMMANDS
    # =================================================================
    
    # decision
    decision_parser = subparsers.add_parser("decision", help="Get BUY/WAIT/WATCH recommendation")
    decision_parser.add_argument("origin", help="Origin airport code")
    decision_parser.add_argument("dest", help="Destination airport code")
    decision_parser.add_argument("--depart", required=True, help="Departure date (YYYY-MM-DD)")
    decision_parser.add_argument("--return", dest="return_date", required=True, help="Return date (YYYY-MM-DD)")
    decision_parser.add_argument("--price", type=float, help="Current price (optional, uses latest from DB)")
    decision_parser.add_argument("--risk", choices=['aggressive', 'moderate', 'conservative'], 
                                  default='moderate', help="Risk profile")
    decision_parser.set_defaults(func=cmd_decision)
    
    # baseline-build
    baseline_parser = subparsers.add_parser("baseline-build", help="Build route baselines")
    baseline_parser.add_argument("origin", nargs='?', help="Origin airport code")
    baseline_parser.add_argument("dest", nargs='?', help="Destination airport code")
    baseline_parser.add_argument("--all", action="store_true", help="Rebuild all routes")
    baseline_parser.add_argument("--rebuild", "-r", action="store_true", help="Force rebuild")
    baseline_parser.add_argument("--min-samples", type=int, default=10, help="Minimum samples")
    baseline_parser.set_defaults(func=cmd_baseline_build)
    
    # backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument("origin", help="Origin airport code")
    backtest_parser.add_argument("dest", help="Destination airport code")
    backtest_parser.add_argument("--strategy", "-s", 
                                  choices=['always_buy', 'below_p25', 'below_p50', 'optimal_window', 'decision_engine'],
                                  default='decision_engine', help="Strategy to test")
    backtest_parser.add_argument("--compare", "-c", action="store_true", help="Compare all strategies")
    backtest_parser.add_argument("--risk", choices=['aggressive', 'moderate', 'conservative'], 
                                  default='moderate', help="Risk profile")
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # promos
    promos_parser = subparsers.add_parser("promos", help="Find promotional pricing")
    promos_parser.add_argument("origin", help="Origin airport code")
    promos_parser.add_argument("dest", help="Destination airport code")
    promos_parser.add_argument("--month", "-m", help="Target month (YYYY-MM)")
    promos_parser.set_defaults(func=cmd_promos)
    
    # classify-route
    classify_parser = subparsers.add_parser("classify-route", help="Classify route archetype")
    classify_parser.add_argument("origin", help="Origin airport code")
    classify_parser.add_argument("dest", help="Destination airport code")
    classify_parser.set_defaults(func=cmd_classify_route)
    
    # feedback
    feedback_parser = subparsers.add_parser("feedback", help="View outcome feedback")
    feedback_parser.add_argument("--profile", type=int, help="Search profile ID")
    feedback_parser.set_defaults(func=cmd_feedback)
    
    # Parse and run
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
            sys.exit(130)
        except Exception as e:
            if args.verbose:
                console.print_exception()
            else:
                console.print(f"[bold red]Error: {e}[/bold red]")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
