#!/usr/bin/env python3
"""
Dashboard script for quick project status overview.

Provides an ADHD-friendly visual summary of:
- Latest session status
- Key metrics from last run
- Quick navigation commands
- Project health check
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD} {text}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(label: str, value: str) -> None:
    """Print labeled info."""
    print(f"  {Colors.CYAN}{label}:{Colors.RESET} {value}")


def get_latest_session() -> Path | None:
    """Get the path to the latest session."""
    results_dir = Path("results")
    latest_link = results_dir / "latest"
    
    if latest_link.exists() and latest_link.is_symlink():
        return latest_link.resolve()
    
    # Fallback: find most recent directory
    sessions = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("XAUUSD")]
    if sessions:
        return max(sessions, key=lambda p: p.stat().st_mtime)
    
    return None


def get_all_sessions() -> list[Path]:
    """Get all session directories sorted by date."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    
    sessions = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("XAUUSD")]
    return sorted(sessions, key=lambda p: p.stat().st_mtime)


def format_session_name(session_path: Path) -> str:
    """Format session name for display."""
    name = session_path.name
    # Parse: XAUUSD_1H_YYYYMMDD_HHMMSS
    parts = name.split("_")
    if len(parts) >= 4:
        date_str = parts[2]
        time_str = parts[3]
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return name


def load_backtest_metrics(session_path: Path) -> dict | None:
    """Load backtest metrics from a session."""
    backtest_file = session_path / "backtest" / "backtest_results.json"
    if backtest_file.exists():
        try:
            with open(backtest_file) as f:
                data = json.load(f)
                return data.get("metrics", {})
        except (json.JSONDecodeError, IOError):
            pass
    return None


def format_number(value: float | int | None, decimals: int = 2) -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def print_metrics_summary(metrics: dict | None) -> None:
    """Print a summary of key metrics."""
    if not metrics:
        print_warning("No metrics available")
        return
    
    print_section("Key Metrics")
    
    # Determine color based on performance
    total_return = metrics.get("total_return_pct", 0)
    return_color = Colors.GREEN if total_return > 0 else Colors.RED
    
    sharpe = metrics.get("sharpe_ratio", 0)
    sharpe_color = Colors.GREEN if sharpe > 1.5 else Colors.YELLOW if sharpe > 0 else Colors.RED
    
    max_dd = metrics.get("max_drawdown_pct", 0)
    dd_color = Colors.GREEN if max_dd < 20 else Colors.YELLOW if max_dd < 30 else Colors.RED
    
    win_rate = metrics.get("win_rate", 0)
    wr_color = Colors.GREEN if win_rate > 0.55 else Colors.YELLOW if win_rate > 0.45 else Colors.RED
    
    print(f"  {Colors.CYAN}Total Return:{Colors.RESET}     {return_color}{format_number(total_return)}%{Colors.RESET}")
    print(f"  {Colors.CYAN}Sharpe Ratio:{Colors.RESET}     {sharpe_color}{format_number(sharpe)}{Colors.RESET}")
    print(f"  {Colors.CYAN}Max Drawdown:{Colors.RESET}     {dd_color}{format_number(max_dd)}%{Colors.RESET}")
    print(f"  {Colors.CYAN}Win Rate:{Colors.RESET}         {wr_color}{format_number(win_rate * 100, 1)}%{Colors.RESET}")
    print(f"  {Colors.CYAN}Total Trades:{Colors.RESET}     {format_number(metrics.get('total_trades'), 0)}")
    print(f"  {Colors.CYAN}Profit Factor:{Colors.RESET}    {format_number(metrics.get('profit_factor'))}")


def print_session_files(session_path: Path) -> None:
    """Print available files in a session."""
    print_section("Available Files")
    
    files_to_check = [
        ("Report", "reports/thesis_report.md"),
        ("JSON Report", "reports/thesis_report.json"),
        ("SHAP Summary", "reports/shap_summary.png"),
        ("Model Disagreement", "reports/model_disagreement.png"),
        ("Confidence Histogram", "reports/confidence_histogram.png"),
        ("Backtest Results", "backtest/backtest_results.json"),
        ("Trade Details", "backtest/trades_detail.csv"),
        ("LightGBM Model", "models/lightgbm_model.pkl"),
        ("LSTM Model", "models/lstm_model.pt"),
        ("Stacking Model", "models/stacking_meta_learner.pkl"),
    ]
    
    for label, rel_path in files_to_check:
        full_path = session_path / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            size_str = f"({size / 1024:.1f} KB)" if size < 1024*1024 else f"({size / (1024*1024):.1f} MB)"
            print_success(f"{label}: {rel_path} {Colors.DIM}{size_str}{Colors.RESET}")
        else:
            print(f"  {Colors.DIM}○ {label}: {rel_path} (not found){Colors.RESET}")


def print_quick_commands(session_path: Path | None) -> None:
    """Print quick command reference."""
    print_section("Quick Commands")
    
    commands = [
        ("Run full pipeline", "pixi run workflow"),
        ("Force re-run", "pixi run force"),
        ("Run tests", "pixi run test"),
        ("Check code", "pixi run lint"),
        ("Format code", "pixi run format"),
        ("List sessions", "python main.py --list-sessions"),
    ]
    
    if session_path:
        commands.extend([
            ("View report", f"cat {session_path}/reports/thesis_report.md"),
            ("View backtest", f"cat {session_path}/backtest/backtest_results.json | jq '.metrics'"),
        ])
    
    for description, command in commands:
        print(f"  {Colors.CYAN}{description}:{Colors.RESET}")
        print(f"    {Colors.YELLOW}{command}{Colors.RESET}")


def print_health_check() -> None:
    """Print project health check."""
    print_section("Health Check")
    
    checks = [
        ("Config file", Path("config.toml").exists()),
        ("Source code", Path("src/thesis").exists()),
        ("Tests", Path("tests").exists()),
        ("Raw data", Path("data/raw/XAUUSD").exists()),
        ("Processed data", Path("data/processed").exists()),
        ("Results directory", Path("results").exists()),
    ]
    
    all_good = True
    for name, exists in checks:
        if exists:
            print_success(name)
        else:
            print_error(name)
            all_good = False
    
    if all_good:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}⚠ Some components missing. Run 'pixi run workflow' to initialize.{Colors.RESET}")


def main() -> int:
    """Main dashboard function."""
    print_header("THESIS PIPELINE DASHBOARD")
    
    # Get latest session
    latest_session = get_latest_session()
    all_sessions = get_all_sessions()
    
    # Session info
    print_section("Session Status")
    if latest_session:
        print_success(f"Latest session: {latest_session.name}")
        print_info("Created", format_session_name(latest_session))
        print_info("Total sessions", f"{len(all_sessions)}")
        
        # Show recent sessions
        if len(all_sessions) > 1:
            print(f"\n  {Colors.CYAN}Recent sessions:{Colors.RESET}")
            for session in all_sessions[-3:]:
                marker = f"{Colors.GREEN}→{Colors.RESET}" if session == latest_session else " "
                print(f"    {marker} {session.name}")
    else:
        print_warning("No sessions found")
        print(f"  {Colors.DIM}Run 'pixi run workflow' to create your first session{Colors.RESET}")
    
    # Metrics from latest session
    if latest_session:
        metrics = load_backtest_metrics(latest_session)
        print_metrics_summary(metrics)
        print_session_files(latest_session)
    
    # Health check
    print_health_check()
    
    # Quick commands
    print_quick_commands(latest_session)
    
    # Footer
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.DIM}Dashboard complete. Run 'pixi run dashboard' anytime for updates.{Colors.RESET}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
