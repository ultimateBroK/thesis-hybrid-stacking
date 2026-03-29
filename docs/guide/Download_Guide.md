# Data Download Guide

Download tick data from Dukascopy for forex, crypto, index, and stock assets using the `download_data.py` CLI tool.

## 🎯 Overview

The download script fetches 1-hour tick data from Dukascopy and saves it as Parquet files. It supports:
- **Forex**: XAUUSD, EURUSD, GBPUSD, etc. (5 trading days/week)
- **Crypto**: BTCUSD, ETHUSD, etc. (24/7 trading)
- **Indices**: US30, SPX500, etc.
- **Stocks**: Individual stock tickers

## ⚡ Quick Start

```bash
# Download XAUUSD from 2018 to present (forex - 5 days/week)
pixi run python download_data.py --symbol XAUUSD --start-year 2018

# Download Bitcoin from 2021 to 2023 (crypto - 24/7)
pixi run python download_data.py --symbol BTCUSD --asset-class crypto \
    --start-year 2021 --end-year 2023

# Dry run to see what would be downloaded
pixi run python download_data.py --symbol EURUSD --start-year 2022 --dry-run
```

## 📋 CLI Reference

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--symbol` | Trading symbol | `XAUUSD`, `BTCUSD`, `EURUSD` |
| `--start-year` | Start year (4 digits) | `2018`, `2020` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--asset-class` | `forex` | Asset type: `forex`, `crypto`, `index`, `stock` |
| `--start-month` | `1` | Start month (1-12) |
| `--end-year` | Current year | End year |
| `--end-month` | Current month | End month (1-12) |
| `--concurrency` | `50` | Parallel downloads (max 100) |
| `--force` | False | Re-download existing files |
| `--skip-current-month` | False | Skip current incomplete month |
| `--config` | `config.toml` | Path to config file |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--dry-run` | False | Preview without downloading |

## 📖 Usage Examples

### 1. Basic Downloads

```bash
# XAUUSD - Gold from 2018 to present (5 days/week trading)
pixi run python download_data.py --symbol XAUUSD --start-year 2018

# EURUSD - Euro/USD from 2020-2022
pixi run python download_data.py --symbol EURUSD --start-year 2020 --end-year 2022

# GBPUSD with specific month range
pixi run python download_data.py --symbol GBPUSD --start-year 2021 --start-month 6 \
    --end-year 2022 --end-month 12
```

### 2. Crypto Downloads (24/7 Trading)

```bash
# Bitcoin from 2021 to present
pixi run python download_data.py --symbol BTCUSD --asset-class crypto --start-year 2021

# Ethereum from 2022 to 2024
pixi run python download_data.py --symbol ETHUSD --asset-class crypto \
    --start-year 2022 --end-year 2024

# Crypto downloads more data (weekends included)
```

### 3. Dry Run (Preview Mode)

```bash
# See what would be downloaded for XAUUSD 2024
pixi run python download_data.py --symbol XAUUSD --start-year 2024 --dry-run

# Output shows:
# [DRY RUN] Would download to: data/raw/XAUUSD
# [DRY RUN] State file: data/raw/XAUUSD/.XAUUSD_download_state.json
# [DRY RUN] Would check 2024-01 for missing hours
# [DRY RUN] Would check 2024-02 for missing hours
# ...
```

### 4. Force Re-download

```bash
# Re-download all data (ignore existing files)
pixi run python download_data.py --symbol XAUUSD --start-year 2018 --force

# Force re-download with high concurrency for speed
pixi run python download_data.py --symbol XAUUSD --start-year 2022 \
    --force --concurrency 100
```

### 5. Skip Current Month

```bash
# Download up to last complete month only
pixi run python download_data.py --symbol XAUUSD --start-year 2023 \
    --skip-current-month
```

### 6. Debug Logging

```bash
# Full debug output
pixi run python download_data.py --symbol XAUUSD --start-year 2024 \
    --log-level DEBUG 2>&1 | head -100

# Silent mode (errors only)
pixi run python download_data.py --symbol XAUUSD --start-year 2024 \
    --log-level ERROR
```

### 7. Custom Config File

```bash
# Use alternative config file
pixi run python download_data.py --symbol XAUUSD --start-year 2022 \
    --config production_config.toml
```

## 📊 Output Structure

After downloading, your data directory will look like:

```
data/raw/XAUUSD/
├── 2018-01.parquet      # January 2018 tick data (~50M rows)
├── 2018-02.parquet      # February 2018
├── 2018-03.parquet      # March 2018
├── ...
├── 2025-12.parquet      # December 2025
├── 2026-01.parquet      # January 2026
└── .XAUUSD_download_state.json  # Download progress tracking
```

**State File**: The `.XAUUSD_download_state.json` file tracks:
- Which months are complete
- Which hours have been downloaded
- File row counts
- Last download timestamp

## ⏱️ Download Times

| Symbol | Years | Size | Time (50 concurrency) |
|--------|-------|------|----------------------|
| XAUUSD | 1 year | ~4 GB | 15-20 minutes |
| XAUUSD | 5 years | ~20 GB | 1-2 hours |
| XAUUSD | 8 years | ~32 GB | 2-3 hours |
| BTCUSD | 1 year | ~8 GB | 20-30 minutes |
| EURUSD | 1 year | ~3 GB | 10-15 minutes |

**Tips to Speed Up:**
- Increase concurrency: `--concurrency 100`
- Use wired connection (not WiFi)
- Run during off-peak hours
- Resume interrupted downloads (automatic)

## 🔄 Resuming Downloads

The script automatically resumes interrupted downloads:

```bash
# First attempt (interrupted)
pixi run python download_data.py --symbol XAUUSD --start-year 2018
# Ctrl+C after 10 minutes...

# Resume (checks state file, downloads missing hours only)
pixi run python download_data.py --symbol XAUUSD --start-year 2018
# Much faster - only downloads what was missing
```

## 🔍 Checking Download Status

```bash
# See what files exist
ls -lh data/raw/XAUUSD/

# Check state file
cat data/raw/XAUUSD/.XAUUSD_download_state.json | python -m json.tool

# Verify Parquet files
pixi run python -c "
import polars as pl
df = pl.read_parquet('data/raw/XAUUSD/2024-01.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {df.columns}')
print(f'Date range: {df[\"time\"].min()} to {df[\"time\"].max()}')
"
```

## 🐛 Troubleshooting

### "No module named 'thesis'"

**Fix**: Run from project root with pixi
```bash
cd /home/ultimatebrok/Downloads/thesis
pixi run python download_data.py --symbol XAUUSD --start-year 2024
```

### Download is very slow

**Fix**: Increase concurrency
```bash
pixi run python download_data.py --symbol XAUUSD --start-year 2024 --concurrency 100
```

### "Connection timeout" errors

**Fix**: Reduce concurrency and retry
```bash
pixi run python download_data.py --symbol XAUUSD --start-year 2024 \
    --concurrency 20 --log-level DEBUG
```

### Some months are incomplete

**Fix**: Force re-download specific months
```bash
pixi run python download_data.py --symbol XAUUSD --start-year 2024 --force
```

### Out of disk space

**Check space**: 
```bash
df -h data/raw/
```

**Expected sizes:**
- Forex (5 days/week): ~4 GB/year
- Crypto (7 days/week): ~8 GB/year

### Data validation fails

**Fix**: Delete corrupted file and re-download
```bash
rm data/raw/XAUUSD/2024-03.parquet
pixi run python download_data.py --symbol XAUUSD --start-year 2024 --start-month 3 --end-month 3
```

## 📝 Common Asset Symbols

### Forex
| Symbol | Description | Trading Days |
|--------|-------------|--------------|
| `XAUUSD` | Gold vs USD | Mon-Fri |
| `EURUSD` | Euro vs USD | Mon-Fri |
| `GBPUSD` | Pound vs USD | Mon-Fri |
| `USDJPY` | USD vs Yen | Mon-Fri |
| `AUDUSD` | Aussie vs USD | Mon-Fri |

### Crypto
| Symbol | Description | Trading Days |
|--------|-------------|--------------|
| `BTCUSD` | Bitcoin vs USD | 24/7 |
| `ETHUSD` | Ethereum vs USD | 24/7 |
| `LTCUSD` | Litecoin vs USD | 24/7 |

### Indices
| Symbol | Description |
|--------|-------------|
| `US30` | Dow Jones |
| `SPX500` | S&P 500 |
| `NAS100` | Nasdaq 100 |

## ⚙️ Asset Class Differences

### Forex (Default)
- **Trading hours**: 00:00-23:59 Monday-Friday
- **Downloads**: 120 hours/week (5 days × 24 hours)
- **Weekends**: No data (markets closed)
- **Use for**: XAUUSD, EURUSD, GBPUSD, etc.

### Crypto
- **Trading hours**: 00:00-23:59, 7 days/week
- **Downloads**: 168 hours/week (7 days × 24 hours)
- **Weekends**: Full data (markets always open)
- **Use for**: BTCUSD, ETHUSD, etc.

**Important**: Always specify `--asset-class crypto` for crypto assets, or you'll miss weekend data!

## 🎓 Best Practices

1. **Start with dry-run**
   ```bash
   pixi run python download_data.py --symbol XAUUSD --start-year 2024 --dry-run
   ```

2. **Download in chunks for large ranges**
   ```bash
   # Download 2018-2020 first
   pixi run python download_data.py --symbol XAUUSD --start-year 2018 --end-year 2020
   
   # Then 2021-2023
   pixi run python download_data.py --symbol XAUUSD --start-year 2021 --end-year 2023
   ```

3. **Verify after download**
   ```bash
   ls data/raw/XAUUSD/*.parquet | wc -l  # Should match months downloaded
   ```

4. **Keep state files intact**
   - Don't delete `.XAUUSD_download_state.json`
   - It enables resumable downloads

5. **Use appropriate concurrency**
   - 50: Good for most connections
   - 100: Fast but may trigger rate limits
   - 20: Stable for slow connections

## ✅ Download Checklist

Before running your thesis pipeline:

- [ ] Data directory exists: `data/raw/XAUUSD/`
- [ ] Parquet files for target date range exist
- [ ] State file exists: `.XAUUSD_download_state.json`
- [ ] File sizes are reasonable (~300MB-500MB per month for forex)
- [ ] No months with 0 rows (check state file)
- [ ] First and last months have complete data

## 🔗 Next Steps

After downloading data:

1. **Process ticks to OHLCV**
   ```bash
   python main.py --stage data
   ```

2. **Generate features**
   ```bash
   python main.py --stage features
   ```

3. **Run full pipeline**
   ```bash
   python main.py
   ```

See `Quickstart.md` for the complete workflow.

---

**Need more symbols?** Check Dukascopy's website for available tickers.

**Questions?** See `Config.md` for configuration options.
