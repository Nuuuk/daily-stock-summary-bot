# 📈 Daily Stock Positions Summary Bot

[English](#english) | [中文说明](#中文说明)

---

<a name="english"></a>
## English

An automated, privacy-first portfolio analysis and risk management bot powered by **Google Gemini 3.7 Flash**, **yfinance**, and **Finnhub**. Designed specifically for aggressive, long-term investors holding US equities, the bot tracks multi-broker tax lots, enforces strict tax-clock discipline (holding > 1 year), monitors macro liquidity catalysts, and delivers actionable HTML email briefings before market open and during mid-day trading via **GitHub Actions**.

### ✨ Key Features

* **Zero-Leakage Privacy Architecture**: Portfolio data, cash reserves, tax brackets, and broker lots reside exclusively inside encrypted GitHub Secrets (`APP_CONFIG_JSON`). The codebase contains zero hardcoded financial numbers.
* **Granular Tax Lot & Tax-Clock Tracking**: Tracks independent purchase lots across brokers (Robinhood, Fidelity, Moomoo, Tiger, etc.). Flags long-term vs. short-term capital gains status and guards against premature selling on lots approaching the 365-day mark.
* **Macro & Individual Stock Data Aggregation**: Fetches real-time price action, relative volume, 50/200 DMA trends, VIX, US 10-Year Yields (`^TNX`), DXY, and breaking news catalysts with deduplicated caching.
* **Disciplined Decision Engine**: Default "Hold" stance to eliminate noise, with "Fat Pitch" cash allocation recommendations triggered only during severe pullbacks or high-conviction setups.
* **Self-Healing Resilience**: Built-in exponential backoff retries handling transient 429 rate limits or 503 model overload events.

---

### 🕒 Workflow Schedule

| Session | Execution Time (EST) | Purpose & Focus |
|---|---|---|
| **Pre-Market Briefing** | **08:30 AM** | Macro sentiment, overnight news, earnings announcements, daily decision tone, Tax Clock overview. |
| **Mid-Day Scan** | **01:30 PM** | Trend confirmation, relative volume spikes, breaking midday catalysts, false breakout/breakdown filtering. |

---

### 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── summary.yml       # GitHub Actions cron & manual dispatch workflow
├── data_collector.py         # Market data, news aggregator & Tax Lot calculator
├── main.py                   # Gemini 3.7 Flash analysis engine & SMTP dispatcher
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
