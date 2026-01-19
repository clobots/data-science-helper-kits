# CLAUDE.md - Data Science Agent

## Project Overview

This is a Python-based data science agent for Woolworths Australia supermarket analytics. It uses LangChain with Google Gemini for LLM-powered analysis of operational metrics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no API key required)
python main.py --demo

# Run with LLM (requires GEMINI_API_KEY in .env)
python main.py --demo --use-llm
```

## Key Commands

```bash
python main.py --demo              # Full demo workflow
python main.py --quick-scan        # Quick critical issues scan
python main.py --generate-data     # Generate sample data only
python main.py --metrics sales wages --audience executive
```

## Architecture

The agent has 5 main tools:
1. **Data Extractor** - Simulated BigQuery interface for metric data
2. **Hypothesis Engine** - Detects significant changes (WoW, budget, trends)
3. **Insights Engine** - Investigates hypotheses through hierarchy levels
4. **Summary Engine** - Generates audience-specific summaries
5. **Summariser** - General-purpose content summarization

## File Organization

- `config/` - Settings and prompt templates (YAML)
- `models/` - Pydantic models for data and LLM I/O
- `clients/` - LLM client wrappers for each tool
- `tools/` - Business logic for each tool
- `agent/` - Main orchestration agent
- `utils/` - Response parsing and metrics calculations

## Key Patterns

### No `.withStructuredOutput`
All LLM outputs use custom JSON parsing via `utils/response_parser.py`. Prompts include schema instructions and responses are parsed to Pydantic models.

### Dual Mode Operation
All tools support both LLM-powered and rule-based operation (`use_llm=True/False`). This allows testing without API keys.

### Hierarchy Navigation
Data is organized: National -> Zone -> State -> Group -> Store -> Category. The insights engine drills through these levels to find root causes.

## Environment Variables

```
GEMINI_API_KEY=your_api_key_here
```

## Common Tasks

### Add a new metric
1. Add to `MetricType` enum in `models/data_models.py`
2. Add config in `METRIC_CONFIGS` in `data/generator.py`
3. Add description in `config/settings.py`

### Modify prompts
Edit YAML files in `config/prompts/`. Each tool has its own prompt file.

### Change thresholds
Edit `SignificanceThresholds` in `config/settings.py`

## Testing

```bash
pytest tests/  # When tests are added
python main.py --demo  # Manual verification
```

## Metrics Tracked

| Metric | Description | Direction |
|--------|-------------|-----------|
| wages | Labor cost as % of sales | Lower better |
| voice_of_customer | Satisfaction score 0-100 | Higher better |
| sales | Revenue in AUD | Higher better |
| stockloss | Inventory shrinkage % | Lower better |
| order_pickrate | Fulfillment rate items/hr | Higher better |

## Hierarchy Levels

```
national -> zone (6) -> state (8) -> group (~50) -> store (~100) -> category (20)
```
