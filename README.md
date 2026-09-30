# Multi-Agent Financial Analysis System

AAI-520 Natural Language Processing and GenAI, University of San Diego.
Final Team Project.

**Team (Group 7):** Eric Hernandez, Nolan Robbins, Pranav Dhinakar

**Due:** October 19, 2026, 11:59 pm (end of Module 7). No extensions.

## Project summary

We are building an autonomous Investment Research Agent that takes a stock
symbol, plans its research, calls financial data tools, critiques its own
output, and keeps notes that improve later runs. The system also has to
demonstrate three agentic workflow patterns: prompt chaining, routing, and
evaluator-optimizer.

The graded deliverable is a single code notebook exported to PDF (preferred)
or HTML. It must include a link to this repository and comments that explain
agent design and workflows, agent functions and capabilities, and evaluation
and iteration.

| Rubric area       | Weight | Points |
| ----------------- | ------ | ------ |
| Agent Functions   | 33.8%  | 120    |
| Workflow Patterns | 33.8%  | 120    |
| Code (notebook)   | 32.4%  | 115    |
| **Total**         |        | 355    |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── environment.yml         # conda environment (aai520-finagent)
├── pyproject.toml          # ruff config (PEP 8, 79 char lines)
├── .env.example            # copy to .env and add your API keys
├── scripts/
│   ├── check_setup.py      # verifies your keys and every data source
│   └── run_chain.py        # runs the prompt chain, prints each step
├── src/finagent/
│   ├── config.py           # settings and API key loading (reads .env)
│   ├── llm.py              # single entry point for Claude calls
│   ├── cache.py            # disk cache for API responses
│   ├── schemas.py          # shared data shapes (Article, NewsDigest)
│   ├── tools/              # Workstream A: data sources + registry
│   ├── workflows/          # Workstreams A and C: the three patterns
│   └── agent/              # Workstream B: the research agent
├── notebooks/
│   └── final_notebook.ipynb  # the graded deliverable
├── data/                   # API cache and local datasets (git-ignored)
├── memory/                 # agent notes across runs (git-ignored)
└── tests/                  # offline tests, no keys needed
```

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/pdhinaka/AAI-520-NLP-Final-Team-Project-Group7.git
cd AAI-520-NLP-Final-Team-Project-Group7
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Or with conda (installs the same requirements plus the package):

```bash
conda env create -f environment.yml
conda activate aai520-finagent
python -m ipykernel install --user --name aai520-finagent \
    --display-name "Python (aai520-finagent)"
cp .env.example .env
```

### API keys

Every key goes in the `.env` file at the repo root, which you just made
from `.env.example`. Open it and paste each key after the `=` sign. The
code reads keys from there through `src/finagent/config.py`; never put a
key directly in code or a notebook.

`.env` is git-ignored, so your keys stay on your machine. Only the empty
template `.env.example` is committed. Before any commit, `git status`
should never list `.env`.

| Variable | Used for | Where to get it | Cost |
| -------- | -------- | --------------- | ---- |
| `ANTHROPIC_API_KEY` | LLM calls (Claude) | [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys) | Paid per token; small trial credit for new accounts |
| `NEWSAPI_KEY` | News articles | [newsapi.org/register](https://newsapi.org/register) | Free developer plan |
| `FRED_API_KEY` | Economic data | [fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys) (create a free account first) | Free |
| `ALPHAVANTAGE_API_KEY` | Earnings (EPS vs. estimate) | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) | Free, about 25 calls a day |
| `SEC_USER_AGENT` | SEC EDGAR filings | No key. Enter your name and email, e.g. `"Jane Doe jdoe@sandiego.edu"` | Free |

Yahoo Finance (prices, financials, and news) needs no key.

For Anthropic, set a monthly spend limit in the console so a runaway
loop can't run up a bill. `LLM_MODEL` defaults to `claude-haiku-4-5`
for development; switch to `claude-sonnet-5` for the final run.

If the setup check fails with "This API key is not scoped to a
workspace", either create a new key from inside a workspace in the
console (simplest), or put the workspace ID in `ANTHROPIC_WORKSPACE_ID`.

### Check your setup

```bash
python scripts/check_setup.py
```

This makes one small live call to each source and prints `OK`, `FAIL`,
or `SKIP` (key not set). Everything should say `OK` before you start
working. The offline tests need no keys:

```bash
python -m pytest -q
```

### Caching

API responses are cached as JSON under `data/cache/` so repeated runs
don't burn free-tier limits. Delete that folder to force fresh data.
Set `CACHE_MODE=offline` in `.env` to read only from the cache, which
is what we will use for the final notebook run so the numbers don't
change between runs. LLM replies made through
`llm.complete_json_cached` are cached too (under `data/cache/llm/`),
so an offline run replays them exactly.

To use a Kaggle financial news dataset as an offline fallback, save it
as `data/local_news.csv`. It needs a `title` (or `headline`) column and
a date column; `text`, `source`, `url`, and `ticker` columns are used if
present.

## How we work

1. Pick a workstream below and put your name next to it.
2. Branch off `main` for each task: `git checkout -b a/news-tool`
   (prefix with the workstream letter).
3. Commit small and often. GitHub history is how contribution is measured.
4. Open a pull request into `main` and ask one teammate to review.
5. Tick the checkbox in this README in the same PR that finishes the task.
6. Run `ruff check .` and `python -m pytest -q` before pushing.

If we use an AI assistant for any code, we note it in a comment or in the
notebook. The course requires disclosure and explanation of AI-assisted work.

## Decisions to make first

- [x] Pick the LLM backend: Anthropic Claude, wired
      up in `src/finagent/llm.py` (Haiku 4.5 for development, Sonnet 5
      for the final run)
- [ ] Decide whether each of us uses our own Anthropic key or we share
      Pranav's
- [ ] Pick an agent framework, or agree to write plain Python
      (LangGraph, smolagents, CrewAI, and so on)
- [ ] Pick 2 or 3 demo tickers everyone tests against (for example AAPL,
      NVDA, JPM)
- [ ] Everyone: get API keys (see Setup) and pass `check_setup.py`
      (Pranav done; Eric and Nolan to do)
- [ ] Agree on shared result shapes. Started in `src/finagent/schemas.py`
      (`Article`); add report and evaluation shapes there
- [ ] Choose a team representative for the Module 4 status update and the
      final submission

## Workstreams

Each workstream is roughly one third of the work. Put your name in the
**Owner** line when you claim it.

### Workstream A: Data tools and the prompt chaining workflow

**Owner:** Pranav Dhinakar

Covers the "uses tools dynamically" data layer and Workflow Pattern 1.

Data tools (`src/finagent/tools/`)
- [x] `market_data.py`: yfinance wrapper for price history, key
      financials, and company info
- [x] `news.py`: NewsAPI (or Yahoo Finance news) fetch for a ticker, with
      a Kaggle financial news fallback for offline runs
- [x] `macro.py`: FRED series fetch (rates, CPI, unemployment)
- [x] `filings.py`: SEC EDGAR recent filings lookup (10-K, 10-Q, 8-K)
- [x] Optional: Alpha Vantage for earnings data (`earnings.py`)
- [x] Simple on-disk caching in `data/` so we do not burn free-tier limits
- [x] Each tool has a short docstring the agent can read to decide when to
      use it
- [x] Tool registry (`registry.py`) with Anthropic tool specs for the
      agent

Prompt chaining (`src/finagent/workflows/prompt_chain.py`)
- [x] Ingest: pull news articles for a ticker
- [x] Preprocess: clean text, dedupe, trim to length
- [x] Classify: label each article (earnings, product, legal, macro,
      analyst rating, other) and sentiment
- [x] Extract: pull entities, numbers, dates, and key events
- [x] Summarize: produce a short news digest for the ticker
- [x] Log the intermediate output of every step so the notebook can show
      the chain

Try it with `python scripts/run_chain.py AAPL`. Each run is saved to
`data/runs/`; the agent can call the whole chain as the
`get_news_digest` tool.

### Workstream B: The Investment Research Agent (agent functions)

**Owner:** Eric Hernandez

Covers all four Agent Functions (120 pts).

- [ ] `planner.py`: given a ticker, the LLM writes a list of research
      steps before doing anything
- [ ] `research_agent.py`: loop that executes the plan and chooses which
      tool to call at each step based on what it has so far
- [ ] `reflection.py`: after drafting the report, the agent scores its own
      output against a checklist (coverage, evidence, recency, balance) and
      lists gaps
- [ ] `memory.py`: save short notes after each run (what worked, what was
      missing, ticker-specific facts) to `memory/` and load them at the
      start of the next run
- [ ] Show a second run on the same ticker that visibly uses the memory
      from the first
- [ ] Final output is a structured research report (dict or markdown)

### Workstream C: Routing, evaluator-optimizer, and the final notebook

**Owner:** Nolan Robbins

Covers Workflow Patterns 2 and 3 and most of the Code rubric (115 pts).

Routing (`src/finagent/workflows/routing.py`)
- [ ] Router that sends each piece of content to the right specialist
- [ ] Specialists: earnings analyzer, news analyzer, market or technical
      analyzer (at least three)
- [ ] Log the routing decision and reason for each item

Evaluator-optimizer (`src/finagent/workflows/evaluator_optimizer.py`)
- [ ] Generator produces an analysis
- [ ] Evaluator scores it with a rubric and returns written feedback
- [ ] Optimizer revises using the feedback; loop until the score passes
      or a max iteration count is hit
- [ ] Record the score for each iteration so we can plot improvement

Final notebook (`notebooks/final_notebook.ipynb`)
- [ ] Section per requirement with markdown explaining design choices
- [ ] Visualizations: agent flow diagram, price chart, routing
      distribution, evaluator scores across iterations
- [ ] Repository link at the top
- [ ] Clean top-to-bottom run, then export to PDF
- [ ] Final PEP 8 / ruff pass on the whole repo

## Shared tasks (everyone)

- [ ] Module 4: Team Project Status Update Form submitted by the
      representative
- [ ] Integration run: agent calls the chain, router, and evaluator on one
      ticker end to end
- [ ] Each member reviews at least one PR from each other member
- [ ] AI tool usage disclosed in the notebook
- [ ] Final notebook PDF submitted to Canvas by one representative
- [ ] Each member submits their individual Peer Evaluation (Module 7)

## Tool reference (for Workstreams B and C)

All tools return plain JSON-friendly dicts or lists and are cached.
The agent gets them through the registry:

```python
from finagent import llm
from finagent.tools import registry

registry.describe()                  # one line per tool, for prompts
specs = registry.tool_specs()        # pass as tools= to the LLM
registry.call("get_price_summary", {"ticker": "AAPL"})
llm.complete("...", system="...")    # plain text reply
llm.complete_json("...")             # parsed JSON, retries on bad JSON
llm.complete_json_cached("...")      # same, reply cached on disk
llm.create_message(messages, tools=specs)  # raw call for tool-use loops
llm.USAGE                            # running token totals
```

| Tool | What it returns |
| ---- | --------------- |
| `get_company_info` | Profile, sector, market cap, P/E, margins, analyst target |
| `get_price_summary` | Last close, 1m/3m/6m/1y returns, 52-week range, volatility |
| `get_price_history` | OHLCV rows for charts |
| `get_financials` | Key income, balance sheet, and cash flow lines |
| `get_news` | Recent articles (NewsAPI + Yahoo, local CSV fallback) |
| `get_news_digest` | The prompt chain's digest: headline, cited summary, catalysts, risks, sentiment and topic counts |
| `get_macro_snapshot` | Fed funds, 10y yield, CPI YoY, unemployment, VIX |
| `get_series` | Any FRED series |
| `get_recent_filings` | Recent 10-K, 10-Q, 8-K with document links |
| `get_filing_text` | Text of a filing, optionally jumping to a section |
| `get_earnings` | Reported vs. estimated EPS by quarter |

Tool errors come back as `{"error": "..."}` from `registry.call`, so the
agent can react instead of crashing.

## Data sources

- Yahoo Finance (`yfinance`): prices and financials
- NewsAPI.org and Kaggle financial news datasets
- FRED API: economic data
- SEC EDGAR: company filings
- Alpha Vantage (free tier)
