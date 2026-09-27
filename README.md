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
├── pyproject.toml          # ruff config (PEP 8, 79 char lines)
├── .env.example            # copy to .env and add your API keys
├── src/finagent/
│   ├── config.py           # settings and API key loading
│   ├── llm.py              # single entry point for LLM calls
│   ├── tools/              # Workstream A: data sources
│   ├── workflows/          # Workstreams A and C: the three patterns
│   └── agent/              # Workstream B: the research agent
├── notebooks/
│   └── final_notebook.ipynb  # the graded deliverable
├── data/                   # local data cache (git-ignored)
├── memory/                 # agent notes across runs (git-ignored)
└── tests/
```

## Setup

```bash
git clone https://github.com/pdhinaka/AAI-520-NLP-Final-Team-Project-Group7.git
cd AAI-520-NLP-Final-Team-Project-Group7
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
pip install -e .
cp .env.example .env    # then fill in your keys
```

Never commit `.env`. It is in `.gitignore`.

## How we work

1. Pick a workstream below and put your name next to it.
2. Branch off `main` for each task: `git checkout -b a/news-tool`
   (prefix with the workstream letter).
3. Commit small and often. GitHub history is how contribution is measured.
4. Open a pull request into `main` and ask one teammate to review.
5. Tick the checkbox in this README in the same PR that finishes the task.
6. Run `ruff check .` before pushing.

If we use an AI assistant for any code, we note it in a comment or in the
notebook. The course requires disclosure and explanation of AI-assisted work.

## Decisions to make first

- [ ] Pick the LLM backend (Hugging Face model, OpenAI, or other) and
      wrap it in `src/finagent/llm.py` so everyone calls the same function
- [ ] Pick an agent framework, or agree to write plain Python
      (LangGraph, smolagents, CrewAI, and so on)
- [ ] Pick 2 or 3 demo tickers everyone tests against (for example AAPL,
      NVDA, JPM)
- [ ] Get API keys: NewsAPI, FRED, Alpha Vantage (all free tier)
- [ ] Agree on a shared result schema so modules can pass data between
      each other (dict or dataclass in `src/finagent/__init__.py`)
- [ ] Choose a team representative for the Module 4 status update and the
      final submission

## Workstreams

Each workstream is roughly one third of the work. Put your name in the
**Owner** line when you claim it.

### Workstream A: Data tools and the prompt chaining workflow

**Owner:** _unclaimed_

Covers the "uses tools dynamically" data layer and Workflow Pattern 1.

Data tools (`src/finagent/tools/`)
- [ ] `market_data.py`: yfinance wrapper for price history, key
      financials, and company info
- [ ] `news.py`: NewsAPI (or Yahoo Finance news) fetch for a ticker, with
      a Kaggle financial news fallback for offline runs
- [ ] `macro.py`: FRED series fetch (rates, CPI, unemployment)
- [ ] `filings.py`: SEC EDGAR recent filings lookup (10-K, 10-Q, 8-K)
- [ ] Optional: Alpha Vantage for earnings data
- [ ] Simple on-disk caching in `data/` so we do not burn free-tier limits
- [ ] Each tool has a short docstring the agent can read to decide when to
      use it

Prompt chaining (`src/finagent/workflows/prompt_chain.py`)
- [ ] Ingest: pull news articles for a ticker
- [ ] Preprocess: clean text, dedupe, trim to length
- [ ] Classify: label each article (earnings, product, legal, macro,
      analyst rating, other) and sentiment
- [ ] Extract: pull entities, numbers, dates, and key events
- [ ] Summarize: produce a short news digest for the ticker
- [ ] Log the intermediate output of every step so the notebook can show
      the chain

### Workstream B: The Investment Research Agent (agent functions)

**Owner:** _unclaimed_

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

**Owner:** _unclaimed_

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

## Data sources

- Yahoo Finance (`yfinance`): prices and financials
- NewsAPI.org and Kaggle financial news datasets
- FRED API: economic data
- SEC EDGAR: company filings
- Alpha Vantage (free tier)
