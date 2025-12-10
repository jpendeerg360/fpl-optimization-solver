# FPL Optimization Model

This project calculates projected Fantasy Premier League (FPL) points and optimizes squads for Wildcards and Free Hits using Linear Programming.

## Prerequisites
- Python 3.8 or higher

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/jpendeerg360/fpl-optimization-solver.git
cd fpl-optimization-solver
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the model:**
Default (gives top 10 players at each position, an optimized squad for the wildcard and freehit chip)
```bash
python main.py --minutes fplreview_final.csv --cs cs_1621.csv
```
Custom Projections (look at print_projections description for guide)
Examples:
Filter by forwards:
```bash
python main.py --minutes fplreview.csv --cs cs_1621.csv --position Forward
```
Sort by Price (Shows Top 50 most expensive players):
```bash
python main.py --minutes fplreview.csv --cs cs_1621.csv --sort price --top 50
```
Specific Gameweek Defenders (Shows all Defenders sorted by GW16 points):
```bash
python main.py --minutes fplreview.csv --cs cs_1621.csv --position Def --sort gw --gw 16 --top 0
```
## filepath
- `--minutes` is the input file containing projected playing time and points  
- `--cs` is the input file containing clean sheet probabilities.
