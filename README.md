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

```bash
python main.py --minutes fplreview_final.csv --cs cs_1621.csv
```

## Example
- `--minutes` is the input file containing projected playing time and points  
- `--cs` is the input file containing clean sheet probabilities.
