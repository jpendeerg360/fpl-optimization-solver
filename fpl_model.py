import numpy as np
import pandas as pd
from scipy.stats import poisson
# Helps account for playing easy teams vs hard teams
DIFFICULTY_MAP = {
    # Tier 1 (Hardest - 0.85)
    'ARS': 0.85, 'MCI': 0.85, 'LIV': 0.85, 'BRE': 0.85,
    # Tier 2 (Hard - 0.90)
    'MUN': 0.90, 'NEW': 0.90, 'EVE': 0.90, 'CHE': 0.90,
    # Tier 3 (Neutral - 1.00)
    'CRY': 1.00, 'FUL': 1.00, 'BHA': 1.00, 'AVL': 1.00,
    # Tier 4 (Easy - 1.10)
    'BOU': 1.10, 'NFO': 1.10, 'TOT': 1.10, 'SUN': 1.10,
    # Tier 5 (Target - 1.20)
    'LEE': 1.20, 'WHU': 1.20, 'WOL': 1.20, 'BUR': 1.20
}

def calculate_opp_goal_probs(cs_prob):
    """
    Computes opponent goal probabilities (2-3 goals, 4+ goals) 
    derived from the team's clean sheet probability using Poisson.
    """
    lam = -np.log(cs_prob)
    p0 = np.exp(-lam)
    p1 = lam * p0
    p2 = (lam**2) * p0 / 2
    p3 = (lam**3) * p0 / 6
    # p23 = Prob of conceding exactly 2 or 3 goals
    p23 = p2 + p3
    # p4plus = Prob of conceding 4 or more
    p4plus = 1 - (p0 + p1 + p2 + p3)
    return p23, p4plus

def prepare_data_for_modeling(df):
    """
    Converts columns to numeric, calculates per90 stats, and applies form multipliers.
    """
    cols_to_numeric = [
        "now_cost", "expected_goals", "expected_goals_per_90", 
        "minutes", "expected_assists", "expected_assists_per_90", 
        "saves", "bonus", "form"
    ]
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col])

    # Subset the players we actually want to model
    df = df[df["minutes"] > 250]
    df = df[df["xMin16"] > 55] 

    # per90 conversions
    df["now_cost"] = df["now_cost"] / 10.
    df["bonus_per_90"] = df["bonus"] / df["minutes"] * 90
    df["yellow_cards_per_90"] = df["yellow_cards"] / df["minutes"] * 90
    df["def_contrib_per90"] = df["defensive_contribution"] / df["minutes"] * 90

    # Form Multiplier Logic
    conditions = [
        (df['form'] >= 6.0),
        (df['form'] >= 2.0),
        (df['form'] >= 0)
    ]
    choices = [1.1, 1, 0.9]
    df['form_multiplier'] = np.select(conditions, choices, default=0.8)
    
    return df

def fill_missing_history_with_current(df):
    """Fills missing historical data (g, a, b, etc.) with current season xG/xA."""
    df["g"] = df["g"].fillna(df["expected_goals_per_90"])
    df["a"] = df["a"].fillna(df["expected_assists_per_90"])
    df["b"] = df["b"].fillna(df["bonus_per_90"])
    df["s"] = df["s"].fillna(df["saves_per_90"])
    df["y"] = df["y"].fillna(df["yellow_cards_per_90"])
    return df

def compute_gw_points(df, weights, gw=16):
    """
    Calculates expected points for a specific gameweek based on position weights.
    
    weights: dict containing {goal, assist, cs, save, def_weight, defcon1, defcon2}
    """
    # Unpack column names for the specific GW
    xmin_col = f"xMin{gw}"
    cs_col = f"xClean_sheets{gw}"
    x2_col = f"x2_{gw}"
    x4_col = f"x4_{gw}"
    opp_col = f"Opp{gw}"

    # Fixture Difficulty
    df["fixture_mult"] = df[opp_col].map(DIFFICULTY_MAP).fillna(1.0)

    # Defensive calculations (Prob of conceding 0 or 1, etc.)
    df["lambda_def"] = df["def_contrib_per90"] * (1 - 0.5 * df[cs_col])
    df["p_10"] = 1 - poisson.cdf(9, df["lambda_def"])
    df["p_12"] = 1 - poisson.cdf(11, df["lambda_def"])

    # Offensive Component
    # Weighted average of Historical (30%) and Expected (70%)
    offensive_points = (
        (df["expected_goals_per_90"]*0.7 + df["g"]*0.3) * df["form_multiplier"] * df["fixture_mult"] * weights['goal'] +
        (df["bonus_per_90"]*0.7 + df["b"]*0.3) * df["form_multiplier"] * df["fixture_mult"] +
        (df["expected_assists_per_90"]*0.7 + df["a"]*0.3) * df["form_multiplier"] * df["fixture_mult"] * weights['assist']
    ) * (df[xmin_col] / 90)

    # Defensive Component
    defensive_points = (
        df[cs_col] * weights['cs'] +
        (df["saves_per_90"]*0.5 + df["s"]*0.5) * weights['save'] -
        df[x2_col] * weights['def'] -
        2 * df[x4_col] * weights['def'] +
        2 * df["p_10"] * weights['defcon1'] +
        2 * df["p_12"] * weights['defcon2']
    )

    # Final Sum
    df[f"expected_points{gw}"] = (
        2 + # Base appearance points (assuming 60 mins played)
        offensive_points +
        defensive_points -
        (df["yellow_cards_per_90"]*0.3 + df["y"]*0.7)
    )
    return df

def run_projections(players_df):
    """
    Iterates through GW 16-21 and calculates points for all positions.
    """
    # Split by position
    keepers = players_df[players_df["position"] == "Goalkeeper"].copy()
    defenders = players_df[players_df["position"] == "Defender"].copy()
    midfielders = players_df[players_df["position"] == "Midfielder"].copy()
    forwards = players_df[players_df["position"] == "Forward"].copy()

    # Define Position-Specific Weights
    # Structure: goal, assist, cs, save, def, defcon1, defcon2
    w_fwd = {'goal': 4, 'assist': 3, 'cs': 0, 'save': 0, 'def': 0, 'defcon1': 0, 'defcon2': 1}
    w_mid = {'goal': 5, 'assist': 3, 'cs': 1, 'save': 0, 'def': 0, 'defcon1': 0, 'defcon2': 1}
    w_def = {'goal': 6, 'assist': 3, 'cs': 4, 'save': 0, 'def': 1, 'defcon1': 1, 'defcon2': 0}
    w_gk  = {'goal': 0, 'assist': 0, 'cs': 4, 'save': 1/3., 'def': 0, 'defcon1': 1, 'defcon2': 0}
    # Computes points for each position and week
    for gw in range(16, 22):
        forwards = compute_gw_points(forwards, w_fwd, gw)
        midfielders = compute_gw_points(midfielders, w_mid, gw)
        defenders = compute_gw_points(defenders, w_def, gw)
        keepers = compute_gw_points(keepers, w_gk, gw)

    # Merge back to master dataframe
    players_df["total_expected_points"] = 0
    for gw in range(16, 22):
        col = f"expected_points{gw}"
        players_df.loc[forwards.index, col] = forwards[col]
        players_df.loc[midfielders.index, col] = midfielders[col]
        players_df.loc[defenders.index, col] = defenders[col]
        players_df.loc[keepers.index, col] = keepers[col]
        
    return players_df

def apply_decay(df, decay_rate=0.92):
    decayed_cols = []
    for i, gw in enumerate(range(16, 22)):
        col_name = f"expected_points{gw}"
        decayed_col = f"decayed_points{gw}"
        weight = decay_rate ** i
        df[decayed_col] = df[col_name] * weight
        decayed_cols.append(decayed_col)
    
    df["total_decayed_points"] = df[decayed_cols].sum(axis=1)
    return df