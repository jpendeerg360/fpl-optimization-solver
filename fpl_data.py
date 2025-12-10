import requests
import pandas as pd
import concurrent.futures

# Constants for FPL API
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HISTORY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"
# Team name mapping to help with merges
TEAM_NAME_MAPPING = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Burnley': 'BUR', 'Bournemouth': 'BOU',
    'Brentford': 'BRE', 'Brighton': 'BHA', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Leeds': 'LEE', 'Liverpool': 'LIV',
    'Man City': 'MCI', 'Man Utd': 'MUN', 'Newcastle': 'NEW', "Nott'm Forest": 'NFO',
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL'
}

def fetch_api_data():
    """Fetches data from the FPL API."""
    r = requests.get(API_URL)
    data = r.json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    positions = pd.DataFrame(data["element_types"])
    return players, teams, positions

def load_and_prep_minutes(filepath):
    """
    Loads minutes projections and renames columns dynamically for GW 16-21.
    """
    minutes_df = pd.read_csv(filepath)
    minutes_rename_map = {f"{i}_xMins": f"xMin{i}" for i in range(16, 22)}
    minutes_df = minutes_df.rename(columns={"ID": "player_id", **minutes_rename_map})
    return minutes_df, minutes_rename_map

def load_and_prep_clean_sheets(filepath):
    """
    Loads clean sheet probabilities, standardizes column names, and converts % to floats.
    """
    cs_df = pd.read_csv(filepath)
    
    for i in range(1, 7):
        gw = i + 15
        opp_col_orig = f"Opp{i}"
        cs_col_orig = f"CS{i}%"

        # Rename columns to standard format
        cs_df = cs_df.rename(columns={opp_col_orig: f"Opp{gw}"})

        # Process CS% Column (Strip % and convert to float)
        if cs_df[cs_col_orig].dtype == object:
            cs_df[cs_col_orig] = cs_df[cs_col_orig].str.rstrip('%').astype(float) / 100
        
        cs_df = cs_df.rename(columns={cs_col_orig: f"xClean_sheets{gw}"})

    cs_df = cs_df.rename(columns={"Team": "team_name"})
    return cs_df

def fetch_single_player_history(player_id):
    """
    Worker function to fetch and calculate historical stats for a single player.
    """
    try:
        data = requests.get(HISTORY_URL.format(player_id)).json()
        past = pd.DataFrame(data["history_past"])
        # Check if the player has previous season data
        if past.empty:
            return 0
        # Subset the past data for the last three seasons where they played at least 250 minutes
        past = past[past["season_name"].isin(["2022/23", "2023/24", "2024/25"])]
        past = past[past["minutes"] > 250]
        # Check if the player still has data after the subset
        if past.empty:
            return 0

        # Calculate per-90 stats
        stats = ["goals_scored", "assists", "bonus", "saves", "yellow_cards"]
        cols = ["g", "a", "b", "s", "y"]
        
        for stat, col in zip(stats, cols):
            past[col] = (past[stat] / past["minutes"]) * 90

        # Weighted average based on number of available seasons
        n_seasons = past.shape[0]
        weights = []
        # Sets weights for past seasons
        if n_seasons == 1:
            weights = [1.0]
        elif n_seasons == 2:
            weights = [0.35, 0.65]
        else:
            # Assuming sorted by season; last entry is most recent
            weights = [0.1, 0.25, 0.65] 

        final_stats = []
        for col in cols:
            val = sum(past[col].iloc[i] * weights[i] for i in range(n_seasons))
            final_stats.append(val)
            
        final_stats.append(player_id)
        return final_stats
        
    except Exception:
        return 0

def get_player_history(player_ids):
    """
    Orchestrates threaded fetching of history for all players.
    """
    past_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_single_player_history, pid): pid for pid in player_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            if result != 0:
                past_list.append(result)
    
    return pd.DataFrame(past_list, columns=["g", "a", "b", "s", "y", "id"]).set_index("id")

def build_master_dataframe(players, teams, positions, minutes_df, minutes_map, cs_df):
    """
    Merges all separate dataframes into one master player list.
    """
    players['player_id'] = players['id']
    
    # 1. Merge Base Data
    players_full = (
        players
        .merge(positions[["id", "singular_name"]], left_on="element_type", right_on="id", how="left")
        .merge(teams[["id", "name"]], left_on="team", right_on="id", suffixes=("", "_team"))
    )

    # 2. Merge Minutes
    players_full = players_full.merge(
        minutes_df[["player_id"] + list(minutes_map.values())], 
        on="player_id", 
        how="left", 
        validate="1:1"
    )

    # 3. Clean Team Names & Positions
    players_full["short_team_name"] = players_full["name"].map(TEAM_NAME_MAPPING)
    players_full = players_full.rename(columns={"singular_name": "position", "name": "team_name"})

    # 4. Merge Clean Sheet Data
    cs_cols_to_merge = ["team_name"]
    for gw in range(16, 22):
        cs_cols_to_merge.extend([f"Opp{gw}", f"xClean_sheets{gw}", f"x2_{gw}", f"x4_{gw}"])

    players_full = players_full.merge(
        cs_df[cs_cols_to_merge], 
        on="team_name", 
        how="left", 
        validate="many_to_one"
    )
    
    return players_full