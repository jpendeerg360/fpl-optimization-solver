# Load libraries
import argparse
import fpl_data
import fpl_model
import fpl_solver
import sys

# File names for me
MINUTES_FILE = "/Users/johnpendergrass/Downloads/fplreview_final.csv"
CS_FILE = "/Users/johnpendergrass/Downloads/cs_1621.csv"

def print_top_players(df, n=10):
    """
    Displays the best n players for the next six gameweeks by position
    """
    positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    cols_to_show = ["first_name", "second_name", "team_name", "now_cost", "total_decayed_points"]
    
    print(f"\n{'='*40}")
    print(f"   TOP {n} PROJECTIONS BY POSITION")
    print(f"{'='*40}")

    for pos in positions:
        print(f"\n--- {pos}s ---")
        subset = df[df["position"] == pos].sort_values(by="total_decayed_points", ascending=False).head(n)
        # Format for cleaner output
        print(subset[cols_to_show].to_string(index=False, float_format="%.1f"))

def main():
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Run FPL Optimization Model")
    
    parser.add_argument(
        "--minutes", 
        required=True, 
        help="Path to the minutes projection CSV (e.g., fplreview.csv)"
    )
    parser.add_argument(
        "--cs", 
        required=True, 
        help="Path to the clean sheet CSV (e.g., cs_1621.csv)"
    )
    
    args = parser.parse_args()
    # Data loading from FPL API
    players, teams, positions = fpl_data.fetch_api_data()
    # Load external helper datasets
    print(f"Loading CSVs from:\n  Minutes: {args.minutes}\n  Clean Sheets: {args.cs}")
    try:
        minutes_df, minutes_map = fpl_data.load_and_prep_minutes(args.minutes)
        cs_df = fpl_data.load_and_prep_clean_sheets(args.cs)
    except FileNotFoundError as e:
        print(f"\nError: Could not find file. {e}")
        sys.exit(1)

    # Use helper method to calculate the probabilities how many goals a team concedes
    # based on their clean sheet probability
    for gw in range(16, 22):
        col = f"xClean_sheets{gw}"
        p23, p4plus = fpl_model.calculate_opp_goal_probs(cs_df[col])
        cs_df[f"x2_{gw}"] = p23
        cs_df[f"x4_{gw}"] = p4plus
    # Merge all dataaframes into a master dataframe
    players_full = fpl_data.build_master_dataframe(
        players, teams, positions, minutes_df, minutes_map, cs_df
    )

    # Data preprocessing
    players_full = fpl_model.prepare_data_for_modeling(players_full)
    # Load the data for past seasons for each player
    past_data = fpl_data.get_player_history(players_full['player_id'])
    
    # Merge the history dataset with the current season dataset
    players_full = players_full.merge(past_data, left_on="player_id", right_index=True, how="left")
    players_full = fpl_model.fill_missing_history_with_current(players_full)

    # Run projections for the next 6 gameweeks
    players_full = fpl_model.run_projections(players_full)
    
    # Apply decay to make far future gameweek projections less important than
    # near future projections with less variance
    players_full = fpl_model.apply_decay(players_full, decay_rate=0.92)

    # Display 10 top players in each position
    print_top_players(players_full, n=10)

    
    # A. Wildcard Solver
    wc_team = fpl_solver.solve_fpl_team(players_full, mode="wildcard")
    print("\n############################################")
    print("   OPTIMIZED WILDCARD TEAM (GW 16-21)   ")
    print("   Obj: Max Decayed Pts | Bench Cost >= 18.5m ")
    print("############################################")
    if wc_team is not None:
        print(f"Total Cost: {wc_team['Cost'].sum():.1f}")
        print(f"Projected Points (Decayed): {wc_team['Points'].sum():.1f}")
        print(wc_team.to_string(index=False))

    # B. Free Hit Solver
    fh_team = fpl_solver.solve_fpl_team(players_full, mode="free_hit")
    print("\n\n############################################")
    print("      OPTIMIZED FREE HIT TEAM (GW 16)       ")
    print("      Obj: Max GW16 Pts | No Bench Limit    ")
    print("############################################")
    if fh_team is not None:
        print(f"Total Cost: {fh_team['Cost'].sum():.1f}")
        starter_pts = fh_team[fh_team['Role']=='Starter']['Points'].sum()
        print(f"Projected Starter Points (GW16): {starter_pts:.1f}")
        print(fh_team.to_string(index=False))

if __name__ == "__main__":
    main()