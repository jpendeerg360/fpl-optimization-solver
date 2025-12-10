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

def print_projections(df, sort_by="total_decayed_points", n=None, gw=None, position=None):
    """
    Prints player stats with custom sorting and filtering.
    Default sort is by total decayed points
    --sort price to sort by price
    --sort gw --gw 16 to sort for just gw 16
    --top 0 to display all players, --top 50 to display top 50 for example
    --position gk, def, mid, or forward to subset by position
    """
    # 1. Determine sort column and display columns
    # Start with the standard clean columns
    cols_to_show = ["first_name", "second_name", "team_name", "position", "now_cost", "total_decayed_points"]
    
    if sort_by == "cost":
        sort_col = "now_cost"
        display_name = "Price"
    elif sort_by == "gw" and gw:
        sort_col = f"expected_points{gw}"
        display_name = f"GW{gw} Pts"
        # Add ONLY the relevant GW column to the display
        cols_to_show.append(sort_col)
    else:
        sort_col = "total_decayed_points"
        display_name = "Decayed Total"

    # 2. Filter by Position (Optional)
    subset = df.copy()
    if position:
        pos_map = {
            "g": "Goalkeeper", "gk": "Goalkeeper", "goalkeeper": "Goalkeeper",
            "d": "Defender", "def": "Defender", "defender": "Defender",
            "m": "Midfielder", "mid": "Midfielder", "midfielder": "Midfielder",
            "f": "Forward", "fwd": "Forward", "forward": "Forward"
        }
        clean_pos = pos_map.get(position.lower(), position.capitalize())
        subset = subset[subset["position"] == clean_pos]
        display_name += f" ({clean_pos}s only)"

    # 3. Sort
    if sort_col not in subset.columns:
        print(f"Error: Column '{sort_col}' not found.")
        return
    
    subset = subset.sort_values(by=sort_col, ascending=False)
    
    # 4. Limit rows
    if n is not None:
        subset = subset.head(n)

    # 5. Display
    print(f"\n{'='*60}")
    print(f"   CUSTOM PROJECTIONS: {display_name}")
    print(f"{'='*60}")
    
    if subset.empty:
        print("No players found matching criteria.")
    else:
        # float_format="%.1f" keeps numbers clean (e.g. 5.4 instead of 5.43211)
        print(subset[cols_to_show].to_string(index=False, float_format="%.1f"))

def main():
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Run FPL Optimization Model")
    
    # Core Files
    parser.add_argument("--minutes", required=True, help="Path to minutes CSV")
    parser.add_argument("--cs", required=True, help="Path to clean sheet CSV")
    
    # Optional Customization Flags
    parser.add_argument("--sort", choices=["decayed", "price", "gw"], 
                        help="Sort mode for custom view.")
    parser.add_argument("--gw", type=int, help="Gameweek number (required if sorting by 'gw')")
    parser.add_argument("--top", type=int, default=10, help="Number of players to show.")
    parser.add_argument("--position", help="Filter by position (e.g. 'Forward', 'Mid', 'Def', 'GK')")

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

    is_custom_run = args.sort or args.position or args.gw

    if is_custom_run:
        limit = args.top if args.top > 0 else None
        print_projections(
            players_full, 
            sort_by=args.sort if args.sort else "decayed", 
            n=limit, 
            gw=args.gw,
            position=args.position
        )
        return

    else:
        print_top_players(players_full, n=args.top)

        # Optimization Solvers
        wc_team = fpl_solver.solve_fpl_team(players_full, mode="wildcard")
        print("\n############################################")
        print("   OPTIMIZED WILDCARD TEAM (GW 16-21)   ")
        print("   Obj: Max Decayed Pts | Bench Cost >= 18.5m ")
        print("############################################")
        if wc_team is not None:
            print(f"Total Cost: {wc_team['Cost'].sum():.1f}")
            print(f"Projected Points (Decayed): {wc_team['Points'].sum():.1f}")
            print(wc_team.to_string(index=False))

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