import pulp
import pandas as pd

def solve_fpl_team(df_input, mode="wildcard"):
    """
    Solves for the best FPL team using Linear Programming.
    """
    # Set up dictionary
    df = df_input.copy()
    if "player_id" in df.columns:
        df = df.set_index("player_id")
    
    ids = df.index.tolist()
    
    names = (df["first_name"] + " " + df["second_name"]).to_dict()
    teams = df["team_name"].to_dict()
    positions = df["position"].to_dict()
    costs = df["now_cost"].to_dict()
    
    wc_points = df["total_decayed_points"].to_dict()
    fh_points = df["expected_points16"].to_dict()
    
    # Initialize Problem
    prob = pulp.LpProblem(f"FPL_{mode}", pulp.LpMaximize)

    # -- Variables --
    squad_vars = pulp.LpVariable.dicts("Squad", ids, cat="Binary")   # 1 if in 15-man squad
    starter_vars = pulp.LpVariable.dicts("Starter", ids, cat="Binary") # 1 if in Starting XI

    # -- Objective Function --
    if mode == "wildcard":
        # Wildcard: Maximize Starters + 50% of Bench Points
        prob += pulp.lpSum([wc_points[i] * starter_vars[i] for i in ids]) + \
                0.5 * pulp.lpSum([wc_points[i] * (squad_vars[i] - starter_vars[i]) for i in ids])
    else:
        # Free Hit: Maximize Starting XI Points ONLY
        prob += pulp.lpSum([fh_points[i] * starter_vars[i] for i in ids])

    # 15 players in a squad and 11 starters
    prob += pulp.lpSum([squad_vars[i] for i in ids]) == 15
    prob += pulp.lpSum([starter_vars[i] for i in ids]) == 11

    for i in ids:
        prob += starter_vars[i] <= squad_vars[i]

    # 100 million budget
    prob += pulp.lpSum([costs[i] * squad_vars[i] for i in ids]) <= 100

    # Bench has to be at least 18.5 million
    if mode == "wildcard":
        bench_cost = pulp.lpSum([costs[i] * squad_vars[i] for i in ids]) - \
                     pulp.lpSum([costs[i] * starter_vars[i] for i in ids])
        prob += bench_cost >= 18.5

    # A maximum of three players for each team
    unique_teams = set(teams.values())
    for t in unique_teams:
        team_ids = [k for k, v in teams.items() if v == t]
        prob += pulp.lpSum([squad_vars[i] for i in team_ids]) <= 3

    # 2 gk, 5 def, 5 mid, 2 forwards
    prob += pulp.lpSum([squad_vars[i] for i in ids if positions[i] == "Goalkeeper"]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in ids if positions[i] == "Defender"]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in ids if positions[i] == "Midfielder"]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in ids if positions[i] == "Forward"]) == 3

    # 1 starting gk, at least 3 defenders, at least 1 forward starting
    prob += pulp.lpSum([starter_vars[i] for i in ids if positions[i] == "Goalkeeper"]) == 1
    prob += pulp.lpSum([starter_vars[i] for i in ids if positions[i] == "Defender"]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in ids if positions[i] == "Forward"]) >= 1

    # -- Solve --
    print(f"Solving for {mode}...")
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        print("No optimal solution found.")
        return None

    # -- Format Results --
    selected = []
    for i in ids:
        if pulp.value(squad_vars[i]) == 1:
            role = "Starter" if pulp.value(starter_vars[i]) == 1 else "Bench"
            selected.append({
                "Role": role,
                "Name": names[i],
                "Team": teams[i],
                "Pos": positions[i],
                "Cost": costs[i],
                "Points": fh_points[i] if mode == "free_hit" else wc_points[i]
            })

    res_df = pd.DataFrame(selected)
    
    # Sorting for readable output
    role_map = {"Starter": 0, "Bench": 1}
    pos_map = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
    res_df["r_sort"] = res_df["Role"].map(role_map)
    res_df["p_sort"] = res_df["Pos"].map(pos_map)
    res_df = res_df.sort_values(by=["r_sort", "p_sort"]).drop(columns=["r_sort", "p_sort"])

    return res_df