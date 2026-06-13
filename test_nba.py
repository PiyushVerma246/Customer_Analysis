import pandas as pd
from nba_engine import NBAEngine

def test_nba_engine():
    # Load actual data to simulate real environment
    try:
        df_customers = pd.read_csv("customer_segments.csv")
        df_rules = pd.read_csv("recommendation_rules.csv")
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return

    engine = NBAEngine(df_customers, df_rules)

    # Test case 1: VIP Customer (from CSV inspection: 12415.0)
    vip_id = 12415
    vip_actions = engine.get_actions(vip_id)
    print(f"\n--- Actions for VIP Customer #{vip_id} ---")
    for a in vip_actions:
        print(f"[{a['priority']}] {a['title']}: {a['impact_score']}")
    
    assert len(vip_actions) == 3
    assert vip_actions[0]['title'] == "Exclusive Loyalty Reward"

    # Test case 2: At Risk Customer (from CSV inspection: 12346.0)
    risk_id = 12346
    risk_actions = engine.get_actions(risk_id)
    print(f"\n--- Actions for At-Risk Customer #{risk_id} ---")
    for a in risk_actions:
        print(f"[{a['priority']}] {a['title']}: {a['impact_score']}")

    assert len(risk_actions) == 3
    assert risk_actions[0]['title'] == "Win Back Campaign"

    # Test case 3: Regular Customer (from CSV inspection: 12347.0)
    reg_id = 12347
    reg_actions = engine.get_actions(reg_id)
    print(f"\n--- Actions for Regular Customer #{reg_id} ---")
    for a in reg_actions:
        print(f"[{a['priority']}] {a['title']}: {a['impact_score']}")

    assert len(reg_actions) == 3
    assert reg_actions[0]['title'] == "Upsell Recommendation"

    # Test case 4: Non-existent Customer
    none_actions = engine.get_actions(99999)
    assert none_actions == []
    print("\nTest passed: Non-existent customer handled correctly.")

    print("\nAll NBA Engine tests PASSED.")

if __name__ == "__main__":
    test_nba_engine()
