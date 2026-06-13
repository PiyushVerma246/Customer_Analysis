import pandas as pd
import numpy as np

class NBAEngine:
    """
    Customer Next Best Action (NBA) Engine.
    
    This engine analyzes customer RFM metrics and segments to recommend
    strategic business actions with calculated impact scores and priorities.
    Using percentile-based normalization to ensure fair score distribution.
    """

    def __init__(self, customers_df, rules_df):
        """
        Initialize the NBA Engine with customer and recommendation data.
        
        Args:
            customers_df (pd.DataFrame): Dataframe containing CustomerID, Recency, 
                                        Frequency, Monetary, and Segment.
            rules_df (pd.DataFrame): Dataframe containing market basket association rules.
        """
        self.df_customers = customers_df.copy()
        self.df_rules = rules_df
        
        # Calculate Percentile Ranks for RFM metrics (0 to 100)
        if not self.df_customers.empty:
            # Recency: Lower is better, so we rank ascending and then invert
            # rank(pct=True) returns 0.0 to 1.0
            # To make lower recency = higher score: 1.0 - rank
            self.df_customers['r_score'] = (1.0 - self.df_customers['Recency'].rank(pct=True)) * 100
            
            # Frequency: Higher is better
            self.df_customers['f_score'] = self.df_customers['Frequency'].rank(pct=True) * 100
            
            # Monetary: Higher is better
            self.df_customers['m_score'] = self.df_customers['Monetary'].rank(pct=True) * 100
            
            # Create a lookup dictionary for performance
            self.score_lookup = self.df_customers.set_index('CustomerID')[['r_score', 'f_score', 'm_score']].to_dict('index')
        else:
            self.score_lookup = {}

    def get_actions(self, customer_id):
        """
        Generate a list of recommended actions for a specific customer.
        
        Args:
            customer_id (int): The unique ID of the customer.
            
        Returns:
            list: A list of dictionaries, each representing an action.
                  Returns an empty list if the customer ID is not found.
        """
        # 1. Validate customer existence
        cust_row = self.df_customers[self.df_customers['CustomerID'] == customer_id]
        if cust_row.empty:
            return []

        customer = cust_row.iloc[0]
        segment = customer['Segment']

        # 2. Retrieve Percentile-based Factors (0-100)
        scores = self.score_lookup.get(customer_id, {'r_score': 0, 'f_score': 0, 'm_score': 0})
        recency_factor = scores['r_score']
        frequency_factor = scores['f_score']
        monetary_factor = scores['m_score']

        # 3. Calculate Impact Score
        # Weights: 40% Recency, 30% Frequency, 30% Monetary
        impact_score = (0.4 * recency_factor) + (0.3 * frequency_factor) + (0.3 * monetary_factor)
        impact_score = round(min(100, max(0, impact_score)), 2)

        # 4. Determine Priority Level
        if impact_score >= 80:
            priority = "High"
        elif impact_score >= 60:
            priority = "Medium"
        else:
            priority = "Low"

        # 5. Generate Actions based on Segment
        actions = []
        
        if segment == "VIP":
            actions = [
                {
                    "title": "Exclusive Loyalty Reward",
                    "description": "Grant access to the Elite Tier rewards program and assign a dedicated account manager.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Early Product Access",
                    "description": "Invite to the pre-launch event for the upcoming luxury collection 48 hours before general release.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Premium Cross-Sell Recommendation",
                    "description": "Leverage high purchase power to recommend complementary high-margin accessory sets.",
                    "priority": priority,
                    "impact_score": impact_score
                }
            ]
            
        elif segment == "Regular":
            actions = [
                {
                    "title": "Upsell Recommendation",
                    "description": "Suggest a premium version of their most frequently purchased item category.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Increase Purchase Frequency Campaign",
                    "description": "Provide a 'Buy 3, Get 1 Free' voucher valid for the next 14 days to encourage a return visit.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Product Bundle Suggestion",
                    "description": "Create a personalized bundle based on common association rules to increase average basket size.",
                    "priority": priority,
                    "impact_score": impact_score
                }
            ]
            
        elif segment == "At Risk":
            actions = [
                {
                    "title": "Win Back Campaign",
                    "description": "Send a 'We Miss You' email with a significant 25% discount code on their favorite category.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Personalized Discount Offer",
                    "description": "Targeted mobile app notification offering free shipping for the next 48 hours.",
                    "priority": priority,
                    "impact_score": impact_score
                },
                {
                    "title": "Retention Outreach",
                    "description": "Customer service follow-up call to identify potential service friction points.",
                    "priority": priority,
                    "impact_score": impact_score
                }
            ]

        return actions
