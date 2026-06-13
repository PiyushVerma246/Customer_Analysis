# RetailIQ Database Schema

RetailIQ uses a relational SQLite database (`database/retail.db`) to serve as the single source of truth for runtime application state.

## 1. `users` Table
Handles platform authentication and security.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-incrementing internal ID |
| `username` | TEXT | UNIQUE, NOT NULL | Login name |
| `password_hash` | TEXT | NOT NULL | Werkzeug-generated password hash |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Account creation time |

## 2. `customers` Table
Stores core identity data for retail customers.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-incrementing internal ID |
| `customer_id` | TEXT | UNIQUE | Permanent business ID (e.g., CUST0001) |
| `name` | TEXT | | Full name (or fallback text) |
| `phone` | TEXT | | Contact number |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

## 3. `transactions` Table
Stores granular purchase history linked to specific customers. RFM metrics are calculated dynamically from this table.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-incrementing internal ID |
| `customer_id` | TEXT | | Foreign Key to `customers.customer_id` |
| `purchase_amount` | REAL | | Total order value |
| `purchase_date` | TIMESTAMP | | Date of transaction |
| `product_purchased` | TEXT | | Product name or category |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record insertion time |

## 4. `predictions` Table
Caches the output of heavy ML pipelines.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | Auto-incrementing internal ID |
| `customer_id` | TEXT | | Foreign Key to `customers.customer_id` |
| `customer_category` | TEXT | | Segmentation label (e.g., VIP, Regular) |
| `retention_risk` | TEXT | | Risk label (e.g., High, Medium, Low) |
| `churn_probability` | REAL | | Probability score (0.0 to 1.0) |
| `prediction_time` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When the model was run |
