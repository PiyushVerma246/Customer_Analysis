# RetailIQ API Reference

This document describes the internal REST APIs used by the frontend to fetch dynamic data asynchronously.

All endpoints require an active authenticated session (`@login_required`).

## 1. Global Customer Search
**Endpoint**: `GET /api/search-customers`

Searches the entire customer database by ID, Name, Phone, Category, or Retention Risk.

- **Query Parameters**:
  - `q` (string): The search query. Needs to be at least 2 characters long for optimal performance.
- **Response**:
  ```json
  [
    {
      "customer_id": "CUST0001",
      "name": "Jane Doe"
    },
    {
      "customer_id": "CUST0002",
      "name": "John Smith"
    }
  ]
  ```

## 2. Dashboard Statistics
**Endpoint**: `GET /api/dashboard-stats`

Provides real-time Key Performance Indicators (KPIs) and segment distributions for rendering dashboard charts.

- **Response**:
  ```json
  {
    "status": "success",
    "kpis": {
      "total_revenue": 1054320.50,
      "total_customers": 4338,
      "total_orders": 25900,
      "total_products": 3820
    },
    "seg_distribution": {
      "Best Customers": 842,
      "Repeat Customers": 1200,
      "Standard Customers": 1500,
      "Customers You May Lose": 796
    }
  }
  ```

## 3. Generate Demo Data
**Endpoint**: `POST /api/generate-demo-data`

Populates the SQLite database with realistic sample customers, transactions, and predictions. Intended for use in fresh environments or empty states to demonstrate platform capabilities.

- **Response**:
  ```json
  {
    "status": "success",
    "message": "Demo data generated successfully."
  }
  ```
  *(Returns 500 status on failure)*
