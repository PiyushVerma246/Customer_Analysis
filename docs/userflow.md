# RetailIQ User Flow Documentation

This document explains the step-by-step user journey through the RetailIQ platform and how data flows through the system.

## 1. Authentication Flow
- **Entry**: The user lands on the Home Page (`/`). If not authenticated, they only see marketing copy and Call-to-Action buttons.
- **Login/Register**: Users access `/login` or `/register`.
- **Demo Access**: For demonstration purposes, users can click **One-Click Demo Login** to automatically log in as `testuser`.
- **Session**: Once authenticated, a secure Flask session is created, granting access to the CRM dashboard.

## 2. Onboarding & Data Import
When an authenticated user logs in for the first time with an empty database:
- **Empty State**: The dashboard displays a "No Customer Data Found" state.
- **Action**: The user clicks **Generate Demo Data** (or Upload Dataset).
- **Processing**: The `/api/generate-demo-data` endpoint creates dummy customers and transactions, mimicking a real import.
- **Onboarding Card**: Upon refresh, the user sees an onboarding card directing them to the four core modules. This card is dismissed and saved to `localStorage`.

## 3. CRM Workflows

### A. Exploring the Dashboard
- **Top Metrics**: Total Revenue, Total Customers, Total Orders, and At-Risk count are immediately visible.
- **Charts**: Users toggle between Customer Count, Revenue, and Average Spending within the Categories donut chart. Clicking a chart slice deep-links to the filtered Customer Directory.

### B. Searching for Customers
- **Global Search**: Users type in the top-navigation search bar. This queries `/api/search-customers` with the input (Name, ID, or Phone) and displays instant autocomplete results.
- **Directory Search**: On the `/customers` page, users filter the paginated table by Category (e.g., "Best Customers"), Retention Risk, or text query.

### C. Viewing Customer Profiles
- **Profile View (`/customer/<id>`)**: When a specific customer is selected, the application queries the SQLite database for:
  - Personal Details
  - Lifetime Value & Order History
  - AI Recommended Products (frequently bought together)
  - Next Expected Purchase Date
- **Action**: Retailers use this view when a customer calls or enters the store, immediately seeing if they are VIP, what to cross-sell, and if they are at risk of churning.

### D. Entering New Transactions
- **Customer Entry (`/add`)**: 
  - The retailer selects "New Customer" or "Returning Customer".
  - They input the purchase amount, product categories, and date.
  - The system dynamically updates the customer's RFM (Recency, Frequency, Monetary) metrics.
  - *Data Flow*: Form -> `POST /add` -> Insert into `transactions` table -> Background ML Pipeline (optional) -> UI Update.

## 4. Theme & Preferences Flow
- **Theme Switcher**: Users click the top-right monitor icon to select Light, Dark, or System theme.
- **Storage**: The preference is saved in `localStorage`.
- **Render**: The JavaScript immediately sets `data-theme="light"` or `dark` on the `<html>` element. CSS variables adapt, and Chart.js instances are forcefully re-rendered to match the new text colors.
