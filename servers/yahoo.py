"""
Yahoo Finance MCP Server.
Provides stock data, financial statements, options, holders, and analyst recommendations via yfinance.
Adapted from https://github.com/shzhiqi/yahoo-finance-mcp/blob/main/server.py
"""

import json
import pandas as pd
import yfinance as yf
from enum import Enum
from mcp.server.fastmcp import FastMCP

# ======================
#   Enums
# ======================
class FinancialType(str, Enum):
    """Financial statement type."""
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    """Holder info type."""
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    """Recommendation info type."""
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


# ======================
#   Server
# ======================
server = FastMCP("yfinance")


# ======================
#   Tools
# ======================

@server.tool()
def get_historical_stock_prices(
    ticker: str, start_date: str, end_date: str, interval: str = "1d"
) -> str:
    """
      Get historical stock price OHLCV data for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        interval (str): Price interval such as 1d or 1h.
      Returns:
        result (str): JSON array of records with date and price fields.
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting historical stock prices for {ticker}: {e}"

    hist_data = company.history(start=start_date, end=end_date, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    return hist_data.to_json(orient="records", date_format="iso")


@server.tool()
def get_stock_info(ticker: str) -> str:
    """
      Get general stock and company information for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
      Returns:
        result (str): JSON object of Yahoo Finance info fields.
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
        info = company.info
        return json.dumps(info)
    except Exception as e:
        return f"Error: getting stock information for {ticker}: {e}"


@server.tool()
def get_yahoo_finance_news(ticker: str) -> str:
    """
      Get recent news items for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
      Returns:
        result (str): Plain-text list of titles and links for each news item.
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
        news = company.news
    except Exception as e:
        return f"Error: getting news for {ticker}: {e}"

    if not news:
        return f"No news found for company {ticker}."

    lines = []
    for n in news:
        title = n.get("title", "")
        link = n.get("link", "")
        lines.append(f"- {title}\n  {link}")
    return "\n\n".join(lines)


@server.tool()
def get_stock_actions(ticker: str) -> str:
    """
      Get dividend and split actions for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
      Returns:
        result (str): JSON array of corporate action records.
    """
    try:
        company = yf.Ticker(ticker)
        actions_df = company.actions
        actions_df = actions_df.reset_index(names="Date")
        return actions_df.to_json(orient="records", date_format="iso")
    except Exception as e:
        return f"Error: getting stock actions for {ticker}: {e}"


@server.tool()
def get_financial_statement(ticker: str, financial_type: str) -> str:
    """
      Get a selected financial statement for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
        financial_type (str): Statement type such as income_stmt or balance_sheet.
      Returns:
        result (str): JSON array of statement records keyed by date.
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting financial statement for {ticker}: {e}"

    if financial_type == FinancialType.income_stmt:
        financial_statement = company.income_stmt
    elif financial_type == FinancialType.quarterly_income_stmt:
        financial_statement = company.quarterly_income_stmt
    elif financial_type == FinancialType.balance_sheet:
        financial_statement = company.balance_sheet
    elif financial_type == FinancialType.quarterly_balance_sheet:
        financial_statement = company.quarterly_balance_sheet
    elif financial_type == FinancialType.cashflow:
        financial_statement = company.cashflow
    elif financial_type == FinancialType.quarterly_cashflow:
        financial_statement = company.quarterly_cashflow
    else:
        return f"Error: invalid financial type {financial_type}"

    result = []
    for column in financial_statement.columns:
        date_str = column.strftime("%Y-%m-%d") if isinstance(column, pd.Timestamp) else str(column)
        date_obj = {"date": date_str}
        for index, value in financial_statement[column].items():
            date_obj[index] = None if pd.isna(value) else value
        result.append(date_obj)

    return json.dumps(result)


@server.tool()
def get_holder_info(ticker: str, holder_type: str) -> str:
    """
      Get holder or ownership information for a ticker from Yahoo Finance.
      Args:
        ticker (str): Stock ticker symbol.
        holder_type (str): Holder info type such as major_holders or institutional_holders.
      Returns:
        result (str): JSON array of holder records.
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting holder info for {ticker}: {e}"

    try:
        if holder_type == HolderType.major_holders:
            return company.major_holders.reset_index(names="metric").to_json(orient="records")
        if holder_type == HolderType.institutional_holders:
            return company.institutional_holders.to_json(orient="records")
        if holder_type == HolderType.mutualfund_holders:
            return company.mutualfund_holders.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_transactions:
            return company.insider_transactions.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_purchases:
            return company.insider_purchases.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_roster_holders:
            return company.insider_roster_holders.to_json(orient="records", date_format="iso")
        return f"Error: invalid holder type {holder_type}."
    except Exception as e:
        return f"Error: fetching holder info: {e}"


@server.tool()
def get_option_expiration_dates(ticker: str) -> str:
    """
      Get available option expiration dates for a ticker.
      Args:
        ticker (str): Stock ticker symbol.
      Returns:
        result (str): JSON list of expiration date strings.
    """
    company = yf.Ticker(ticker)
    try:
        return json.dumps(company.options)
    except Exception as e:
        return f"Error: getting option expiration dates for {ticker}: {e}"


@server.tool()
def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    """
      Get an option chain for a ticker, expiration date, and calls or puts side.
      Args:
        ticker (str): Stock ticker symbol.
        expiration_date (str): Expiration date in YYYY-MM-DD format.
        option_type (str): Which side to return, 'calls' or 'puts'.
      Returns:
        result (str): JSON array of option contracts.
    """
    company = yf.Ticker(ticker)
    try:
        if expiration_date not in company.options:
            return f"No options available for {expiration_date}."
        option_chain = company.option_chain(expiration_date)
        if option_type == "calls":
            return option_chain.calls.to_json(orient="records", date_format="iso")
        elif option_type == "puts":
            return option_chain.puts.to_json(orient="records", date_format="iso")
        return f"Invalid option type: {option_type}. Use 'calls' or 'puts'."
    except Exception as e:
        return f"Error: getting option chain for {ticker}: {e}"


@server.tool()
def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    """
      Get analyst recommendations or upgrades/downgrades for a ticker.
      Args:
        ticker (str): Stock ticker symbol.
        recommendation_type (str): 'recommendations' or 'upgrades_downgrades'.
        months_back (int): How many months of history to include.
      Returns:
        result (str): JSON array of recommendation records.
    """
    company = yf.Ticker(ticker)
    try:
        if recommendation_type == RecommendationType.recommendations:
            return company.recommendations.to_json(orient="records")
        elif recommendation_type == RecommendationType.upgrades_downgrades:
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["GradeDate"] >= cutoff_date]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
        else:
            return f"Invalid recommendation_type: {recommendation_type}"
    except Exception as e:
        return f"Error: getting recommendations for {ticker}: {e}"


# ======================
#   Entry Point
# ======================
if __name__ == "__main__":
    server.run(transport="stdio")
