# OKX MCP Server

A Model Context Protocol server that provides real-time cryptocurrency price data from OKX exchange.

## Features

This MCP server connects to the OKX API to provide cryptocurrency price information through a simple tool interface. It includes comprehensive error handling, request logging, and rate limiting via OKX's API.

### Tools

#### `get_candlesticks`

Retrieves historical candlestick (OHLCV) data for any instrument on OKX.

- **Input**:
  - `instrument`: String (required) - Instrument ID (e.g. "BTC-USDT")
  - `bar`: String (optional) - Time interval (e.g. "1m", "5m", "1H", "1D"), default "1m"
  - `limit`: Number (optional) - Number of candlesticks to return (max 100), default 100
- **Output**: Array of JSON objects, each containing:
  - `timestamp`: ISO timestamp of the candlestick
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `volume`: Trading volume
  - `volumeCurrency`: Volume in currency terms

Example usage:

```json
[
  {
    "timestamp": "2025-03-07T17:00:00.000Z",
    "open": "87242.8",
    "high": "87580.2",
    "low": "86548.0",
    "close": "87191.8",
    "volume": "455.72150427",
    "volumeCurrency": "39661166.242091111"
  }
]
```

#### `get_price`

Fetches the latest price and 24-hour market data for any instrument on OKX.

- **Input**:
  - `instrument`: String (required) - Instrument ID (e.g. "BTC-USDT")
- **Output**: JSON object containing:
  - `instrument`: The requested instrument ID
  - `lastPrice`: Latest trade price
  - `bid`: Current best bid price
  - `ask`: Current best ask price
  - `high24h`: 24-hour high price
  - `low24h`: 24-hour low price
  - `volume24h`: 24-hour trading volume
  - `timestamp`: ISO timestamp of the data

Example usage:

```json
{
  "instrument": "BTC-USDT",
  "lastPrice": "65432.1",
  "bid": "65432.0",
  "ask": "65432.2",
  "high24h": "66000.0",
  "low24h": "64000.0",
  "volume24h": "1234.56",
  "timestamp": "2024-03-07T17:22:28.000Z"
}
```

## Development

Install dependencies:

```bash
npm install
```

Build the server:

```bash
npm run build
```

For development with auto-rebuild:

```bash
npm run watch
```

## Installation

To use with Claude Desktop or VSCode, add the server config to your MCP settings:

macOS (VSCode):

```bash
~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
```

macOS (Claude Desktop):

```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

Windows (VSCode):

```bash
%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
```

Windows (Claude Desktop):

```bash
%APPDATA%/Claude/claude_desktop_config.json
```

Configuration:

```json
{
  "mcpServers": {
    "okx": {
      "command": "node",
      "args": ["/path/to/okx-mcp-server/build/index.js"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Error Handling

The server implements comprehensive error handling:

- Network errors are captured and returned with context
- Invalid instrument IDs return appropriate error messages
- API rate limits are respected through axios timeout configuration
- All errors are logged for debugging purposes
