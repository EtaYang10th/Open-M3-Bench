[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README_es.md)

# NASA-MCP. Integration via MCP with NASA APIs

**NASA is the National Aeronautics and Space Administration of the United States.**

**NASA-MCP** allows you to retrieve astronomical data, space weather information, Earth imagery, and more from NASA's APIs directly from Claude AI and other MCP compatible clients, using the **Model Context Protocol (MCP)**.

NASA-MCP is an MCP server that exposes tools enabling LLMs to query data from various NASA APIs, including APOD (Astronomy Picture of the Day), Asteroids NeoWs, DONKI (Space Weather Database), Earth imagery, EPIC (Earth Polychromatic Imaging Camera), and Exoplanet data.

It includes secure handling of API keys and proper error management for all API requests.

## Key Features

- Access to **Astronomy Picture of the Day (APOD)** with explanations and imagery
- Query **Near Earth Objects** data and asteroid information
- Retrieve **Space Weather** data from DONKI, including solar flares, geomagnetic storms, and more
- Get **Earth imagery** from Landsat 8 satellite for specific coordinates
- Access **EPIC** camera images showing the full Earth disk
- Query the **Exoplanet Archive** database for information about planets outside our solar system

## Installation

### Installing via Smithery

To install NASA API Integration Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@AnCode666/nasa-mcp):

```bash
npx -y @smithery/cli install @AnCode666/nasa-mcp --client claude
```

### Install with uv

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installing uv

The first step is to install `uv`, a package manager for Python.  
**It can be installed from the command line**.

On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:  

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You can also install it with pip:  

```bash
pip install uv
```

For more information about installing uv, visit the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Integration with clients like Claude for Desktop

Once **uv** is installed, you can use the MCP server from any compatible client such as Claude for Desktop, in which case the steps to follow are:

1. Go to **Claude > Settings > Developer > Edit Config > `claude_desktop_config.json`**
2. Add the following block inside `"mcpServers"`:

```json
"nasa-mcp": {
    "command": "uvx",
    "args": [
        "nasa_mcp"
    ],
    "env": {
        "NASA_API_KEY": "YOUR_NASA_API_KEY"
    }
}
```

3. Get a free API key from NASA at: <https://api.nasa.gov/>
4. Replace `YOUR_NASA_API_KEY` with your actual API key (leave the quotes). You can also use "DEMO_KEY" for limited testing.
5. If you already have another MCP server configured, separate each with a comma `,`.

In general, to integrate it into any other MCP-compatible client such as Cursor, CODEGPT, or Roo Code, simply go to the MCP server configuration of your client and add the same code block.

## Usage Examples

Once properly configured, you can ask things like:

- "Show me today's astronomy picture of the day"
- "Find asteroids that will pass near Earth in the next week"
- "Get information about solar flares from January 2023"
- "Show me Earth imagery for coordinates 29.78, -95.33"
- "Find exoplanets in the habitable zone"

## DISTRIBUTIONS

### Smithery

[![smithery badge](https://smithery.ai/badge/@AnCode666/nasa-mcp)](https://smithery.ai/server/@AnCode666/nasa-mcp)

### MCP Review

[MCP review certified](https://mcpreview.com/mcp-servers/ancode666/nasa-mcp)
