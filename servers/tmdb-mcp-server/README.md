# TMDB MCP Server

A Model Context Protocol (MCP) server that provides access to The Movie Database (TMDB) API. This server enables AI assistants to search and retrieve movie information through the MCP interface.

## Features

- Search movies by title, year, and other criteria
- Retrieve detailed movie information
- Easy integration with MCP-compatible AI assistants

## Prerequisites

- Node.js >= 18
- TMDB API key (get one from [TMDB](https://www.themoviedb.org/documentation/api))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rakeshgangwar/tmdb-mcp-server.git
cd tmdb-mcp-server
```

2. Install dependencies:
```bash
npm install
```

3. Build the server:
```bash
npm run build
```

## Configuration

Configure the MCP server in your MCP settings file (typically `cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "tmdb": {
      "command": "node",
      "args": ["/path/to/tmdb-mcp-server/dist/index.js"],
      "env": {
        "TMDB_API_KEY": "your-api-key-here"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

Replace `your-api-key-here` with your TMDB API key and replace `/path/to/` with actual path.

## Available Tools

### search_movies
Search for movies using The Movie Database API.

Parameters:
- `query` (required): Search query string
- `year` (optional): Filter by release year
- `page` (optional): Page number (default: 1)

Example:
```javascript
{
  "query": "Inception",
  "year": 2010,
  "page": 1
}
```

## Development

1. Make your changes in the `src` directory
2. Build the project:
```bash
npm run build
```

3. Test your changes by configuring the MCP server in your settings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TMDB API](https://www.themoviedb.org/documentation/api) for providing the movie database API
- [Model Context Protocol](https://github.com/modelcontextprotocol) for the MCP SDK
