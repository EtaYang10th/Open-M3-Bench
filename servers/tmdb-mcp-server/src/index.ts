#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import axios from 'axios';

const TMDB_ACCESS_TOKEN = process.env.TMDB_ACCESS_TOKEN;
if (!TMDB_ACCESS_TOKEN) {
  throw new Error('TMDB_ACCESS_TOKEN environment variable is required');
}

class TMDBServer {
  private server: Server;
  private axiosInstance;

  constructor() {
    this.server = new Server(
      {
        name: 'tmdb-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.axiosInstance = axios.create({
      baseURL: 'https://api.themoviedb.org/3',
      headers: {
        'Authorization': `Bearer ${TMDB_ACCESS_TOKEN}`,
        'accept': 'application/json'
      },
    });

    this.setupToolHandlers();
    
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'search_movies',
          description: `Search for movies using The Movie Database API.

Args:
  query (string): Search query text. Required.
  year (number, optional): Filter by release year.
  page (number, optional): Page number (default: 1).

Returns:
  text: JSON string of TMDB /search/movie response.`,
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query',
              },
              year: {
                type: 'number',
                description: 'Filter by release year (optional)',
              },
              page: {
                type: 'number',
                description: 'Page number (default: 1)',
              },
            },
            required: ['query'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name !== 'search_movies') {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }

      const args = request.params.arguments as {
        query: string;
        year?: number;
        page?: number;
      };
      const { query, year, page = 1 } = args;

      try {
        const response = await this.axiosInstance.get('/search/movie', {
          params: {
            query,
            year,
            page,
          },
        });

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(response.data, null, 2),
            },
          ],
        };
      } catch (error) {
        if (axios.isAxiosError(error)) {
          return {
            content: [
              {
                type: 'text',
                text: `TMDB API error: ${
                  error.response?.data.message ?? error.message
                }`,
              },
            ],
            isError: true,
          };
        }
        throw error;
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('TMDB MCP server running on stdio');
  }
}

const server = new TMDBServer();
server.run().catch(console.error);
