import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { DinoXApiClient } from "../dino-x/client.js";
import { BASE_URL, DEFAULT_TIMEOUT, Tool, ToolConfigs } from "../constants/tool.js";
import { z } from 'zod';
import { parseBbox, parsePoseKeypoints } from "../utils/index.js";
import { Config, ResultCategory } from "../types/index.js";
import express from "express";
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';

/**
 * MCP Server class for managing DINO-X MCP server configuration and functionality
 * Specifically designed for Streamable HTTP transport mode
 */
export class MCPStreamHTTPServer {
  private server!: McpServer;
  private config!: Config;
  private api!: DinoXApiClient;

  constructor(customConfig: Config) {
    this.config = customConfig;

    // Create default API client if apiKey is provided in config
    if (this.config.apiKey) {
      this.api = new DinoXApiClient({
        apiKey: this.config.apiKey,
        baseUrl: BASE_URL,
        timeout: DEFAULT_TIMEOUT,
      });
    }

    this.server = new McpServer({
      name: 'dinox-mcp-server',
      version: '1.0.0',
    });

    this.registerTools();
  }

  /**
   * Get API key from request with fallback priority
   * Priority: URL parameter 'key' (if allowed) > server config apiKey
   * @param req Express request object
   * @returns API key string
   * @throws Error if no valid API key found
   */
  private getApiKeyFromRequest(req: express.Request): string {
    const urlApiKey = req.query.key as string;
    if (urlApiKey && typeof urlApiKey === 'string' && urlApiKey.trim()) {
      return urlApiKey.trim();
    }

    throw new Error(
      'API key required. Please provide via URL parameter (?key=your-dinox-api-key)'
    );
  }

  /**
   * Set API client for current request
   * @param apiKey API key string
   */
  private setApiClient(apiKey: string): void {
    this.api = new DinoXApiClient({
      apiKey,
      baseUrl: BASE_URL,
      timeout: DEFAULT_TIMEOUT,
    });
  }

  private registerTools(): void {
    this.registerDetectByTextTool();
    this.registerDetectAllObjectsTool();
    this.registerDetectHumanPoseKeypointsTool();
  }

  private registerDetectByTextTool(): void {
    const { name, description } = ToolConfigs[Tool.DETECT_BY_TEXT];
    this.server.tool(
      name,
      description,
      {
        imageFileUri: z.string().describe("Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted."),
        textPrompt: z.string().describe("Nouns of target objects (English only, avoid adjectives). Use periods to separate multiple categories (e.g., 'person.car.traffic light')."),
        includeDescription: z.boolean().describe("Whether to return a description of the objects detected in the image, but will take longer to process."),
      },
      async (args) => {
        try {
          const { imageFileUri, textPrompt, includeDescription } = args;

          if (!imageFileUri || !textPrompt) {
            return {
              content: [
                {
                  type: 'text',
                  text: 'Image file URI and text prompt are required',
                },
              ],
            }
          }

          const { objects } = await this.api.detectObjectsByText(imageFileUri, textPrompt, includeDescription);

          const categories: ResultCategory = {};
          for (const object of objects) {
            if (!categories[object.category]) {
              categories[object.category] = [];
            }
            categories[object.category].push(object);
          }

          const objectsInfo = objects.map(obj => {
            const bbox = parseBbox(obj.bbox);
            return {
              name: obj.category,
              bbox,
              ...(includeDescription ? {
                description: obj.caption,
              } : {}),
            }
          });

          return {
            content: [
              {
                type: "text",
                text: `Objects detected in image: ${Object.keys(categories).map(cat =>
                  `${cat} (${categories[cat].length})`
                )?.join(', ')}.`
              },
              {
                type: "text",
                text: `Detailed object detection results: ${JSON.stringify((objectsInfo), null, 2)}`
              },
              {
                type: "text",
                text: `Note: The bbox coordinates are in {xmin, ymin, xmax, ymax} format, where the origin (0,0) is at the top-left corner of the image. These coordinates help determine the exact position and spatial relationships of objects in the image.`
              },
            ]
          };
        } catch (error) {
          return {
            content: [
              {
                type: 'text',
                text: `Failed to detect objects from image: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
        }
      }
    )
  }

  private registerDetectAllObjectsTool(): void {
    const { name, description } = ToolConfigs[Tool.DETECT_ALL_OBJECTS];
    this.server.tool(
      name,
      description,
      {
        imageFileUri: z.string().describe("Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted."),
        includeDescription: z.boolean().describe("Whether to return a description of the objects detected in the image, but will take longer to process."),
      },
      async (args) => {
        try {
          const { imageFileUri, includeDescription } = args;

          if (!imageFileUri) {
            return {
              content: [
                {
                  type: 'text',
                  text: 'Image file URI is required',
                },
              ],
            }
          }

          const { objects } = await this.api.detectAllObjects(imageFileUri, includeDescription);
          const categories: ResultCategory = {};

          for (const object of objects) {
            if (!categories[object.category]) {
              categories[object.category] = [];
            }
            categories[object.category].push(object);
          }

          const objectsInfo = objects.map(obj => {
            const bbox = parseBbox(obj.bbox);
            return {
              name: obj.category,
              bbox,
              ...(includeDescription ? {
                description: obj.caption,
              } : {}),
            }
          });

          return {
            content: [
              {
                type: "text",
                text: `Objects detected in image: ${Object.keys(categories).map(cat =>
                  `${cat} (${categories[cat].length})`
                )?.join(', ')}.`
              },
              {
                type: "text",
                text: `Detailed object detection results: ${JSON.stringify(objectsInfo, null, 2)}`
              },
              {
                type: "text",
                text: `Note: The bbox coordinates are in [xmin, ymin, xmax, ymax] format, where the origin (0,0) is at the top-left corner of the image. These coordinates help determine the exact position and spatial relationships of objects in the image.`
              },
            ]
          };
        } catch (error) {
          return {
            content: [
              {
                type: 'text',
                text: `Failed to detect objects from image: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
        }
      }
    )
  }

  private registerDetectHumanPoseKeypointsTool(): void {
    const { name, description } = ToolConfigs[Tool.DETECT_HUMAN_POSE_KEYPOINTS];
    this.server.tool(
      name,
      description,
      {
        imageFileUri: z.string().describe("Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted."),
        includeDescription: z.boolean().describe("Whether to return a description of the objects detected in the image, but will take longer to process."),
      },
      async (args) => {
        try {
          const { imageFileUri, includeDescription } = args;

          if (!imageFileUri) {
            return {
              content: [
                {
                  type: 'text',
                  text: 'Image file URI is required',
                },
              ],
            }
          }

          const { objects } = await this.api.detectHumanPoseKeypoints(imageFileUri, includeDescription);

          const categories: ResultCategory = {};
          for (const object of objects) {
            if (!categories[object.category]) {
              categories[object.category] = [];
            }
            categories[object.category].push(object);
          }

          const objectsInfo = objects.map(obj => {
            const bbox = parseBbox(obj.bbox);
            const pose = obj.pose ? parsePoseKeypoints(obj.pose) : undefined;

            return {
              name: obj.category,
              bbox,
              pose,
              ...(includeDescription ? {
                description: obj.caption,
              } : {}),
            }
          });

          return {
            content: [
              {
                type: "text",
                text: `${objectsInfo.length} human(s) detected in image.`
              },
              {
                type: "text",
                text: `Detailed human pose keypoints detection results: ${JSON.stringify((objectsInfo), null, 2)}`
              },
              {
                type: "text",
                text: `Note: The bbox coordinates are in {xmin, ymin, xmax, ymax} format, where the origin (0,0) is at the top-left corner of the image. The pose keypoints follow the same coordinate system, with visibility states (not visible, visible).`
              },
            ]
          };
        } catch (error) {
          return {
            content: [
              {
                type: 'text',
                text: `Failed to detect human pose keypoints from image: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
        }
      }
    )
  }

  async start(): Promise<void> {
    console.error("Starting DINO-X MCP server on Streamable HTTP transport...");

    const app = express();
    app.use(express.json());

    const authToken = this.config.authToken;

    const authMiddleware = (req: express.Request, res: express.Response, next: express.NextFunction) => {

      if (this.config.enableClientKey) {
        try {
          const apiKey = this.getApiKeyFromRequest(req);
          this.setApiClient(apiKey);
        } catch (apiKeyError) {
          console.error('API Key error:', apiKeyError);
          return res.status(401).json({
            jsonrpc: '2.0',
            error: {
              code: -32000,
              message: 'Unauthorized: Missing or invalid API key',
            },
            id: null,
          });
        }
      }

      // Handle auth token validation if configured
      if (!authToken) {
        return next();
      }

      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Unauthorized: Missing or invalid Authorization header",
          },
          id: null,
        });
      }

      const token = authHeader.substring(7);
      if (token !== authToken) {
        return res.status(401).json({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Unauthorized: Invalid token",
          },
          id: null,
        });
      }

      next();
    };

    app.all('/mcp', authMiddleware, async (req, res) => {
      try {
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: undefined
        })

        await this.server.connect(transport);

        await transport.handleRequest(req, res, req.body);
      } catch (error) {
        console.error('Error handling MCP request:', error);
        if (!res.headersSent) {
          res.status(500).json({
            jsonrpc: '2.0',
            error: {
              code: -32603,
              message: 'Internal server error',
            },
            id: null,
          });
        }
      }
    });

    app.listen(this.config.port, () => {
      console.error(`DINO-X MCP Server running on http://localhost:${this.config.port}`);
      if (this.config.enableClientKey) {
        console.error(`API key is required at client side. Please provide via URL parameter (http://localhost:${this.config.port}?key=your-dinox-api-key)`);
      }
    });
  }
}

