import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

import { VERSION } from './constants.js';
import { 
  FindParksSchema, 
  GetParkDetailsSchema, 
  GetAlertsSchema,
  GetVisitorCentersSchema,
  GetCampgroundsSchema,
  GetEventsSchema
} from './schemas.js';
import { findParksHandler } from './handlers/findParks.js';
import { getParkDetailsHandler } from './handlers/getParkDetails.js';
import { getAlertsHandler } from './handlers/getAlerts.js';
import { getVisitorCentersHandler } from './handlers/getVisitorCenters.js';
import { getCampgroundsHandler } from './handlers/getCampgrounds.js';
import { getEventsHandler } from './handlers/getEvents.js';

// Create and configure the server
export function createServer() {
  const server = new Server(
    {
      name: "nationalparks-mcp-server",
      version: VERSION,
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register tool definitions
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: "findParks",
          description: "  Search for national parks using location or filter criteria.\n  Args:\n    (see schema): Use FindParksSchema fields such as state, name, and activities.\n  Returns:\n    result (text): JSON string with matching park records.",
          inputSchema: zodToJsonSchema(FindParksSchema),
        },
        {
          name: "getParkDetails",
          description: "  Get detailed information for a specific national park.\n  Args:\n    (see schema): Use GetParkDetailsSchema fields such as park code.\n  Returns:\n    result (text): JSON string describing the selected park.",
          inputSchema: zodToJsonSchema(GetParkDetailsSchema),
        },
        {
          name: "getAlerts",
          description: "  Get current alerts for national parks.\n  Args:\n    (see schema): Use GetAlertsSchema fields such as park codes or state.\n  Returns:\n    result (text): JSON string listing active alerts.",
          inputSchema: zodToJsonSchema(GetAlertsSchema),
        },
        {
          name: "getVisitorCenters",
          description: "  Get information about visitor centers and operating hours.\n  Args:\n    (see schema): Use GetVisitorCentersSchema fields such as park code or state.\n  Returns:\n    result (text): JSON string listing visitor centers.",
          inputSchema: zodToJsonSchema(GetVisitorCentersSchema),
        },
        {
          name: "getCampgrounds",
          description: "  Get information about available campgrounds and amenities.\n  Args:\n    (see schema): Use GetCampgroundsSchema fields such as park code or state.\n  Returns:\n    result (text): JSON string listing campgrounds.",
          inputSchema: zodToJsonSchema(GetCampgroundsSchema),
        },
        {
          name: "getEvents",
          description: "  Find upcoming events at national parks.\n  Args:\n    (see schema): Use GetEventsSchema fields such as park code or date filters.\n  Returns:\n    result (text): JSON string listing events.",
          inputSchema: zodToJsonSchema(GetEventsSchema),
        },
      ],
    };
  });

  // Handle tool executions
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
      if (!request.params.arguments) {
        throw new Error("Arguments are required");
      }

      switch (request.params.name) {
        case "findParks": {
          const args = FindParksSchema.parse(request.params.arguments);
          return await findParksHandler(args);
        }

        case "getParkDetails": {
          const args = GetParkDetailsSchema.parse(request.params.arguments);
          return await getParkDetailsHandler(args);
        }

        case "getAlerts": {
          const args = GetAlertsSchema.parse(request.params.arguments);
          return await getAlertsHandler(args);
        }

        case "getVisitorCenters": {
          const args = GetVisitorCentersSchema.parse(request.params.arguments);
          return await getVisitorCentersHandler(args);
        }

        case "getCampgrounds": {
          const args = GetCampgroundsSchema.parse(request.params.arguments);
          return await getCampgroundsHandler(args);
        }

        case "getEvents": {
          const args = GetEventsSchema.parse(request.params.arguments);
          return await getEventsHandler(args);
        }

        default:
          throw new Error(`Unknown tool: ${request.params.name}`);
      }
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          content: [{ 
            type: "text", 
            text: JSON.stringify({
              error: 'Validation error',
              details: error.errors
            }, null, 2)
          }]
        };
      }
      
      console.error('Error executing tool:', error);
      return {
        content: [{ 
          type: "text", 
          text: JSON.stringify({
            error: 'Server error',
            message: error instanceof Error ? error.message : 'Unknown error'
          }, null, 2)
        }]
      };
    }
  });

  return server;
} 