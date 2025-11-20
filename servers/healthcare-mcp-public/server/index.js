#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { randomUUID } from 'crypto';

// Import our tools and services
import { CacheService } from './cache-service.js';
import { FDATool } from './fda-tool.js';
import { PubMedTool } from './pubmed-tool.js';
import { HealthTopicsTool } from './health-topics-tool.js';
import { ClinicalTrialsTool } from './clinical-trials-tool.js';
import { MedicalTerminologyTool } from './medical-terminology-tool.js';
import { MedRxivTool } from './medrxiv-tool.js';
import { MedicalCalculatorTool } from './medical-calculator-tool.js';
import { NcbiBookshelfTool } from './ncbi-bookshelf-tool.js';
import { DicomTool } from './dicom-tool.js';
import { UsageService } from './usage-service.js';

// Initialize services
const cacheService = new CacheService(parseInt(process.env.CACHE_TTL) || 86400);
const usageService = new UsageService();

// Initialize tools
const fdaTool = new FDATool(cacheService);
const pubmedTool = new PubMedTool(cacheService);
const healthTopicsTool = new HealthTopicsTool(cacheService);
const clinicalTrialsTool = new ClinicalTrialsTool(cacheService);
const medicalTerminologyTool = new MedicalTerminologyTool(cacheService);
const medrxivTool = new MedRxivTool(cacheService);
const medicalCalculatorTool = new MedicalCalculatorTool(cacheService);
const ncbiBookshelfTool = new NcbiBookshelfTool(cacheService);
const dicomTool = new DicomTool(cacheService);

// Generate a unique session ID for this connection
const sessionId = randomUUID();

// Create MCP server
const server = new Server(
  {
    name: "healthcare-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async (request) => {
  return {
    tools: [
      {
        name: "fda_drug_lookup",
        description: "  Look up FDA drug information by name.\n  Args:\n    drug_name (string): Drug name to search.\n    search_type (string): Information type: general, label, or adverse_events.\n  Returns:\n    result (object): JSON object with status, search parameters, and drug records.",
        inputSchema: {
          type: "object",
          properties: {
            drug_name: {
              type: "string",
              description: "Name of the drug to search for",
            },
            search_type: {
              type: "string",
              description: "Type of information to retrieve: 'label', 'adverse_events', or 'general'",
              enum: ["general", "label", "adverse_events"],
              default: "general",
            },
          },
          required: ["drug_name"],
        },
      },
      {
        name: "pubmed_search",
        description: "  Search medical literature in PubMed.\n  Args:\n    query (string): Search query text.\n    max_results (number): Maximum number of results to return.\n    date_range (string): Optional year-window filter.\n    open_access (boolean): Whether to filter for open access only.\n  Returns:\n    result (object): JSON object with status, query parameters, and article records.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query for medical literature",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return",
              default: 5,
              minimum: 1,
              maximum: 100,
            },
            date_range: {
              type: "string",
              description: "Limit to articles published within years (e.g. '5' for last 5 years)",
              default: "",
            },
            open_access: {
              type: "boolean",
              description: "Filter for open access articles",
              default: false,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "medrxiv_search",
        description: "  Search preprint articles on medRxiv.\n  Args:\n    query (string): Search query text.\n    max_results (number): Maximum number of results to return.\n  Returns:\n    result (object): JSON object with status, query parameters, and article records.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query for medRxiv articles",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return",
              default: 10,
              minimum: 1,
              maximum: 100,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "calculate_bmi",
        description: "  Calculate Body Mass Index (BMI).\n  Args:\n    height_meters (number): Height in meters.\n    weight_kg (number): Weight in kilograms.\n  Returns:\n    result (object): JSON object with status and BMI value.",
        inputSchema: {
          type: "object",
          properties: {
            height_meters: {
              type: "number",
              description: "Height in meters",
            },
            weight_kg: {
              type: "number",
              description: "Weight in kilograms",
            },
          },
          required: ["height_meters", "weight_kg"],
        },
      },
      {
        name: "ncbi_bookshelf_search",
        description: "  Search books and chapters in the NCBI Bookshelf.\n  Args:\n    query (string): Search query text.\n    max_results (number): Maximum number of results to return.\n  Returns:\n    result (object): JSON object with status, query parameters, and book records.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query for NCBI Bookshelf",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return",
              default: 10,
              minimum: 1,
              maximum: 100,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "extract_dicom_metadata",
        description: "  Extract key metadata from a local DICOM file.\n  Args:\n    file_path (string): Local filesystem path to the DICOM file.\n  Returns:\n    result (object): JSON object with status and selected DICOM metadata fields.",
        inputSchema: {
          type: "object",
          properties: {
            file_path: {
              type: "string",
              description: "Path to the DICOM file",
            },
          },
          required: ["file_path"],
        },
      },
      {
        name: "health_topics",
        description: "  Get evidence-based health topic summaries from Health.gov.\n  Args:\n    topic (string): Health topic search term.\n    language (string): Result language code, en or es.\n  Returns:\n    result (object): JSON object with status, topic, language, and topic entries.",
        inputSchema: {
          type: "object",
          properties: {
            topic: {
              type: "string",
              description: "Health topic to search for information",
            },
            language: {
              type: "string",
              description: "Language for content (en or es)",
              enum: ["en", "es"],
              default: "en",
            },
          },
          required: ["topic"],
        },
      },
      {
        name: "clinical_trials_search",
        description: "  Search clinical study records from ClinicalTrials.gov.\n  Args:\n    condition (string): Condition or disease name to search.\n    status (string): Trial status filter such as recruiting or completed.\n    max_results (number): Maximum number of trials to return.\n  Returns:\n    result (object): JSON object with status, query parameters, and trial records.",
        inputSchema: {
          type: "object",
          properties: {
            condition: {
              type: "string",
              description: "Medical condition or disease to search for",
            },
            status: {
              type: "string",
              description: "Trial status",
              enum: ["recruiting", "completed", "active", "not_recruiting", "all"],
              default: "recruiting",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return",
              default: 10,
              minimum: 1,
              maximum: 100,
            },
          },
          required: ["condition"],
        },
      },
      {
        name: "lookup_icd_code",
        description: "  Look up ICD-10 diagnosis codes and descriptions.\n  Args:\n    code (string): ICD-10 code to look up (optional if description is provided).\n    description (string): Condition description to search (optional if code is provided).\n    max_results (number): Maximum number of code records to return.\n  Returns:\n    result (object): JSON object with status, search parameters, and code records.",
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "ICD-10 code to look up (optional if description is provided)",
            },
            description: {
              type: "string",
              description: "Medical condition description to search for (optional if code is provided)",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return",
              default: 10,
              minimum: 1,
              maximum: 50,
            },
          },
        },
      },
      {
        name: "get_usage_stats",
        description: "  Get usage statistics for the current MCP session.\n  Args:\n    (none): This tool takes no input arguments.\n  Returns:\n    result (object): JSON object with session ID, start time, and tool call counts.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_all_usage_stats",
        description: "  Get aggregated usage statistics across all sessions.\n  Args:\n    (none): This tool takes no input arguments.\n  Returns:\n    result (object): JSON object with overall usage statistics by session and tool.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// Handle tool execution
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    // Record usage
    usageService.recordUsage(sessionId, name);

    let result;

    switch (name) {
      case "fda_drug_lookup":
        result = await fdaTool.lookupDrug(args.drug_name, args.search_type);
        break;

      case "pubmed_search":
        result = await pubmedTool.searchLiterature(args.query, args.max_results, args.date_range, args.open_access);
        break;

      case "medrxiv_search":
        result = await medrxivTool.search(args.query, args.max_results);
        break;

      case "calculate_bmi":
        result = medicalCalculatorTool.calculateBmi(args.height_meters, args.weight_kg);
        break;

      case "ncbi_bookshelf_search":
        result = await ncbiBookshelfTool.search(args.query, args.max_results);
        break;

      case "extract_dicom_metadata":
        result = dicomTool.extractMetadata(args.file_path);
        break;

      case "health_topics":
        result = await healthTopicsTool.getHealthTopics(args.topic, args.language);
        break;

      case "clinical_trials_search":
        result = await clinicalTrialsTool.searchTrials(args.condition, args.status, args.max_results);
        break;

      case "lookup_icd_code":
        result = await medicalTerminologyTool.lookupICDCode(args.code, args.description, args.max_results);
        break;

      case "get_usage_stats":
        result = usageService.getSessionUsage(sessionId);
        break;

      case "get_all_usage_stats":
        result = usageService.getAllUsageStats();
        break;

      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    console.error(`Error executing tool ${name}:`, error);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "error",
            error_message: error.message,
          }, null, 2),
        },
      ],
      isError: true,
    };
  }
});

// Start the server
const transport = new StdioServerTransport();
server.connect(transport);

console.error("Healthcare MCP Node.js server running...");
console.error(`Session ID: ${sessionId}`);
console.error(`Cache TTL: ${cacheService.cache.options.stdTTL} seconds`);
