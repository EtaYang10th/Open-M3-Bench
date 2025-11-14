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
        description: "Lookup FDA drug information.\n  \n  Args:\n  - drug_name (string): Required.\n  - search_type (string): One of 'general'|'label'|'adverse_events'. Default 'general'.\n  \n  Returns:\n  - { status, drug_name, search_type, total_results, drugs[] }\n  - drugs[] items: { product_number, generic_name, brand_name, labeler_name, product_type, route[], marketing_status }\n  - For 'label'/'adverse_events': may include { indications[], dosage[], warnings[], contraindications[], adverse_reactions[], drug_interactions[], pregnancy[], boxed_warning[] }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Search PubMed literature.\n  \n  Args:\n  - query (string): Required.\n  - max_results (number): 1–100, default 5.\n  - date_range (string): Years window (e.g., '5').\n  - open_access (boolean): Filter open access, default false.\n  \n  Returns:\n  - { status, query, total_results, date_range, open_access, articles[] }\n  - articles[] items: { id, title, authors[], journal, publication_date, abstract_url, doi? }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Search medRxiv preprints.\n  \n  Args:\n  - query (string): Required.\n  - max_results (number): 1–100, default 10.\n  \n  Returns:\n  - { status, query, total_results, articles[] }\n  - articles[] items: { title, authors, doi, abstract_url, publication_date }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Calculate Body Mass Index (BMI).\n  \n  Args:\n  - height_meters (number): Required.\n  - weight_kg (number): Required.\n  \n  Returns:\n  - { status, bmi } where bmi is a string rounded to 2 decimals.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Search the NCBI Bookshelf.\n  \n  Args:\n  - query (string): Required.\n  - max_results (number): 1–100, default 10.\n  \n  Returns:\n  - { status, query, total_results, books[] }\n  - books[] items: { id, title, authors[], publication_date, url }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Extract metadata from a DICOM file.\n  \n  Args:\n  - file_path (string): Required local path to a DICOM file.\n  \n  Returns:\n  - { status, patientName, patientId, studyDescription, seriesDescription }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Get evidence-based health topics from Health.gov.\n  \n  Args:\n  - topic (string): Required.\n  - language (string): 'en'|'es', default 'en'.\n  \n  Returns:\n  - { status, search_term, language, total_results, health_topics[] }\n  - health_topics[] items: { title, url, last_updated, section, description, content[] }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Search ClinicalTrials.gov studies.\n  \n  Args:\n  - condition (string): Required medical condition/disease.\n  - status (string): 'recruiting'|'completed'|'active'|'not_recruiting'|'all', default 'recruiting'.\n  - max_results (number): 1–100, default 10.\n  \n  Returns:\n  - { status, condition, search_status, total_results, trials[] }\n  - trials[] items: { nct_id, title, status, phase[], study_type, conditions[], locations[] { facility, city, state, country }, sponsor, url, eligibility { gender, min_age, max_age, healthy_volunteers } }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Look up ICD-10 codes and descriptions.\n  \n  Args:\n  - code (string): Optional if description provided.\n  - description (string): Optional if code provided.\n  - max_results (number): 1–50, default 10.\n  \n  Returns:\n  - { status, search_type, search_term, total_results, codes[] }\n  - codes[] items: { code, description, category }.\n  \n  Errors:\n  - { status: 'error', error_message }",
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
        description: "Get usage for the current session.\n  \n  Returns:\n  - { status, session_id, session_start, total_calls, tool_usage { [toolName]: count } }",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_all_usage_stats",
        description: "Get overall usage across sessions.\n  \n  Returns:\n  - { status, overall_stats { session_start, total_calls, tool_usage { [toolName]: count } } }",
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
