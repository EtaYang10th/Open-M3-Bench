declare module '@modelcontextprotocol/sdk' {
  export class Server {
    constructor(info: { name: string; version: string }, options: { capabilities: { tools: any } });
    setRequestHandler(schema: any, handler: Function): void;
    connect(transport: any): Promise<void>;
  }

  export class StdioServerTransport {
    constructor();
  }

  export const CallToolRequestSchema: unique symbol;
  export const ListToolsRequestSchema: unique symbol;

  export interface CallToolRequest {
    params: {
      name: string;
      arguments?: Record<string, unknown>;
    };
  }
}
