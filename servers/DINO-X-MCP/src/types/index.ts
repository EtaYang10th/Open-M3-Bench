
export type TaskStatus = "waiting" | "running" | "success" | "failed";

export interface TaskResponse<T> {
  code: number;
  data: {
    error: string;
    result?: T;
    status: TaskStatus;
    uuid: string;
  };
  msg: string;
}

export namespace APIResponse {
  export interface DINOX {
    objects: {
      category: string;
      score: number;
      bbox: [number, number, number, number];
      pose?: number[];
    }[]
  }

  export interface DINOXRegionVL {
    objects: {
      region: [number, number, number, number];
      caption: string;
      roc: string;
      ocr: string;
    }[];
  }
}

export interface ResultCategory {
  [key: string]: {
    category: string;
    score: number;
    bbox: [number, number, number, number];
    pose?: number[];
  }[]
}

export enum Transport {
  STDIO = "stdio",
  StreamableHTTP = "http",
}

export const DEFAULT_TRANSPORT = Transport.STDIO;

export const DEFAULT_PORT = 3020;


export interface Config {
  apiKey?: string;                   // optional for HTTP transport (can be passed via URL parameter)
  transport: Transport;
  imageStorageDirectory?: string;    // only used for stdio transport
  port?: number;                     // only used for streamable http transport
  authToken?: string;                // only used for streamable http transport
  enableClientKey?: boolean;         // whether to allow API key via URL parameter (streamable http transport only)
}
