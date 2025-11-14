import axios, { AxiosInstance } from "axios";
import fs from "fs";
import { fileURLToPath } from 'url';
import { APIResponse, TaskResponse } from "../types/index.js";

export interface DinoXApiConfig {
  apiKey: string;
  baseUrl: string;
  timeout: number;
}

interface DetectionPayload {
  model: string;
  image: string;
  prompt: {
    type: string;
    text?: string;
    universal?: number;
  };
  targets: string[];
  bbox_threshold: number;
  iou_threshold: number;
}

interface DetectionResult {
  objects: {
    category: string;
    score: number;
    bbox: [number, number, number, number];
    pose?: number[];
    caption?: string;
  }[];
}

export class DinoXApiClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly axiosClient: AxiosInstance;

  constructor(config: DinoXApiConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl;
    this.axiosClient = axios.create({
      baseURL: this.baseUrl,
      timeout: config.timeout,
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      headers: {
        Token: `${this.apiKey}`,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Client-Source": "mcp"
      },
    });
  }

  async createTask<T>(endpoint: string, payload: any): Promise<T> {
    if (!this.apiKey) {
      throw new Error("API Key is missing");
    }

    try {
      const response = await this.axiosClient.post(
        `task/${endpoint}`,
        payload
      );

      if (response.data.code !== 0) {
        throw new Error(`Failed to create task: ${response.data.msg}`);
      }

      const taskUuid = response.data.data.task_uuid;

      const maxRetries = 60;
      const retryInterval = 1000;
      let retryCount = 0;

      while (retryCount < maxRetries) {
        const statusResponse = await this.axiosClient.get<TaskResponse<T>>(
          `task_status/${taskUuid}`,
        );

        const status = statusResponse.data;

        if (status.data.status === "failed") {
          throw new Error(status.data.error || "Task failed");
        }

        if (status.data.status === "success" && status.data.result) {
          return status.data.result;
        }

        if (status.data.status === "waiting" || status.data.status === "running") {
          await new Promise(resolve => setTimeout(resolve, retryInterval));
          retryCount++;
          continue;
        }

        throw new Error("Task polling timeout");
      }

      throw new Error("Task polling timeout");
    } catch (error) {
      console.error("API request error:", error);
      if (axios.isAxiosError(error) && error.response) {
        const data = error.response.data;
        if (error.response.status === 400) {
          throw new Error(`Invalid parameters: ${data.errors?.join(", ")}`);
        }
        throw new Error(
          `API error (${error.response.status}): ${JSON.stringify(data)}`
        );
      }
      throw error;
    }
  }

  private async encodeLocalImageToDataUri(inputPath: string): Promise<string> {
    // Only support local filesystem paths; reject remote URLs
    if (inputPath.startsWith("http://") || inputPath.startsWith("https://")) {
      throw new Error("Only local image files are supported. Provide a filesystem path, not a URL.");
    }

    // Support both plain paths and file:// URIs
    const filePath = inputPath.startsWith("file://") ? fileURLToPath(inputPath) : inputPath;
    const imageBuffer = await fs.promises.readFile(filePath);
    const base64Image = imageBuffer.toString('base64');
    // Keep PNG data URI header for compatibility
    return `data:image/png;base64,${base64Image}`;
  }

  private async addDescriptions(
    imageDataUri: string,
    detectionResponse: APIResponse.DINOX
  ): Promise<DetectionResult> {
    if (detectionResponse.objects.length === 0) {
      return detectionResponse;
    }

    const regions = detectionResponse.objects.map(obj => obj.bbox);
    const captionApiPayload = {
      model: "DINO-X-1.0",
      image: imageDataUri,
      regions,
      targets: ["caption"],
    };

    const captionRsp = await this.createTask<APIResponse.DINOXRegionVL>("dinox/region_vl", captionApiPayload);

    const objects = detectionResponse.objects.map((obj, index) => {
      const captionObj = captionRsp.objects[index];
      return {
        ...obj,
        ...captionObj,
      };
    });

    return { objects };
  }

  private async performDetection(
    imageFileUri: string,
    includeDescription: boolean,
    detectionPayload: Omit<DetectionPayload, 'image'>
  ): Promise<DetectionResult> {
    try {
      const imageDataUri = await this.encodeLocalImageToDataUri(imageFileUri);
      
      const fullPayload = {
        ...detectionPayload,
        image: imageDataUri,
      };

      const detectionRsp = await this.createTask<APIResponse.DINOX>("dinox/detection", fullPayload);

      if (!includeDescription) {
        return detectionRsp;
      }

      return await this.addDescriptions(imageDataUri, detectionRsp);
    } catch (error) {
      console.error("API request error:", error);
      throw error;
    }
  }

  async detectObjectsByText(
    imageFileUri: string,
    textPrompt: string,
    includeDescription: boolean
  ): Promise<DetectionResult> {
    return this.performDetection(imageFileUri, includeDescription, {
      model: "DINO-X-1.0",
      prompt: {
        type: "text",
        text: textPrompt
      },
      targets: ["bbox"],
      bbox_threshold: 0.25,
      iou_threshold: 0.8
    });
  }

  async detectAllObjects(
    imageFileUri: string,
    includeDescription: boolean
  ): Promise<DetectionResult> {
    return this.performDetection(imageFileUri, includeDescription, {
      model: "DINO-X-1.0",
      prompt: {
        type: "universal",
        universal: 1
      },
      targets: ["bbox"],
      bbox_threshold: 0.25,
      iou_threshold: 0.8
    });
  }

  async detectHumanPoseKeypoints(
    imageFileUri: string,
    includeDescription: boolean
  ): Promise<DetectionResult> {
    return this.performDetection(imageFileUri, includeDescription, {
      model: "DINO-X-1.0",
      prompt: {
        type: "text",
        text: "person"
      },
      targets: ["bbox", "pose_keypoints"],
      bbox_threshold: 0.25,
      iou_threshold: 0.8
    });
  }
}