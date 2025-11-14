import sharp from 'sharp';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';

export const parseArgs = () => {
  const args: Record<string, string> = {};
  const argv = process.argv.slice(2);
  
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    
    if (arg.startsWith("--")) {
      const [key, value] = arg.slice(2).split("=");
      
      if (value !== undefined) {
        // Format: --key=value
        args[key] = value;
      } else {
        // Format: --key or --key value
        const nextArg = argv[i + 1];
        if (nextArg && !nextArg.startsWith("--")) {
          // Format: --key value
          args[key] = nextArg;
          i++; // Skip the next argument as it's the value
        } else {
          // Format: --key (boolean flag)
          args[key] = "";
        }
      }
    }
  }
  
  return args;
}

/**
 * Parse pose keypoints array into a structured object
 * @param pose An array of numbers representing human pose keypoints
 * Format: [x, y, visible, score, x2, y2, visible2, score2, ...]
 * Each keypoint is represented by 4 values:
 * - x: x-coordinate (0 to image width)
 * - y: y-coordinate (0 to image height)
 * - visible: visibility state (0: not visible, 1: visible)
 * - score: confidence score (0-1)
 * 
 * Keypoint order:
 * [nose, l-eye, r-eye, l-ear, r-ear, l-shoulder, r-shoulder, 
 *  l-elbow, r-elbow, l-wrist, r-wrist, l-hip, r-hip, 
 *  l-knee, r-knee, l-ankle, r-ankle]
 * 
 * @returns A structured object with keypoint information
 */
export const parsePoseKeypoints = (pose: number[]) => {
  const keypointNames = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
    'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
    'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
    'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
  ];

  const structuredPose: { [key: string]: { x: number; y: number; visible: string; } } = {};

  for (let i = 0; i < keypointNames.length; i++) {
    const baseIndex = i * 4;
    const visibilityMap = {
      0: 'not visible',
      2: 'visible'
    };

    structuredPose[keypointNames[i]] = {
      x: parseFloat(pose[baseIndex].toFixed(1)),
      y: parseFloat(pose[baseIndex + 1].toFixed(1)),
      visible: visibilityMap[pose[baseIndex + 2] as keyof typeof visibilityMap],
    };
  }

  return structuredPose;
};

export const parseBbox = (bbox: number[]) => {
  return {
    xmin: parseFloat(bbox[0].toFixed(1)),
    ymin: parseFloat(bbox[1].toFixed(1)),
    xmax: parseFloat(bbox[2].toFixed(1)),
    ymax: parseFloat(bbox[3].toFixed(1))
  };
};
export interface DetectionResult {
  name: string;
  bbox: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
  description?: string;
}

export interface VisualizationOptions {
  fontSize?: number;
  boxThickness?: number;
  colors?: string[];
  showLabels?: boolean;
  showConfidence?: boolean;
}

/** Saturation-0.9, Brightness-0.85, Hue Interval-137.5 */
const DEFAULT_COLORS = ['#D91616', '#16D94F', '#8716D9', '#D9C016', '#16B8D9', '#D9167F', '#46D916', '#1E16D9', '#D95716', '#16D990', '#C816D9', '#B0D916', '#1677D9', '#D9163E', '#16D926', '#5F16D9', '#D99816', '#16D9D1', '#D916A8', '#6FD916'];

/**
 * Get the appropriate font family for the current platform (for SVG)
 * @returns font family string for SVG
 */
function getPlatformFont(): string {
  const platform = os.platform();

  let fontFamily: string;
  switch (platform) {
    case 'win32':
      fontFamily = '"Microsoft YaHei", "SimHei", "Segoe UI", "Arial Unicode MS", "Arial", "Verdana", "Tahoma", sans-serif';
      break;
    case 'darwin':
      fontFamily = '"PingFang SC", "Helvetica Neue", "Arial Unicode MS", "San Francisco", ".AppleSystemUIFont", "Arial", "Times", sans-serif';
      break;
    case 'linux':
      fontFamily = '"Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans", "Liberation Sans", "Arial Unicode MS", "Arial", "Helvetica", sans-serif';
      break;
    default:
      fontFamily = '"Arial Unicode MS", "Arial", "Helvetica", sans-serif';
  }

  return fontFamily;
}

/**
 * Estimate text width for SVG
 * @param text text to measure
 * @param fontSize font size
 * @returns estimated text width
 */
function estimateTextWidth(text: string, fontSize: number): number {
  return text.length * fontSize * 0.6;
}

/**
 * Draw detection results on the image using Sharp + SVG
 * @param imageUri image file path or URI
 * @param detections detection results array
 * @param options visualization options
 * @returns the path of the visualized image
 */
export async function visualizeDetections(
  imageUri: string,
  detections: DetectionResult[],
  options: VisualizationOptions = {},
  storageDir: string
): Promise<string> {

  let imageBuffer: Buffer;
  let imagePath = '';
  
  // Accept local filesystem paths and file:// URIs only
  if (imageUri.startsWith('http://') || imageUri.startsWith('https://')) {
    throw new Error('Only local image files are supported. Provide a filesystem path, not a URL.');
  }

  imagePath = imageUri.startsWith('file://') ? decodeURIComponent(fileURLToPath(imageUri)) : imageUri;
  if (!fs.existsSync(imagePath)) {
    throw new Error('Image file not found: ' + imagePath);
  }
  imageBuffer = fs.readFileSync(imagePath);

  // Get image metadata
  const imageInfo = await sharp(imageBuffer).metadata();
  const { width = 0, height = 0 } = imageInfo;

  const {
    fontSize = 24,
    boxThickness = 4,
    colors = DEFAULT_COLORS,
    showLabels = true,
  } = options;

  const fontFamily = getPlatformFont();

  // Create category to color mapping
  const categoryColorMap = new Map<string, string>();
  let colorIndex = 0;

  // Build SVG overlay with all annotations
  const svgElements: string[] = [];

  for (const detection of detections) {
    if (!categoryColorMap.has(detection.name)) {
      categoryColorMap.set(detection.name, colors[colorIndex % colors.length]);
      colorIndex++;
    }

    const color = categoryColorMap.get(detection.name)!;
    const { bbox } = detection;

    const rectWidth = bbox.xmax - bbox.xmin;
    const rectHeight = bbox.ymax - bbox.ymin;

    svgElements.push(`<rect x="${bbox.xmin}" y="${bbox.ymin}" width="${rectWidth}" height="${rectHeight}" fill="none" stroke="${color}" stroke-width="${boxThickness}" shape-rendering="crispEdges"/>`);

    if (showLabels) {
      const label = detection.name;

      const textWidth = estimateTextWidth(label, fontSize);
      const textHeight = fontSize;
      const padding = 4;

      let labelY = bbox.ymin - textHeight - padding * 2;
      if (labelY < 0) {
        labelY = bbox.ymin;
      }

      const bgX = bbox.xmin;
      const bgY = labelY;
      const bgWidth = textWidth + padding * 2;
      const bgHeight = textHeight + padding * 2;

      // Draw label background
      svgElements.push(`<rect x="${bgX}" y="${bgY}" width="${bgWidth}" height="${bgHeight}" fill="${color}"/>`);

      // Draw label text
      const textX = bgX + padding;
      const textY = bgY + padding + textHeight * 0.8;
      
      // Escape special characters in label text for XML
      const escapedLabel = label.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      
      svgElements.push(`<text x="${textX}" y="${textY}" fill="white" font-size="${fontSize}" font-family='${fontFamily}' text-rendering="optimizeLegibility" dominant-baseline="alphabetic">${escapedLabel}</text>`);
    }
  }

  // Generate complete SVG overlay
  const svgOverlay = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <style>
        text {
          font-weight: normal;
          text-rendering: optimizeLegibility;
        }
        rect {
          shape-rendering: crispEdges;
        }
      </style>
    </defs>
    ${svgElements.join('\n    ')}
  </svg>`;
  
  // Generate output path
  const outputPath = generateOutputPath(imagePath, storageDir);

  // Composite SVG overlay onto image using Sharp
  await sharp(imageBuffer)
    .composite([
      { input: Buffer.from(svgOverlay), top: 0, left: 0 }
    ])
    .png({ quality: 100, compressionLevel: 6 })
    .toFile(outputPath);

  return outputPath;
}

/**
 * Get the default image storage directory based on the operating system
 * @returns default directory path
 */
export function getDefaultImageStorageDirectory(): string {
  const platform = os.platform();

  switch (platform) {
    case 'win32':
      return path.join(os.tmpdir(), 'dinox-mcp');
    case 'darwin':
    case 'linux':
      return path.join(os.tmpdir(), 'dinox-mcp');
    default:
      return path.join(os.tmpdir(), 'dinox-mcp');
  }
}

/**
 * Ensure the directory exists, create if it doesn't
 * @param dirPath directory path to ensure
 */
function ensureDirectoryExists(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

/**
 * Generate output file path
 * @param originalPath original image path
 * @returns output file path
 */
function generateOutputPath(originalPath: string, storageDir: string): string {
  ensureDirectoryExists(storageDir);

  const ext = path.extname(originalPath);
  const basename = path.basename(originalPath, ext);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

  return path.join(storageDir, `${basename}_visualized_${timestamp}.png`);
}