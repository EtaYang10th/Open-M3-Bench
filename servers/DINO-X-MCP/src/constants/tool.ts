export const BASE_URL = 'https://api.deepdataspace.com/v2/';

export const DEFAULT_TIMEOUT = 60000;

export enum Tool {
  DETECT_BY_TEXT = "detect-objects-by-text",
  DETECT_ALL_OBJECTS = "detect-all-objects",
  DETECT_HUMAN_POSE_KEYPOINTS = "detect-human-pose-keypoints",
  VISUALIZE_DETECTION_RESULT = "visualize-detection-result",
}

export const ToolConfigs: Record<Tool, {
  name: string;
  description: string;
}> = {
  [Tool.DETECT_ALL_OBJECTS]: {
    name: Tool.DETECT_ALL_OBJECTS,
    description: `Analyze an image to detect all identifiable objects.

  Args:
    imageFileUri (string): Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted.
    includeDescription (boolean): Whether to include a natural-language description per object.

  Returns:
    text: Plain-text summary of categories and counts, plus JSON of per-object details with { name, bbox: { xmin, ymin, xmax, ymax }, description? }.`,
    },
    [Tool.DETECT_BY_TEXT]: {
      name: Tool.DETECT_BY_TEXT,
      description: `Analyze an image using a text prompt to find specific objects.

  Args:
    imageFileUri (string): Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted.
    textPrompt (string): Noun categories (English), separated by periods, e.g. 'person.car.traffic light'.
    includeDescription (boolean): Whether to include a natural-language description per detected object.

  Returns:
    text: Plain-text summary of matched categories and counts, plus JSON of per-object details with { name, bbox: { xmin, ymin, xmax, ymax }, description? }.`,
    },
    [Tool.DETECT_HUMAN_POSE_KEYPOINTS]: {
      name: Tool.DETECT_HUMAN_POSE_KEYPOINTS,
      description: `Detect 17 human pose keypoints per person and associated bounding boxes.

  Args:
    imageFileUri (string): Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted.
    includeDescription (boolean): Whether to include a natural-language description per person.

  Returns:
    text: Count of detected persons and JSON per person with { name, bbox: { xmin, ymin, xmax, ymax }, pose: [ { x, y, v } × 17 ], description? }.`,
    },
    [Tool.VISUALIZE_DETECTION_RESULT]: {
      name: Tool.VISUALIZE_DETECTION_RESULT,
      description: `Visualize detection results by drawing boxes and labels on the image.

  Args:
    imageFileUri (string): Local image file path. You may pass a plain filesystem path or a file:// URI. Remote URLs are not accepted.
    detections (array): Array of { name: string, bbox: { xmin, ymin, xmax, ymax } }.
    fontSize (number, optional): Font size for labels (default: 24).
    boxThickness (number, optional): Box line thickness (default: 4).
    showLabels (boolean, optional): Whether to show category labels (default: true).

  Returns:
    text: File path where the visualized image is saved (respects IMAGE_STORAGE_DIRECTORY).`,
  },
}