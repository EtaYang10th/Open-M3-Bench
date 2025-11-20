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
    description: `Analyze an image and detect all objects.

  Args:
    imageFileUri (string): Local image file path or file:// URI.
    includeDescription (boolean): Whether to include a text description per object.

  Returns:
    text (string): Summary of object categories, counts, and JSON details.`,
    },
    [Tool.DETECT_BY_TEXT]: {
      name: Tool.DETECT_BY_TEXT,
      description: `Detect objects in an image using a text prompt as categories.

  Args:
    imageFileUri (string): Local image file path or file:// URI.
    textPrompt (string): Period-separated English noun categories.
    includeDescription (boolean): Whether to include a text description per detected object.

  Returns:
    text (string): Summary of matched categories, counts, and JSON details.`,
    },
    [Tool.DETECT_HUMAN_POSE_KEYPOINTS]: {
      name: Tool.DETECT_HUMAN_POSE_KEYPOINTS,
      description: `Detect human pose keypoints and bounding boxes in an image.

  Args:
    imageFileUri (string): Local image file path or file:// URI.
    includeDescription (boolean): Whether to include a text description per person.

  Returns:
    text (string): Count of persons and JSON pose details per person.`,
    },
    [Tool.VISUALIZE_DETECTION_RESULT]: {
      name: Tool.VISUALIZE_DETECTION_RESULT,
      description: `Render detection results onto an image with boxes and labels.

  Args:
    imageFileUri (string): Local image file path or file:// URI.
    detections (array): Object detection results with names and bounding boxes.
    fontSize (number, optional): Label font size.
    boxThickness (number, optional): Bounding box line thickness.
    showLabels (boolean, optional): Whether to draw category labels.

  Returns:
    text (string): File path of the saved visualized image.`,
  },
}