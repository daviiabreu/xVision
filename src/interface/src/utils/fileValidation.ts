import { ERROR_MESSAGES } from "./errorMessages";

const MAX_VIDEO_SIZE = 128 * 1024 * 1024;

export function validateVideoFile(file: File): string | null {
  if (!file.type.startsWith("video/")) {
    return ERROR_MESSAGES.VIDEO_INVALID;
  }

  if (file.size === 0) {
    return ERROR_MESSAGES.VIDEO_EMPTY;
  }

  if (file.size > MAX_VIDEO_SIZE) {
    return ERROR_MESSAGES.VIDEO_TOO_LARGE;
  }

  return null;
}
