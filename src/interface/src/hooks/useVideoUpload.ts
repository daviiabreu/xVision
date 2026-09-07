import { useState } from "react";

import { enviarVideo } from "../services/videoService";
import { ERROR_MESSAGES } from "../utils/errorMessages";
import { validateVideoFile } from "../utils/fileValidation";

export function useVideoUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  function handleFileSelect(selectedFile: File) {
    const validationError = validateVideoFile(selectedFile);

    setError(validationError);
    setFile(validationError ? null : selectedFile);
  }

  async function confirmVideo(): Promise<number | null> {
    if (!file) {
      return null;
    }

    const enviado = file;

    try {
      setError(null);
      setIsUploading(true);
      setUploadProgress(0);

      const id = await enviarVideo(enviado, setUploadProgress);

      setFile((atual) => (atual === enviado ? null : atual));

      return id;
    } catch (erro) {
      setError(erro instanceof Error ? erro.message : ERROR_MESSAGES.VIDEO_UPLOAD_FAILED);

      return null;
    } finally {
      setIsUploading(false);
    }
  }

  return { file, error, isUploading, uploadProgress, handleFileSelect, confirmVideo };
}
