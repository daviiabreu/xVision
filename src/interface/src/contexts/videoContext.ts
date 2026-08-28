import { createContext } from "react";

import type { VideoStatus } from "../services/videoService";
import type { CachedVideo } from "../services/videoCache";

export interface Video extends CachedVideo {
  status: VideoStatus | null;
  statusError: boolean;
}

export interface VideoContextData {
  videos: Video[];
  recarregar: () => Promise<void>;
  remover: (id: number) => void;
}

export const VideoContext = createContext<VideoContextData | undefined>(undefined);
