import { createContext } from "react";

import type { VideoStatistics, VideoSummary } from "../services/videoService";

export interface VideoContextData {
  videos: VideoSummary[];
  total: number;
  statistics: VideoStatistics;
  loading: boolean;
  error: string | null;
  recarregar: () => Promise<void>;
}

export const VideoContext = createContext<VideoContextData | undefined>(undefined);
