import type {
  ProcessedVideo,
  VideoProgress,
  VideoResult,
  VideoSummary,
} from "../services/videoService";

export const PROGRESSO_FINALIZADO: VideoProgress = {
  videoStatus: "Finalizado",
  reason: null,
  currentStage: null,
  completedStages: 9,
  totalStages: 9,
  progress: 100,
  queuePosition: null,
  retryAfterSeconds: null,
};

export const RESULTADO_PRINCIPAL: VideoResult = {
  name: "shot",
  ok: true,
  state: "ok",
  reason: null,
  durationMs: 45,
  frame: null,
  metrics: [],
};

export const VIDEO_PRONTO: ProcessedVideo = {
  videoStatus: "Finalizado",
  name: "treino.mp4",
  contentType: "video/mp4",
  createdAt: "2026-09-01T12:00:00Z",
  updatedAt: "2026-09-01T12:01:00Z",
  results: [RESULTADO_PRINCIPAL],
  reason: null,
  renderUrl: null,
};

export function resumo(parcial: Partial<VideoSummary> = {}): VideoSummary {
  return {
    id: 3,
    name: "treino.mp4",
    contentType: "video/mp4",
    createdAt: "2026-09-01T12:00:00Z",
    updatedAt: "2026-09-01T12:01:00Z",
    ...PROGRESSO_FINALIZADO,
    ...parcial,
  };
}
