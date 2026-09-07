import { apiBlob, apiFetch, apiUpload } from "./api";

export type VideoStatus = "Em processamento" | "Pendente" | "Falhou" | "Finalizado";

const TERMINAIS: ReadonlySet<VideoStatus> = new Set<VideoStatus>(["Finalizado", "Falhou"]);

export function terminou(status: VideoStatus | null): boolean {
  return status !== null && TERMINAIS.has(status);
}

export type MetricStatus = "ok" | "low_confidence" | "unavailable";

export type MetricType = "boolean" | "float" | "text" | null;

interface MetricaResponse {
  nome: string;
  unidade_de_medida: string;
  tipo: MetricType;
  valor: boolean | number | string | null;
  status: MetricStatus;
}

interface ResultadoResponse {
  nome_processo: string;
  status: boolean;
  situacao: "ok" | "failed" | "skipped";
  motivo: string | null;
  duracao_ms: number;
  frame: string | null;
  metricas: MetricaResponse[];
}

interface ProcessedVideoApiResponse {
  video_status: VideoStatus;
  nome_arquivo: string;
  content_type: string;
  criado_em: string;
  atualizado_em: string;
  resultados: ResultadoResponse[];
  motivo: string | null;
  render_url: string | null;
}

interface VideoProgressResponse {
  video_status: VideoStatus;
  motivo: string | null;
  etapa_atual: string | null;
  etapas_concluidas: number;
  etapas_total: number;
  progresso: number;
  posicao_fila: number | null;
  reconsultar_em_s: number | null;
}

interface VideoSummaryResponse extends VideoProgressResponse {
  id: number;
  nome_arquivo: string;
  content_type: string;
  criado_em: string;
  atualizado_em: string;
}

interface VideoListResponse {
  videos: VideoSummaryResponse[];
  total: number;
  pendentes: number;
  em_processamento: number;
  finalizadas: number;
  falhas: number;
}

export interface VideoMetric {
  name: string;
  unit: string;
  type: MetricType;
  value: boolean | number | string | null;
  status: MetricStatus;
}

export interface VideoResult {
  name: string;
  ok: boolean;
  state: "ok" | "failed" | "skipped";
  reason: string | null;
  durationMs: number;
  frame: string | null;
  metrics: VideoMetric[];
}

export interface ProcessedVideo {
  videoStatus: VideoStatus;
  name: string;
  contentType: string;
  createdAt: string;
  updatedAt: string;
  results: VideoResult[];
  reason: string | null;
  renderUrl: string | null;
}

export interface VideoProgress {
  videoStatus: VideoStatus;
  reason: string | null;
  currentStage: string | null;
  completedStages: number;
  totalStages: number;
  progress: number;
  queuePosition: number | null;
  retryAfterSeconds: number | null;
}

export interface VideoSummary extends VideoProgress {
  id: number;
  name: string;
  contentType: string;
  createdAt: string;
  updatedAt: string;
}

export interface VideoStatistics {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

function toVideoMetric(metrica: MetricaResponse): VideoMetric {
  return {
    name: metrica.nome,
    unit: metrica.unidade_de_medida,
    type: metrica.tipo,
    value: metrica.valor,
    status: metrica.status,
  };
}

function toVideoResult(resultado: ResultadoResponse): VideoResult {
  return {
    name: resultado.nome_processo,
    ok: resultado.status,
    state: resultado.situacao,
    reason: resultado.motivo,
    durationMs: resultado.duracao_ms,
    frame: resultado.frame,
    metrics: (resultado.metricas ?? []).map(toVideoMetric),
  };
}

export async function getProcessedVideo(id: number): Promise<ProcessedVideo> {
  const response = await apiFetch(`/v1/videos/${id}`);
  const body = (await response.json()) as ProcessedVideoApiResponse;

  return {
    videoStatus: body.video_status,
    name: body.nome_arquivo,
    contentType: body.content_type,
    createdAt: body.criado_em,
    updatedAt: body.atualizado_em,
    results: (body.resultados ?? []).map(toVideoResult),
    reason: body.motivo,
    renderUrl: body.render_url,
  };
}

export async function getRenderVideo(url: string): Promise<string | null> {
  const blob = await apiBlob(url);

  return blob === null ? null : URL.createObjectURL(blob);
}

interface UploadUrlResponse {
  id: number;
  uploadUrl: string;
}

export async function enviarVideo(
  arquivo: File,
  onProgress?: (percentage: number) => void,
): Promise<number> {
  const reserva = await apiFetch("/v1/videos/upload-url", {
    method: "POST",
    body: JSON.stringify({ fileName: arquivo.name, contentType: arquivo.type }),
  });

  const { id, uploadUrl } = (await reserva.json()) as UploadUrlResponse;

  await apiUpload(uploadUrl, arquivo, onProgress);

  return id;
}

function toVideoProgress(body: VideoProgressResponse): VideoProgress {
  return {
    videoStatus: body.video_status,
    reason: body.motivo,
    currentStage: body.etapa_atual,
    completedStages: body.etapas_concluidas,
    totalStages: body.etapas_total,
    progress: body.progresso,
    queuePosition: body.posicao_fila,
    retryAfterSeconds: body.reconsultar_em_s,
  };
}

function toVideoSummary(body: VideoSummaryResponse): VideoSummary {
  return {
    id: body.id,
    name: body.nome_arquivo,
    contentType: body.content_type,
    createdAt: body.criado_em,
    updatedAt: body.atualizado_em,
    ...toVideoProgress(body),
  };
}

export async function getVideos(): Promise<{
  videos: VideoSummary[];
  total: number;
  statistics: VideoStatistics;
}> {
  const response = await apiFetch("/v1/videos?limite=100&deslocamento=0");
  const body = (await response.json()) as VideoListResponse;

  return {
    videos: body.videos.map(toVideoSummary),
    total: body.total,
    statistics: {
      pending: body.pendentes,
      processing: body.em_processamento,
      completed: body.finalizadas,
      failed: body.falhas,
    },
  };
}

export async function getVideoStatus(id: number): Promise<VideoProgress> {
  const response = await apiFetch(`/v1/videos/status/${id}`);
  const body = (await response.json()) as VideoProgressResponse;

  return toVideoProgress(body);
}
