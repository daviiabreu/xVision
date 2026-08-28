import { apiBlob, apiFetch, apiUpload } from "./api";

export type VideoStatus = "Em processamento" | "Pendente" | "Falhou" | "Finalizado";

const TERMINAIS: ReadonlySet<VideoStatus> = new Set<VideoStatus>(["Finalizado", "Falhou"]);

export function terminou(status: VideoStatus | null): boolean {
  return status !== null && TERMINAIS.has(status);
}

export type MetricStatus = "ok" | "low_confidence" | "unavailable";

// `null` quando a feature nao produziu leitura: o tipo sai do valor, e sem valor nao ha tipo.
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
  frame: string | null;
  metricas: MetricaResponse[];
}

interface ProcessedVideoApiResponse {
  video_status: VideoStatus;
  resultados: ResultadoResponse[];
  motivo: string | null;
  render_url: string | null;
}

interface VideoStatusResponse {
  video_status: VideoStatus;
  motivo: string | null;
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
  frame: string | null;
  metrics: VideoMetric[];
}

export interface ProcessedVideo {
  videoStatus: VideoStatus;
  results: VideoResult[];
  reason: string | null;
  renderUrl: string | null;
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
    frame: resultado.frame,
    metrics: (resultado.metricas ?? []).map(toVideoMetric),
  };
}

export async function getProcessedVideo(id: number): Promise<ProcessedVideo> {
  const response = await apiFetch(`/v1/videos/${id}`);
  const body = (await response.json()) as ProcessedVideoApiResponse;

  return {
    videoStatus: body.video_status,
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

export async function enviarVideo(arquivo: File): Promise<number> {
  const reserva = await apiFetch("/v1/videos/upload-url", {
    method: "POST",
    body: JSON.stringify({ fileName: arquivo.name, contentType: arquivo.type }),
  });

  const { id, uploadUrl } = (await reserva.json()) as UploadUrlResponse;

  await apiUpload(uploadUrl, arquivo);

  return id;
}

export async function getVideoStatus(id: number): Promise<VideoStatusResponse> {
  const response = await apiFetch(`/v1/videos/status/${id}`);

  return (await response.json()) as VideoStatusResponse;
}
