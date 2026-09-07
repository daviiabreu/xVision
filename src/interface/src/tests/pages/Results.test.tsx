import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Results from "../../pages/Results/Results";
import { ErroHttp } from "../../services/api";
import {
  getProcessedVideo,
  getRenderVideo,
  getVideoStatus,
  type VideoResult,
} from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";
import { PROGRESSO_FINALIZADO, RESULTADO_PRINCIPAL, VIDEO_PRONTO } from "../fixtures";

vi.mock("../../services/videoService", () => ({
  getProcessedVideo: vi.fn(),
  getRenderVideo: vi.fn(),
  getVideoStatus: vi.fn(),
}));

function etapa(parcial: Partial<VideoResult> = {}): VideoResult {
  return { ...RESULTADO_PRINCIPAL, name: "ball", ...parcial };
}

function montar(entrada = "/results/1") {
  render(
    <MemoryRouter initialEntries={[entrada]}>
      <Link to="/results/4">ir para o 4</Link>
      <Routes>
        <Route path="/results/:id" element={<Results />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.mocked(getProcessedVideo).mockReset();
  vi.mocked(getRenderVideo).mockReset();
  vi.mocked(getVideoStatus).mockReset();
  vi.mocked(getVideoStatus).mockResolvedValue({
    ...PROGRESSO_FINALIZADO,
    videoStatus: "Em processamento",
    currentStage: "ball",
    completedStages: 3,
    progress: 33,
    retryAfterSeconds: 5,
  });
});

afterEach(() => vi.useRealTimers());

describe("Results", () => {
  it("mostra arquivo, resumo e metricas quando terminou", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue(VIDEO_PRONTO);
    montar();

    expect(await screen.findByRole("heading", { name: "treino.mp4" })).toBeInTheDocument();
    expect(screen.getByText("1 de 1")).toBeInTheDocument();
    expect(screen.getByText("Métricas da finalização")).toBeInTheDocument();
  });

  it("trata o 409 como processamento e mostra progresso real", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(409, "pendente"));
    montar();

    expect(await screen.findByText("Estamos analisando seu vídeo")).toBeInTheDocument();
    expect(screen.getByText(/detecção da bola/i)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "33");
  });

  it("sai da espera automaticamente quando termina", async () => {
    vi.mocked(getProcessedVideo)
      .mockRejectedValueOnce(new ErroHttp(409, "pendente"))
      .mockResolvedValue(VIDEO_PRONTO);
    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText("Estamos analisando seu vídeo")).toBeInTheDocument();
    await act(async () => void (await vi.advanceTimersByTimeAsync(5000)));
    expect(screen.getByRole("heading", { name: "treino.mp4" })).toBeInTheDocument();
  });

  it("continua aguardando com mensagem segura se o status detalhado falhar", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(409, "pendente"));
    vi.mocked(getVideoStatus).mockRejectedValue(new Error("caiu"));
    montar();

    expect(
      await screen.findByText("Preparando o pipeline para iniciar.", { exact: false }),
    ).toBeInTheDocument();
  });

  it("diz que nao encontrou somente no 404 ou id invalido", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(404, ""));
    montar();
    expect(await screen.findByText("Não encontramos esse vídeo")).toBeInTheDocument();

    montar("/results/abacaxi");
    expect(getProcessedVideo).toHaveBeenCalledTimes(1);
  });

  it("traduz outros erros sem vazar detalhe do servico", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(503, "fila crua"));
    montar();

    expect(await screen.findByText("Não conseguimos abrir a análise")).toBeInTheDocument();
    expect(screen.getByText(ERROR_MESSAGES.QUEUE_FULL)).toBeInTheDocument();
    expect(screen.queryByText("fila crua")).toBeNull();
  });

  it("limpa o estado ao navegar para outro video", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValueOnce(new ErroHttp(404, ""));
    montar("/results/9");
    await screen.findByText("Não encontramos esse vídeo");

    vi.mocked(getProcessedVideo).mockResolvedValue(VIDEO_PRONTO);
    await userEvent.click(screen.getByText("ir para o 4"));
    expect(await screen.findByRole("heading", { name: "treino.mp4" })).toBeInTheDocument();
  });

  it("mostra o motivo quando a analise falhou", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      videoStatus: "Falhou",
      reason: "o clipe não abriu",
    });
    montar();

    expect(await screen.findByText("o clipe não abriu")).toBeInTheDocument();
  });

  it("mostra e libera o video anotado", async () => {
    const revogar = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL: vi.fn(), revokeObjectURL: revogar });
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      renderUrl: "/v1/videos/1/render",
    });
    vi.mocked(getRenderVideo).mockResolvedValue("blob:video");

    const { container, unmount } = render(
      <MemoryRouter initialEntries={["/results/1"]}>
        <Routes>
          <Route path="/results/:id" element={<Results />} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByText("Visão superior");
    expect(container.querySelector("video")).toHaveAttribute("src", "blob:video");
    unmount();
    expect(revogar).toHaveBeenCalledWith("blob:video");
    vi.unstubAllGlobals();
  });

  it("mantem os resultados se o render nao baixar", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      renderUrl: "/v1/videos/1/render",
    });
    vi.mocked(getRenderVideo).mockRejectedValue(new Error("caiu"));
    montar();

    expect(await screen.findByText("Métricas da finalização")).toBeInTheDocument();
    expect(screen.queryByText("Visão superior")).toBeNull();
  });

  it("abre uma etapa com diagnostico, duracao e evidencia", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      results: [
        RESULTADO_PRINCIPAL,
        etapa({ frame: "data:image/webp;base64,QUJD", durationMs: 120 }),
      ],
    });
    montar();

    await userEvent.click(await screen.findByText("Detecção da bola"));
    expect(screen.getByText("120 ms", { exact: false })).toBeInTheDocument();
    expect(screen.getByAltText("Quadro de demonstração de Detecção da bola")).toHaveAttribute(
      "src",
      "data:image/webp;base64,QUJD",
    );
  });

  it("completa prefixo para um frame legado em base64", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      results: [RESULTADO_PRINCIPAL, etapa({ frame: "QUJD" })],
    });
    montar();
    await userEvent.click(await screen.findByText("Detecção da bola"));
    expect(screen.getByAltText("Quadro de demonstração de Detecção da bola")).toHaveAttribute(
      "src",
      "data:image/png;base64,QUJD",
    );
  });

  it("explica onde estao as metricas consolidadas", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...VIDEO_PRONTO,
      results: [
        {
          ...RESULTADO_PRINCIPAL,
          metrics: [{ name: "gol", unit: "", type: "boolean", value: true, status: "ok" }],
        },
        etapa(),
      ],
    });
    montar();
    await userEvent.click(await screen.findByText("Detecção da bola"));
    expect(screen.getByText(/métricas consolidadas aparecem/i)).toBeInTheDocument();
  });
});
