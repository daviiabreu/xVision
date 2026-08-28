import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Results from "../../pages/Results/Results";
import { ErroHttp } from "../../services/api";
import {
  getProcessedVideo,
  getRenderVideo,
  type ProcessedVideo,
} from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

vi.mock("../../services/videoService", () => ({
  getProcessedVideo: vi.fn(),
  getRenderVideo: vi.fn(),
}));

const pronto: ProcessedVideo = {
  videoStatus: "Finalizado",
  results: [{ name: "shot", ok: true, frame: null, metrics: [] }],
  reason: null,
  renderUrl: null,
};

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
  localStorage.clear();
  vi.mocked(getProcessedVideo).mockReset();
  vi.mocked(getRenderVideo).mockReset();
});

afterEach(() => vi.useRealTimers());

describe("Results", () => {
  it("mostra o resultado quando a analise terminou", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue(pronto);

    montar();

    expect(await screen.findByText(/analisado com sucesso/)).toBeInTheDocument();
  });

  it("trata o 409 como ainda em analise, e nao como video inexistente", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(409, "esta em 'Pendente'"));

    montar();

    expect(await screen.findByText(/Estamos analisando seu vídeo/)).toBeInTheDocument();
    expect(screen.queryByText(/Não encontramos esse vídeo/)).toBeNull();
  });

  it("sai da espera do 409 sozinha quando a analise termina", async () => {
    vi.mocked(getProcessedVideo)
      .mockRejectedValueOnce(new ErroHttp(409, "esta em 'Pendente'"))
      .mockResolvedValue(pronto);

    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText(/Estamos analisando seu vídeo/)).toBeInTheDocument();

    await act(async () => void (await vi.advanceTimersByTimeAsync(5000)));

    expect(screen.getByText(/analisado com sucesso/)).toBeInTheDocument();
  });

  it("diz que nao encontrou so no 404", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(404, ""));

    montar();

    expect(await screen.findByText(/Não encontramos esse vídeo/)).toBeInTheDocument();
  });

  it("trata endereco com id nao numerico como video ausente, sem chamar o servico", async () => {
    montar("/results/abacaxi");

    expect(await screen.findByText(/Não encontramos esse vídeo/)).toBeInTheDocument();
    expect(getProcessedVideo).not.toHaveBeenCalled();
  });

  it("mostra a mensagem traduzida quando o servico recusa por outro motivo", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new ErroHttp(503, "fila cheia"));

    montar();

    expect(await screen.findByText(/Não conseguimos abrir a análise/)).toBeInTheDocument();
    expect(screen.getByText(ERROR_MESSAGES.QUEUE_FULL)).toBeInTheDocument();
    expect(screen.queryByText("fila cheia")).toBeNull();
  });

  it("cai numa mensagem de servidor quando o erro nem status tem", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValue(new Error(ERROR_MESSAGES.NO_CONNECTION));

    montar();

    expect(await screen.findByText(ERROR_MESSAGES.NO_CONNECTION)).toBeInTheDocument();
  });

  it("limpa o erro ao navegar para outro video", async () => {
    vi.mocked(getProcessedVideo).mockRejectedValueOnce(new ErroHttp(404, ""));

    montar("/results/9");
    expect(await screen.findByText(/Não encontramos esse vídeo/)).toBeInTheDocument();

    vi.mocked(getProcessedVideo).mockResolvedValue(pronto);
    await userEvent.click(screen.getByText("ir para o 4"));

    expect(await screen.findByText(/analisado com sucesso/)).toBeInTheDocument();
    expect(screen.queryByText(/Não encontramos esse vídeo/)).toBeNull();
  });

  it("mostra o motivo quando a analise falhou", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...pronto,
      videoStatus: "Falhou",
      reason: "o clipe nao abriu",
    });

    montar();

    expect(await screen.findByText(/medir esta finalização/)).toBeInTheDocument();
    expect(screen.getByText(/o clipe nao abriu/)).toBeInTheDocument();
  });

  it("mostra o video anotado quando o servico o produziu", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({ ...pronto, renderUrl: "/v1/videos/1/render" });
    vi.mocked(getRenderVideo).mockResolvedValue("blob:objeto-de-teste");

    const { container } = render(
      <MemoryRouter initialEntries={["/results/1"]}>
        <Routes>
          <Route path="/results/:id" element={<Results />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Visão superior")).toBeInTheDocument();
    expect(container.querySelector("video")).toHaveAttribute("src", "blob:objeto-de-teste");
  });

  it("devolve o object URL do video anotado ao sair da tela", async () => {
    const revogar = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL: vi.fn(), revokeObjectURL: revogar });
    vi.mocked(getProcessedVideo).mockResolvedValue({ ...pronto, renderUrl: "/v1/videos/1/render" });
    vi.mocked(getRenderVideo).mockResolvedValue("blob:objeto-de-teste");

    const { unmount } = render(
      <MemoryRouter initialEntries={["/results/1"]}>
        <Routes>
          <Route path="/results/:id" element={<Results />} />
        </Routes>
      </MemoryRouter>,
    );

    await screen.findByText("Visão superior");
    unmount();

    expect(revogar).toHaveBeenCalledWith("blob:objeto-de-teste");
    vi.unstubAllGlobals();
  });

  it("mostra a analise mesmo quando o video anotado nao baixa", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({ ...pronto, renderUrl: "/v1/videos/1/render" });
    vi.mocked(getRenderVideo).mockRejectedValue(new ErroHttp(500, ""));

    montar();

    expect(await screen.findByText(/analisado com sucesso/)).toBeInTheDocument();
    expect(screen.queryByText("Visão superior")).toBeNull();
  });

  it("abre a etapa no acordeao e diz por que ela nao tem metrica", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...pronto,
      results: [
        { name: "shot", ok: true, frame: null, metrics: [] },
        { name: "ball", ok: true, frame: null, metrics: [] },
      ],
    });

    montar();

    await userEvent.click(await screen.findByText("Detecção da bola"));

    expect(
      await screen.findByText(/Esta análise não produziu métrica nenhuma/),
    ).toBeInTheDocument();
  });

  it.each([
    ["ja vem como URL de dados", "data:image/webp;base64,QUJD", "data:image/webp;base64,QUJD"],
    ["vem como base64 cru", "QUJD", "data:image/png;base64,QUJD"],
  ])("mostra o quadro de demonstracao quando ele %s", async (_, frame, esperado) => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...pronto,
      results: [
        { name: "shot", ok: true, frame: null, metrics: [] },
        { name: "ball", ok: true, frame, metrics: [] },
      ],
    });

    montar();

    await userEvent.click(await screen.findByText("Detecção da bola"));

    expect(
      await screen.findByAltText("Quadro de demonstração de Detecção da bola"),
    ).toHaveAttribute("src", esperado);
  });

  it("manda procurar a metrica no resultado principal quando ele tem alguma", async () => {
    vi.mocked(getProcessedVideo).mockResolvedValue({
      ...pronto,
      results: [
        {
          name: "shot",
          ok: true,
          frame: null,
          metrics: [{ name: "gol", unit: "", type: "boolean", value: true, status: "ok" }],
        },
        { name: "ball", ok: true, frame: null, metrics: [] },
      ],
    });

    montar();

    await userEvent.click(await screen.findByText("Detecção da bola"));

    expect(
      await screen.findByText(/As métricas aparecem em Análise da finalização/),
    ).toBeInTheDocument();
  });
});
