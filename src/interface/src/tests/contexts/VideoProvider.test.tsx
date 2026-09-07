import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useVideos } from "../../contexts/useVideos";
import { VideoProvider } from "../../contexts/VideoProvider";
import { getVideos } from "../../services/videoService";
import { resumo } from "../fixtures";

const ZERO_STATISTICS = { pending: 0, processing: 0, completed: 0, failed: 0 };

vi.mock("../../services/videoService", async (original) => ({
  ...(await original<typeof import("../../services/videoService")>()),
  getVideos: vi.fn(),
}));

function Sonda() {
  const { videos, total, statistics, loading, error, recarregar } = useVideos();
  return (
    <div>
      <span>
        {loading
          ? "carregando"
          : videos.map((video) => `${video.name}:${video.videoStatus}`).join("|")}
      </span>
      <span>total:{total}</span>
      <span>finalizadas:{statistics.completed}</span>
      <span>erro:{error ?? "nenhum"}</span>
      <button type="button" onClick={() => void recarregar()}>
        recarregar
      </button>
    </div>
  );
}

function montar() {
  render(
    <VideoProvider>
      <Sonda />
    </VideoProvider>,
  );
}

beforeEach(() => vi.mocked(getVideos).mockReset());
afterEach(() => vi.useRealTimers());

describe("VideoProvider", () => {
  it("carrega lista e total do servidor", async () => {
    vi.mocked(getVideos).mockResolvedValue({
      videos: [resumo()],
      total: 8,
      statistics: { pending: 1, processing: 2, completed: 4, failed: 1 },
    });
    montar();

    expect(await screen.findByText("treino.mp4:Finalizado")).toBeInTheDocument();
    expect(screen.getByText("total:8")).toBeInTheDocument();
    expect(screen.getByText("finalizadas:4")).toBeInTheDocument();
  });

  it("consulta novamente no intervalo indicado enquanto ha processamento", async () => {
    vi.mocked(getVideos)
      .mockResolvedValueOnce({
        videos: [resumo({ videoStatus: "Em processamento", progress: 40, retryAfterSeconds: 2 })],
        total: 1,
        statistics: { ...ZERO_STATISTICS, processing: 1 },
      })
      .mockResolvedValue({
        videos: [resumo()],
        total: 1,
        statistics: { ...ZERO_STATISTICS, completed: 1 },
      });

    vi.useFakeTimers();
    montar();
    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText("treino.mp4:Em processamento")).toBeInTheDocument();

    await act(async () => void (await vi.advanceTimersByTimeAsync(2000)));
    expect(screen.getByText("treino.mp4:Finalizado")).toBeInTheDocument();
  });

  it("para de consultar quando todos estao em estado terminal", async () => {
    vi.mocked(getVideos).mockResolvedValue({
      videos: [resumo()],
      total: 1,
      statistics: { ...ZERO_STATISTICS, completed: 1 },
    });
    vi.useFakeTimers();
    montar();
    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));

    const chamadas = vi.mocked(getVideos).mock.calls.length;
    await act(async () => void (await vi.advanceTimersByTimeAsync(30000)));
    expect(getVideos).toHaveBeenCalledTimes(chamadas);
  });

  it("continua consultando depois de uma falha transitoria durante o processamento", async () => {
    vi.mocked(getVideos)
      .mockResolvedValueOnce({
        videos: [resumo({ videoStatus: "Em processamento", retryAfterSeconds: 2 })],
        total: 1,
        statistics: { ...ZERO_STATISTICS, processing: 1 },
      })
      .mockRejectedValueOnce(new Error("sem conexão"))
      .mockResolvedValue({
        videos: [resumo()],
        total: 1,
        statistics: { ...ZERO_STATISTICS, completed: 1 },
      });
    vi.useFakeTimers();
    montar();
    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    await act(async () => void (await vi.advanceTimersByTimeAsync(2000)));
    expect(screen.getByText("treino.mp4:Em processamento")).toBeInTheDocument();

    await act(async () => void (await vi.advanceTimersByTimeAsync(2000)));

    expect(screen.getByText("treino.mp4:Finalizado")).toBeInTheDocument();
    expect(screen.getByText("erro:nenhum")).toBeInTheDocument();
  });

  it("expoe o erro sem inventar uma lista", async () => {
    const resposta = {
      total: 0,
      statistics: ZERO_STATISTICS,
    } as Awaited<ReturnType<typeof getVideos>>;
    Object.defineProperty(resposta, "videos", {
      get: () => {
        throw new Error("resposta inválida");
      },
    });
    vi.mocked(getVideos).mockResolvedValue(resposta);
    montar();

    expect(await screen.findByText("erro:resposta inválida")).toBeInTheDocument();
    expect(screen.getByText("total:0")).toBeInTheDocument();
  });

  it("permite uma nova tentativa manual", async () => {
    vi.mocked(getVideos)
      .mockRejectedValueOnce(new Error("sem conexão"))
      .mockResolvedValue({
        videos: [resumo()],
        total: 1,
        statistics: { ...ZERO_STATISTICS, completed: 1 },
      });
    montar();
    await screen.findByText("erro:sem conexão");

    await userEvent.click(screen.getByRole("button", { name: "recarregar" }));
    expect(await screen.findByText("treino.mp4:Finalizado")).toBeInTheDocument();
  });

  it("descarta resposta antiga que chega depois de uma atualizacao", async () => {
    let responderPrimeira!: (value: Awaited<ReturnType<typeof getVideos>>) => void;
    vi.mocked(getVideos)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            responderPrimeira = resolve;
          }),
      )
      .mockResolvedValue({
        videos: [resumo({ name: "novo.mp4" })],
        total: 1,
        statistics: { ...ZERO_STATISTICS, completed: 1 },
      });
    montar();
    await waitFor(() => expect(getVideos).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "recarregar" }));
    await waitFor(() => expect(getVideos).toHaveBeenCalledTimes(2));

    expect(await screen.findByText("novo.mp4:Finalizado")).toBeInTheDocument();

    responderPrimeira({
      videos: [resumo({ name: "antigo.mp4" })],
      total: 1,
      statistics: { ...ZERO_STATISTICS, completed: 1 },
    });
    await waitFor(() => expect(screen.queryByText("antigo.mp4:Finalizado")).toBeNull());
  });
});
