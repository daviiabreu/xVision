import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useVideos } from "../../contexts/useVideos";
import { VideoProvider } from "../../contexts/VideoProvider";
import { guardarSessao } from "../../services/sessao";
import { addCachedVideo, getCachedVideos } from "../../services/videoCache";
import { getVideoStatus, type VideoStatus } from "../../services/videoService";

vi.mock("../../services/videoService", async (original) => ({
  ...(await original<typeof import("../../services/videoService")>()),
  getVideoStatus: vi.fn(),
}));

function Sonda() {
  const { videos, recarregar, remover } = useVideos();

  return (
    <div>
      <span>{videos.map((v) => `${v.name}:${v.status ?? "erro"}`).join("|")}</span>
      <button type="button" onClick={() => void recarregar()}>
        recarregar
      </button>
      <button type="button" onClick={() => remover(1)}>
        remover
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

beforeEach(() => {
  localStorage.clear();
  vi.mocked(getVideoStatus).mockReset();
  guardarSessao("t", 3600, 1);
  addCachedVideo({ id: 1, name: "c.mp4" });
});

afterEach(() => vi.useRealTimers());

describe("VideoProvider", () => {
  it("consulta de novo ate a analise terminar", async () => {
    vi.mocked(getVideoStatus)
      .mockResolvedValueOnce({ video_status: "Pendente", motivo: null })
      .mockResolvedValue({ video_status: "Finalizado", motivo: null });

    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText("c.mp4:Pendente")).toBeInTheDocument();

    await act(async () => void (await vi.advanceTimersByTimeAsync(5000)));
    expect(screen.getByText("c.mp4:Finalizado")).toBeInTheDocument();
  });

  it("para de consultar depois do estado terminal", async () => {
    vi.mocked(getVideoStatus).mockResolvedValue({ video_status: "Finalizado", motivo: null });

    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText("c.mp4:Finalizado")).toBeInTheDocument();

    const chamadas = vi.mocked(getVideoStatus).mock.calls.length;
    await act(async () => void (await vi.advanceTimersByTimeAsync(20000)));

    expect(vi.mocked(getVideoStatus).mock.calls.length).toBe(chamadas);
  });

  it("marca o erro sem derrubar a lista quando a consulta falha", async () => {
    vi.mocked(getVideoStatus).mockImplementation(async () => {
      throw new Error("caiu");
    });

    montar();

    expect(await screen.findByText("c.mp4:erro")).toBeInTheDocument();
  });

  it("para de consultar o que so devolve erro, em vez de bater no servico para sempre", async () => {
    vi.mocked(getVideoStatus).mockRejectedValue(new Error("caiu"));

    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    expect(screen.getByText("c.mp4:erro")).toBeInTheDocument();

    const chamadas = vi.mocked(getVideoStatus).mock.calls.length;
    await act(async () => void (await vi.advanceTimersByTimeAsync(60000)));

    expect(vi.mocked(getVideoStatus).mock.calls.length).toBe(chamadas);
  });

  it("continua consultando o que ainda roda mesmo com outro em erro", async () => {
    addCachedVideo({ id: 2, name: "d.mp4" });
    vi.mocked(getVideoStatus).mockImplementation(async (id: number) => {
      if (id === 1) {
        throw new Error("caiu");
      }
      return { video_status: "Pendente", motivo: null };
    });

    vi.useFakeTimers();
    montar();

    await act(async () => void (await vi.advanceTimersByTimeAsync(0)));
    const chamadas = vi.mocked(getVideoStatus).mock.calls.length;

    await act(async () => void (await vi.advanceTimersByTimeAsync(5000)));

    expect(vi.mocked(getVideoStatus).mock.calls.length).toBeGreaterThan(chamadas);
  });

  it("descarta a resposta que chega fora de ordem, em vez de sobrescrever a mais nova", async () => {
    const pendentes: ((valor: { video_status: VideoStatus; motivo: null }) => void)[] = [];
    vi.mocked(getVideoStatus).mockImplementation(
      () => new Promise((resolve) => pendentes.push(resolve)),
    );

    montar();
    await act(async () => await Promise.resolve());

    addCachedVideo({ id: 2, name: "d.mp4" });
    await act(async () => {
      screen.getByRole("button", { name: "recarregar" }).click();
      await Promise.resolve();
    });

    expect(pendentes).toHaveLength(3);

    await act(async () => {
      pendentes[1]({ video_status: "Pendente", motivo: null });
      pendentes[2]({ video_status: "Pendente", motivo: null });
      await Promise.resolve();
    });
    expect(screen.getByText("c.mp4:Pendente|d.mp4:Pendente")).toBeInTheDocument();

    await act(async () => {
      pendentes[0]({ video_status: "Finalizado", motivo: null });
      await Promise.resolve();
    });

    expect(screen.getByText("c.mp4:Pendente|d.mp4:Pendente")).toBeInTheDocument();
  });

  it("remove do indice local o video que a pessoa descarta", async () => {
    vi.mocked(getVideoStatus).mockRejectedValue(new Error("caiu"));

    montar();
    await screen.findByText("c.mp4:erro");

    await userEvent.click(screen.getByRole("button", { name: "remover" }));

    expect(getCachedVideos()).toEqual([]);
  });
});
