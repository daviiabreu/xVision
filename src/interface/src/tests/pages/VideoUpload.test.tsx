import { act, createEvent, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VideoContext } from "../../contexts/videoContext";
import VideoUpload from "../../pages/Home/components/VideoUpload";
import { enviarVideo } from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

vi.mock("../../services/videoService", () => ({ enviarVideo: vi.fn() }));

const recarregar = vi.fn().mockResolvedValue(undefined);

function montar() {
  return render(
    <VideoContext.Provider
      value={{
        videos: [],
        total: 0,
        statistics: { pending: 0, processing: 0, completed: 0, failed: 0 },
        loading: false,
        error: null,
        recarregar,
      }}
    >
      <VideoUpload />
    </VideoContext.Provider>,
  );
}

function clipe(nome = "c.mp4", tipo = "video/mp4", tamanho = 1024): File {
  const file = new File(["x"], nome, { type: tipo });
  Object.defineProperty(file, "size", { value: tamanho });
  return file;
}

async function escolher(arquivo: File) {
  await userEvent.upload(screen.getByLabelText("Selecionar vídeo"), arquivo);
}

beforeEach(() => {
  vi.mocked(enviarVideo).mockReset();
  recarregar.mockClear();
});

describe("VideoUpload", () => {
  it("permite selecionar pelo controle acessivel e iniciar a analise", async () => {
    montar();
    await escolher(clipe());

    expect(screen.getByText("c.mp4")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Iniciar análise" })).toBeEnabled();
  });

  it("aceita arrastar e soltar um clipe", () => {
    const { container } = montar();
    const area =
      container.querySelector("[ondrop]") ??
      screen.getByText("Arraste um vídeo até aqui").parentElement;
    fireEvent.drop(area!, { dataTransfer: { files: [clipe("arrastado.mp4")] } });
    expect(screen.getByText("arrastado.mp4")).toBeInTheDocument();
  });

  it("recusa arquivo grande antes do envio", async () => {
    montar();
    await escolher(clipe("grande.mp4", "video/mp4", 200 * 1024 * 1024));
    expect(screen.getByRole("alert")).toHaveTextContent(ERROR_MESSAGES.VIDEO_TOO_LARGE);
    expect(screen.getByRole("button", { name: "Iniciar análise" })).toBeDisabled();
  });

  it("limpa a selecao e recarrega a lista depois do envio", async () => {
    vi.mocked(enviarVideo).mockResolvedValue(7);
    montar();
    await escolher(clipe());
    await userEvent.click(screen.getByRole("button", { name: "Iniciar análise" }));

    expect(await screen.findByText("Arraste um vídeo até aqui")).toBeInTheDocument();
    expect(enviarVideo).toHaveBeenCalledTimes(1);
    expect(recarregar).toHaveBeenCalled();
  });

  it("mostra progresso real reportado pelo transporte", async () => {
    let liberar!: (id: number) => void;
    let relatarProgresso: ((percentual: number) => void) | undefined;
    vi.mocked(enviarVideo).mockImplementation(async (_file, onProgress) => {
      relatarProgresso = onProgress;
      return await new Promise<number>((resolve) => {
        liberar = resolve;
      });
    });
    montar();
    await escolher(clipe());
    await userEvent.click(screen.getByRole("button", { name: "Iniciar análise" }));

    await waitFor(() => expect(enviarVideo).toHaveBeenCalledOnce());
    act(() => relatarProgresso?.(37));
    expect(await screen.findByText("37%")).toBeInTheDocument();
    await act(async () => liberar(7));
  });

  it("mantem o arquivo para nova tentativa quando o envio falha", async () => {
    vi.mocked(enviarVideo).mockRejectedValue(new Error(ERROR_MESSAGES.QUEUE_FULL));
    montar();
    await escolher(clipe());
    await userEvent.click(screen.getByRole("button", { name: "Iniciar análise" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(ERROR_MESSAGES.QUEUE_FULL);
    expect(screen.getByText("c.mp4")).toBeInTheDocument();
  });

  it("mantem o destaque ao arrastar sobre os elementos internos da area", () => {
    montar();
    const area = screen.getByText("Arraste um vídeo até aqui").parentElement!;

    fireEvent.dragEnter(area);
    expect(area.className).toContain("border-amber-500");

    const saida = createEvent.dragLeave(area);
    Object.defineProperty(saida, "relatedTarget", { value: area.firstElementChild });
    fireEvent(area, saida);

    expect(area.className).toContain("border-amber-500");
  });

  it("remove o destaque quando o arraste sai da area", () => {
    montar();
    const area = screen.getByText("Arraste um vídeo até aqui").parentElement!;

    fireEvent.dragEnter(area);
    const saida = createEvent.dragLeave(area);
    Object.defineProperty(saida, "relatedTarget", { value: document.body });
    fireEvent(area, saida);

    expect(area.className).not.toContain("border-amber-500");
  });

  it("ignora arquivos soltos enquanto o envio esta em andamento", async () => {
    vi.mocked(enviarVideo).mockReturnValue(new Promise(() => undefined));
    montar();
    await escolher(clipe("enviando.mp4"));
    await userEvent.click(screen.getByRole("button", { name: "Iniciar análise" }));

    const area = screen.getByText("enviando.mp4").parentElement!;
    fireEvent.drop(area, { dataTransfer: { files: [clipe("intruso.mp4")] } });

    expect(screen.getByText("enviando.mp4")).toBeInTheDocument();
    expect(screen.queryByText("intruso.mp4")).toBeNull();
  });
});
