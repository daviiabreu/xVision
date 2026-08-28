import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VideoContext } from "../../contexts/videoContext";
import VideoUpload from "../../pages/Home/components/VideoUpload";
import { guardarSessao } from "../../services/sessao";
import { getCachedVideos } from "../../services/videoCache";
import { enviarVideo } from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

vi.mock("../../services/videoService", () => ({ enviarVideo: vi.fn() }));

const recarregar = vi.fn().mockResolvedValue(undefined);
const remover = vi.fn();

function montar() {
  return render(
    <VideoContext.Provider value={{ videos: [], recarregar, remover }}>
      <VideoUpload />
    </VideoContext.Provider>,
  );
}

function clipe(nome = "c.mp4", tipo = "video/mp4", tamanho = 1024): File {
  const f = new File(["x"], nome, { type: tipo });
  Object.defineProperty(f, "size", { value: tamanho });

  return f;
}

async function escolher(arquivo: File) {
  const entrada = document.querySelector('input[type="file"]') as HTMLInputElement;
  await userEvent.upload(entrada, arquivo);
}

beforeEach(() => {
  localStorage.clear();
  guardarSessao("t", 3600, 1);
  vi.mocked(enviarVideo).mockReset();
  recarregar.mockClear();
});

describe("VideoUpload", () => {
  it("mostra o arquivo escolhido e oferece confirmar", async () => {
    montar();
    await escolher(clipe());

    expect(screen.getByText("c.mp4")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirmar vídeo" })).toBeEnabled();
  });

  it("recusa o arquivo grande e nao deixa nada armado para envio", async () => {
    montar();
    await escolher(clipe("grande.mp4", "video/mp4", 200 * 1024 * 1024));

    expect(await screen.findByRole("alert")).toHaveTextContent(ERROR_MESSAGES.VIDEO_TOO_LARGE);
    expect(screen.queryByRole("button", { name: "Confirmar vídeo" })).toBeNull();
  });

  it("recusa arquivo vazio antes de subir, em vez de receber 400 do servico", async () => {
    montar();
    await escolher(clipe("vazio.mp4", "video/mp4", 0));

    expect(await screen.findByRole("alert")).toHaveTextContent(ERROR_MESSAGES.VIDEO_EMPTY);
    expect(enviarVideo).not.toHaveBeenCalled();
  });

  it("limpa a selecao depois do envio, para o mesmo clipe nao ir duas vezes", async () => {
    vi.mocked(enviarVideo).mockResolvedValue(7);
    montar();
    await escolher(clipe());

    await userEvent.click(screen.getByRole("button", { name: "Confirmar vídeo" }));

    expect(await screen.findByText("Selecione seu vídeo")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirmar vídeo" })).toBeNull();
    expect(enviarVideo).toHaveBeenCalledTimes(1);
    expect(recarregar).toHaveBeenCalled();
  });

  it("grava o video no indice local, que e a unica lista de envios que existe", async () => {
    vi.mocked(enviarVideo).mockResolvedValue(7);
    montar();
    await escolher(clipe());

    await userEvent.click(screen.getByRole("button", { name: "Confirmar vídeo" }));

    await screen.findByText("Selecione seu vídeo");
    expect(getCachedVideos()).toEqual([{ id: 7, name: "c.mp4" }]);
  });

  it("mostra a mensagem da falha e mantem o arquivo para nova tentativa", async () => {
    vi.mocked(enviarVideo).mockImplementation(async () => {
      throw new Error(ERROR_MESSAGES.QUEUE_FULL);
    });
    montar();
    await escolher(clipe());

    await userEvent.click(screen.getByRole("button", { name: "Confirmar vídeo" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(ERROR_MESSAGES.QUEUE_FULL);
    expect(screen.getByText("c.mp4")).toBeInTheDocument();
    expect(getCachedVideos()).toEqual([]);
  });
});
