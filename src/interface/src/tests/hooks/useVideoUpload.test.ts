import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useVideoUpload } from "../../hooks/useVideoUpload";
import { ErroHttp } from "../../services/api";
import { enviarVideo } from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

vi.mock("../../services/videoService", () => ({ enviarVideo: vi.fn() }));

function arquivo(nome: string, tipo = "video/mp4", tamanho = 1024): File {
  const f = new File(["x"], nome, { type: tipo });
  Object.defineProperty(f, "size", { value: tamanho });

  return f;
}

beforeEach(() => {
  vi.mocked(enviarVideo).mockReset();
});

describe("useVideoUpload", () => {
  it("aceita um video valido", () => {
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("c.mp4")));

    expect(result.current.file?.name).toBe("c.mp4");
    expect(result.current.error).toBeNull();
  });

  it("derruba o arquivo anterior quando o novo e recusado", () => {
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("bom.mp4")));
    act(() =>
      result.current.handleFileSelect(arquivo("grande.mp4", "video/mp4", 200 * 1024 * 1024)),
    );

    expect(result.current.error).toBe(ERROR_MESSAGES.VIDEO_TOO_LARGE);
    expect(result.current.file).toBeNull();
  });

  it("recusa o que nao e video", () => {
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("nota.pdf", "application/pdf")));

    expect(result.current.error).toBe(ERROR_MESSAGES.VIDEO_INVALID);
    expect(result.current.file).toBeNull();
  });

  it("recusa arquivo de zero byte, que o servico devolveria como 400", () => {
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("vazio.mp4", "video/mp4", 0)));

    expect(result.current.error).toBe(ERROR_MESSAGES.VIDEO_EMPTY);
    expect(result.current.file).toBeNull();
  });

  it("limpa a selecao depois do envio, para o mesmo clipe nao ir duas vezes", async () => {
    vi.mocked(enviarVideo).mockResolvedValue(7);
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("c.mp4")));
    await act(async () => {
      await expect(result.current.confirmVideo()).resolves.toBe(7);
    });

    await waitFor(() => expect(result.current.file).toBeNull());
    await act(async () => {
      await expect(result.current.confirmVideo()).resolves.toBeNull();
    });

    expect(enviarVideo).toHaveBeenCalledTimes(1);
  });

  it.each([404, 409])(
    "explica que o endereco de envio expirou ou foi usado quando o servico responde %i",
    async (status) => {
      vi.mocked(enviarVideo).mockRejectedValue(new ErroHttp(status, "endereco de envio"));
      const { result } = renderHook(() => useVideoUpload());

      act(() => result.current.handleFileSelect(arquivo("c.mp4")));
      await act(async () => {
        await expect(result.current.confirmVideo()).resolves.toBeNull();
      });

      expect(result.current.error).toBe(ERROR_MESSAGES.UPLOAD_ADDRESS_EXPIRED);
      expect(result.current.file?.name).toBe("c.mp4");
    },
  );

  it("preserva a selecao trocada durante o envio, em vez de apagar a escolha nova", async () => {
    let liberar!: (id: number) => void;
    const emVoo = new Promise<number>((resolve) => {
      liberar = resolve;
    });
    vi.mocked(enviarVideo).mockReturnValue(emVoo);
    const { result } = renderHook(() => useVideoUpload());

    act(() => result.current.handleFileSelect(arquivo("primeiro.mp4")));

    let envio!: Promise<number | null>;
    act(() => {
      envio = result.current.confirmVideo();
    });

    act(() => result.current.handleFileSelect(arquivo("segundo.mp4")));
    liberar(7);
    await act(async () => {
      await envio;
    });

    expect(result.current.file?.name).toBe("segundo.mp4");
  });
});
