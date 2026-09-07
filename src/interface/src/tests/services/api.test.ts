import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  apiBlob,
  apiFetch,
  apiPublicUrl,
  apiUpload,
  ErroHttp,
  NaoAutenticado,
} from "../../services/api";
import { guardarSessao, lerSessao } from "../../services/sessao";
import { ERROR_MESSAGES } from "../../utils/errorMessages";
import { XMLHttpRequestMock } from "../xhrMock";

const fetchMock = vi.fn();

function responde(status: number, corpo: unknown = { detail: "detalhe cru do servico" }) {
  fetchMock.mockResolvedValue(
    new Response(JSON.stringify(corpo), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function cabecalhos(): Record<string, string> {
  return (fetchMock.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
}

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  XMLHttpRequestMock.reset();
  vi.stubGlobal("XMLHttpRequest", XMLHttpRequestMock);
});

afterEach(() => vi.unstubAllGlobals());

describe("api", () => {
  it("monta links publicos na mesma API usada pelas chamadas", () => {
    expect(apiPublicUrl("/docs")).toBe("http://api.test/docs");
  });

  it("manda o Bearer quando ha sessao", async () => {
    guardarSessao("abc", 3600);
    responde(200);

    await apiFetch("/v1/auth/me");

    expect(cabecalhos().Authorization).toBe("Bearer abc");
  });

  it("deixa quem chama escolher outra credencial, que e como o login confere a conta", async () => {
    guardarSessao("da-sessao", 3600);
    responde(200);

    await apiFetch("/v1/auth/me", { headers: { Authorization: "Bearer recem-emitido" } });

    expect(cabecalhos().Authorization).toBe("Bearer recem-emitido");
  });

  it("nao manda Content-Type em GET, que so obrigaria a um preflight", async () => {
    responde(200);

    await apiFetch("/v1/videos/1");

    expect(cabecalhos()["Content-Type"]).toBeUndefined();
  });

  it("manda Content-Type quando ha corpo", async () => {
    responde(200);

    await apiFetch("/v1/auth/login", { method: "POST", body: "{}" });

    expect(cabecalhos()["Content-Type"]).toBe("application/json");
  });

  it("apaga a sessao e recusa com NaoAutenticado no 401", async () => {
    guardarSessao("abc", 3600);
    responde(401);

    await expect(apiFetch("/v1/auth/me")).rejects.toBeInstanceOf(NaoAutenticado);
    expect(lerSessao()).toBeNull();
  });

  it("nao mostra o texto cru do servico: a mensagem do 401 e sobre a sessao", async () => {
    responde(401);

    await expect(apiFetch("/v1/auth/me")).rejects.toThrow(ERROR_MESSAGES.SESSION_EXPIRED);
  });

  it("guarda o status para quem precise distinguir 409 de 404", async () => {
    responde(409);

    await expect(apiFetch("/v1/videos/1")).rejects.toMatchObject({ status: 409 });
  });

  it.each([
    [400, ERROR_MESSAGES.VIDEO_EMPTY],
    [413, ERROR_MESSAGES.VIDEO_TOO_LARGE],
    [415, ERROR_MESSAGES.VIDEO_UNREADABLE],
    [422, ERROR_MESSAGES.INVALID_REQUEST],
    [429, ERROR_MESSAGES.TOO_MANY_ATTEMPTS],
    [503, ERROR_MESSAGES.QUEUE_FULL],
  ])("o upload traduz o %i em vez de dizer que o servidor caiu", async (status, mensagem) => {
    XMLHttpRequestMock.nextStatus = status;
    XMLHttpRequestMock.nextResponseText = JSON.stringify({ detail: "detalhe cru do servico" });
    const arquivo = new File(["x"], "c.mp4", { type: "video/mp4" });

    await expect(apiUpload("/v1/videos/upload/t", arquivo)).rejects.toThrow(mensagem);
  });

  it("envia credencial e informa o progresso do upload", async () => {
    guardarSessao("abc", 3600);
    const progresso = vi.fn();
    const arquivo = new File(["x"], "c.mp4", { type: "video/mp4" });

    await apiUpload("/v1/videos/upload/t", arquivo, progresso);

    const xhr = XMLHttpRequestMock.instances[0];
    expect(xhr.method).toBe("PUT");
    expect(xhr.headers.Authorization).toBe("Bearer abc");
    expect(xhr.headers["Content-Type"]).toBe("video/mp4");
    expect(progresso).toHaveBeenNthCalledWith(1, 50);
    expect(progresso).toHaveBeenLastCalledWith(100);
  });

  it("recusa endereco de outra origem antes de mandar a credencial", async () => {
    guardarSessao("abc", 3600);

    await expect(apiBlob("http://outro.test/render")).rejects.toThrow(
      ERROR_MESSAGES.UNEXPECTED_ADDRESS,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("aceita endereco absoluto da propria API", async () => {
    fetchMock.mockResolvedValue(new Response("v", { status: 200 }));

    // Confere o conteudo, e nao a classe: o `Blob` que sai do `undici` e o global do ambiente de
    // teste sao construtores diferentes no Node do CI, e `toBeInstanceOf` reprova por isso.
    const blob = await apiBlob("http://api.test/v1/videos/1/render");

    expect(await blob?.text()).toBe("v");
  });

  it("devolve null quando o render nao existe", async () => {
    responde(404);

    await expect(apiBlob("/v1/videos/1/render")).resolves.toBeNull();
  });

  it("diz que faltou conexao quando o fetch nem sai", async () => {
    fetchMock.mockRejectedValue(new TypeError("failed to fetch"));

    await expect(apiFetch("/v1/videos/1")).rejects.toThrow(ERROR_MESSAGES.NO_CONNECTION);
  });

  it("carrega o detalhe cru para o log, sem leva-lo para a tela", async () => {
    responde(500);

    await apiFetch("/v1/videos/1").catch((erro: ErroHttp) => {
      expect(erro.detalhe).toBe("detalhe cru do servico");
      expect(erro.message).toBe(ERROR_MESSAGES.SERVER_ERROR);
    });

    expect.assertions(2);
  });

  it("preserva o detalhe do 422, que o servico devolve como lista e nao como texto", async () => {
    responde(422, { detail: [{ loc: ["body", "senha"], msg: "muito curta" }] });

    await apiFetch("/v1/auth/register", { method: "POST", body: "{}" }).catch((erro: ErroHttp) => {
      expect(erro.detalhe).toContain("muito curta");
      expect(erro.detalhe).not.toBe("[object Object]");
    });

    expect.assertions(2);
  });

  it("usa a mesma origem quando o build nao define uma API separada", async () => {
    vi.stubEnv("VITE_API_URL", "");
    vi.resetModules();

    const { apiFetch: semEndereco } = await import("../../services/api");
    responde(200);

    await expect(semEndereco("/v1/auth/me")).resolves.toBeInstanceOf(Response);
    expect(fetchMock).toHaveBeenCalledWith("/v1/auth/me", expect.any(Object));

    vi.unstubAllEnvs();
    vi.resetModules();
  });
});
