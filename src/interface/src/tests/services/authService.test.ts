import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { autenticar, quemSou, registrar } from "../../services/authService";
import { lerSessao } from "../../services/sessao";

const fetchMock = vi.fn();

const SESSAO = { access_token: "token-recem-emitido", token_type: "bearer", expira_em: 28800 };
const EU = { id: 7, tipo: "usuario", nome: "analista@exemplo.com" };

function json(corpo: unknown, status = 200) {
  return new Response(JSON.stringify(corpo), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function chamada(indice: number): [string, RequestInit] {
  return fetchMock.mock.calls[indice] as [string, RequestInit];
}

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

describe("authService", () => {
  it("guarda a sessao com a conta que o servico confirmou", async () => {
    fetchMock.mockResolvedValueOnce(json(SESSAO)).mockResolvedValueOnce(json(EU));

    await expect(autenticar("analista@exemplo.com", "uma-senha-longa")).resolves.toEqual(EU);

    expect(lerSessao()?.token).toBe("token-recem-emitido");
    expect(lerSessao()?.conta).toBe(7);
  });

  it("confere a conta com o token recem-emitido, e nao com o que estiver no storage", async () => {
    fetchMock.mockResolvedValueOnce(json(SESSAO)).mockResolvedValueOnce(json(EU));

    await autenticar("analista@exemplo.com", "uma-senha-longa");

    const cabecalhos = chamada(1)[1].headers as Record<string, string>;

    expect(chamada(1)[0]).toBe("http://api.test/v1/auth/me");
    expect(cabecalhos.Authorization).toBe("Bearer token-recem-emitido");
  });

  it("login que falha no meio nao deixa sessao gravada, que era o balde de conta nula", async () => {
    fetchMock
      .mockResolvedValueOnce(json(SESSAO))
      .mockRejectedValueOnce(new TypeError("failed to fetch"));

    await expect(autenticar("analista@exemplo.com", "uma-senha-longa")).rejects.toThrow();

    expect(lerSessao()).toBeNull();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("nao grava sessao nenhuma quando o login e recusado", async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: "email ou senha invalidos" }, 401));

    await expect(autenticar("analista@exemplo.com", "senha-errada-longa")).rejects.toThrow();

    expect(lerSessao()).toBeNull();
  });

  it("registra a conta e devolve o principal criado", async () => {
    fetchMock.mockResolvedValueOnce(json(EU, 201));

    await expect(registrar("analista@exemplo.com", "uma-senha-longa")).resolves.toEqual(EU);
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/register");
    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      email: "analista@exemplo.com",
      senha: "uma-senha-longa",
    });
  });

  it("quemSou usa o token que recebe", async () => {
    fetchMock.mockResolvedValueOnce(json(EU));

    await expect(quemSou("outro-token")).resolves.toEqual(EU);
    expect((chamada(0)[1].headers as Record<string, string>).Authorization).toBe(
      "Bearer outro-token",
    );
  });
});
