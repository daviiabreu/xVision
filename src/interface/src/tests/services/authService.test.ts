import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  alterarSenha,
  autenticar,
  atualizarEmail,
  buscarEstatisticasConta,
  quemSou,
  redefinirSenha,
  registrar,
  solicitarRedefinicao,
} from "../../services/authService";
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
    expect(lerSessao()?.nome).toBe("analista@exemplo.com");
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

  it("solicita a recuperacao sem depender de uma sessao", async () => {
    fetchMock.mockResolvedValueOnce(json({ mensagem: "Verifique seu email." }, 202));

    await expect(solicitarRedefinicao("analista@exemplo.com")).resolves.toBe(
      "Verifique seu email.",
    );
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/password/forgot");
    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({ email: "analista@exemplo.com" });
  });

  it("envia o token e a nova senha na confirmacao", async () => {
    fetchMock.mockResolvedValueOnce(json({ mensagem: "Senha redefinida." }));

    await expect(redefinirSenha("token-de-recuperacao-longo", "uma-senha-nova")).resolves.toBe(
      "Senha redefinida.",
    );
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/password/reset");
    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      token: "token-de-recuperacao-longo",
      senha: "uma-senha-nova",
    });
  });

  it("atualiza o email confirmando a senha atual", async () => {
    fetchMock.mockResolvedValueOnce(json({ ...EU, nome: "novo@exemplo.com" }));

    await expect(atualizarEmail("novo@exemplo.com", "senha-atual-longa")).resolves.toMatchObject({
      nome: "novo@exemplo.com",
    });
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/me");
    expect(chamada(0)[1].method).toBe("PATCH");
    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      email: "novo@exemplo.com",
      senha_atual: "senha-atual-longa",
    });
  });

  it("troca a senha pela rota autenticada", async () => {
    localStorage.setItem(
      "sessao",
      JSON.stringify({
        token: "token-antigo",
        expiraEm: Math.floor(Date.now() / 1000) + 100,
        conta: 7,
        nome: "analista@exemplo.com",
      }),
    );
    fetchMock.mockResolvedValueOnce(
      json({
        mensagem: "Senha atualizada.",
        access_token: "token-renovado",
        token_type: "bearer",
        expira_em: 28800,
      }),
    );

    await expect(alterarSenha("senha-atual-longa", "senha-nova-bem-longa")).resolves.toBe(
      "Senha atualizada.",
    );
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/password/change");
    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      senha_atual: "senha-atual-longa",
      nova_senha: "senha-nova-bem-longa",
    });
    expect(lerSessao()).toMatchObject({
      token: "token-renovado",
      conta: 7,
      nome: "analista@exemplo.com",
    });
  });

  it("busca a contagem completa da conta", async () => {
    const estatisticas = {
      total: 9,
      pendentes: 1,
      em_processamento: 2,
      finalizadas: 5,
      falhas: 1,
    };
    fetchMock.mockResolvedValueOnce(json(estatisticas));

    await expect(buscarEstatisticasConta()).resolves.toEqual(estatisticas);
    expect(chamada(0)[0]).toBe("http://api.test/v1/auth/me/statistics");
  });
});
