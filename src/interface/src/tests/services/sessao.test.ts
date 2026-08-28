import { beforeEach, describe, expect, it, vi } from "vitest";

import { apagarSessao, guardarSessao, lerSessao, SESSAO_PERDIDA } from "../../services/sessao";

beforeEach(() => localStorage.clear());

describe("sessao", () => {
  it("trata expira_em como duracao, e nao como epoch", () => {
    guardarSessao("t", 3600);

    expect(lerSessao()?.expiraEm).toBeGreaterThan(Date.now() / 1000);
  });

  it("descarta a sessao vencida na leitura", () => {
    guardarSessao("t", -1);

    expect(lerSessao()).toBeNull();
  });

  it("devolve null e limpa quando o storage esta corrompido", () => {
    localStorage.setItem("sessao", "{nao e json");

    expect(lerSessao()).toBeNull();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it.each([
    ["sem expiraEm, que sem conferencia viraria sessao eterna", { token: "t", conta: 1 }],
    ["sem token", { expiraEm: 9_999_999_999, conta: 1 }],
    ["com token vazio", { token: "", expiraEm: 9_999_999_999, conta: 1 }],
    ["com conta de outro tipo", { token: "t", expiraEm: 9_999_999_999, conta: "1" }],
  ])("recusa JSON valido com forma errada: %s", (_, corpo) => {
    localStorage.setItem("sessao", JSON.stringify(corpo));

    expect(lerSessao()).toBeNull();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("avisa quem escuta ao apagar, que e como a tela volta para o login", () => {
    const ouvinte = vi.fn();
    window.addEventListener(SESSAO_PERDIDA, ouvinte);

    apagarSessao();

    expect(ouvinte).toHaveBeenCalled();
    window.removeEventListener(SESSAO_PERDIDA, ouvinte);
  });

  it("guarda a conta, que e o que separa o cache de videos entre pessoas", () => {
    guardarSessao("t", 3600, 7);

    expect(lerSessao()?.conta).toBe(7);
  });
});
