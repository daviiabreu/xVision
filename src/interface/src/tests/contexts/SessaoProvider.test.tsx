import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SessaoProvider } from "../../contexts/SessaoProvider";
import { useSessao } from "../../contexts/useSessao";
import { apagarSessao, guardarSessao, renovarSessao } from "../../services/sessao";

function Sonda() {
  const { autenticado, nome, atualizarNome, sair } = useSessao();

  return (
    <div>
      <span>{autenticado ? "dentro" : "fora"}</span>
      <span>{nome ?? "sem nome"}</span>
      <button type="button" onClick={sair}>
        sair
      </button>
      <button type="button" onClick={() => atualizarNome("novo@exemplo.com")}>
        atualizar nome
      </button>
    </div>
  );
}

function montar() {
  render(
    <SessaoProvider>
      <Sonda />
    </SessaoProvider>,
  );
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.useRealTimers());

describe("SessaoProvider", () => {
  it("nasce autenticado quando ha sessao valida", () => {
    guardarSessao("t", 3600, 1, "analista@exemplo.com");

    montar();

    expect(screen.getByText("dentro")).toBeInTheDocument();
    expect(screen.getByText("analista@exemplo.com")).toBeInTheDocument();
  });

  it("nasce fora quando nao ha sessao", () => {
    montar();

    expect(screen.getByText("fora")).toBeInTheDocument();
  });

  it("volta para fora quando o servico apaga a sessao, que e o 401 no meio do uso", async () => {
    guardarSessao("t", 3600, 1);
    montar();
    expect(screen.getByText("dentro")).toBeInTheDocument();

    act(() => apagarSessao());

    expect(await screen.findByText("fora")).toBeInTheDocument();
  });

  it("sai pelo botao do cabecalho e limpa o storage", async () => {
    guardarSessao("t", 3600, 1);
    montar();

    await userEvent.click(screen.getByRole("button", { name: "sair" }));

    expect(screen.getByText("fora")).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("reflete o novo email no contexto e no storage", async () => {
    guardarSessao("t", 3600, 1, "antigo@exemplo.com");
    montar();

    await userEvent.click(screen.getByRole("button", { name: "atualizar nome" }));

    expect(screen.getByText("novo@exemplo.com")).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toContain("novo@exemplo.com");
  });

  it("acompanha o logout feito em outra aba", async () => {
    guardarSessao("t", 3600, 1);
    montar();
    expect(screen.getByText("dentro")).toBeInTheDocument();

    act(() => {
      localStorage.removeItem("sessao");
      window.dispatchEvent(new StorageEvent("storage", { key: "sessao", newValue: null }));
    });

    expect(await screen.findByText("fora")).toBeInTheDocument();
  });

  it("acompanha o login feito em outra aba", async () => {
    montar();
    expect(screen.getByText("fora")).toBeInTheDocument();

    act(() => {
      guardarSessao("t", 3600, 1, "outra@exemplo.com");
      window.dispatchEvent(new StorageEvent("storage", { key: "sessao" }));
    });

    expect(await screen.findByText("dentro")).toBeInTheDocument();
  });

  it("sai sozinho quando a sessao expira com a aba aberta e parada", () => {
    vi.useFakeTimers();
    guardarSessao("t", 60, 1);
    montar();
    expect(screen.getByText("dentro")).toBeInTheDocument();

    act(() => void vi.advanceTimersByTime(59_000));
    expect(screen.getByText("dentro")).toBeInTheDocument();

    act(() => void vi.advanceTimersByTime(2_000));
    expect(screen.getByText("fora")).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("segue a validade renovada pela troca de senha em vez de sair no prazo antigo", () => {
    vi.useFakeTimers();
    guardarSessao("antigo", 60, 1);
    montar();

    act(() => {
      renovarSessao("novo", 3600);
      void vi.advanceTimersByTime(61_000);
    });

    expect(screen.getByText("dentro")).toBeInTheDocument();
  });

  it("ignora mudancas de outras chaves do storage", () => {
    guardarSessao("t", 3600, 1);
    montar();

    act(() => {
      window.dispatchEvent(new StorageEvent("storage", { key: "outra-chave" }));
    });

    expect(screen.getByText("dentro")).toBeInTheDocument();
  });
});
