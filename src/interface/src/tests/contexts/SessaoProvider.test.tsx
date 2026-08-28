import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { SessaoProvider } from "../../contexts/SessaoProvider";
import { useSessao } from "../../contexts/useSessao";
import { apagarSessao, guardarSessao } from "../../services/sessao";

function Sonda() {
  const { autenticado, sair } = useSessao();

  return (
    <div>
      <span>{autenticado ? "dentro" : "fora"}</span>
      <button type="button" onClick={sair}>
        sair
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

describe("SessaoProvider", () => {
  it("nasce autenticado quando ha sessao valida", () => {
    guardarSessao("t", 3600, 1);

    montar();

    expect(screen.getByText("dentro")).toBeInTheDocument();
  });

  it("nasce fora quando nao ha sessao", () => {
    montar();

    expect(screen.getByText("fora")).toBeInTheDocument();
  });

  it("volta para fora quando o servico apaga a sessao, que e o 401 no meio do uso", async () => {
    guardarSessao("t", 3600, 1);
    montar();
    expect(screen.getByText("dentro")).toBeInTheDocument();

    apagarSessao();

    expect(await screen.findByText("fora")).toBeInTheDocument();
  });

  it("sai pelo botao do cabecalho e limpa o storage", async () => {
    guardarSessao("t", 3600, 1);
    montar();

    await userEvent.click(screen.getByRole("button", { name: "sair" }));

    expect(screen.getByText("fora")).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toBeNull();
  });
});
