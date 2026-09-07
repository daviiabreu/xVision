import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessaoContext } from "../../contexts/sessaoContext";
import Account from "../../pages/Account/Account";
import { ErroHttp } from "../../services/api";
import { alterarSenha, atualizarEmail, buscarEstatisticasConta } from "../../services/authService";

vi.mock("../../services/authService", () => ({
  alterarSenha: vi.fn(),
  atualizarEmail: vi.fn(),
  buscarEstatisticasConta: vi.fn(),
}));

const atualizarNome = vi.fn();

function montar() {
  render(
    <MemoryRouter>
      <SessaoContext.Provider
        value={{
          autenticado: true,
          nome: "analista@exemplo.com",
          atualizarNome,
          entrou: vi.fn(),
          sair: vi.fn(),
        }}
      >
        <Account />
      </SessaoContext.Provider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  atualizarNome.mockReset();
  vi.mocked(alterarSenha).mockReset();
  vi.mocked(atualizarEmail).mockReset();
  vi.mocked(buscarEstatisticasConta).mockReset();
  vi.mocked(buscarEstatisticasConta).mockResolvedValue({
    total: 12,
    pendentes: 2,
    em_processamento: 1,
    finalizadas: 8,
    falhas: 1,
  });
});

describe("Account", () => {
  it("mostra as estatisticas completas da conta", async () => {
    montar();

    expect(await screen.findByText("12")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(buscarEstatisticasConta).toHaveBeenCalledTimes(1);
  });

  it("explica quando as estatisticas nao podem ser carregadas", async () => {
    vi.mocked(buscarEstatisticasConta).mockRejectedValue(new Error("Serviço indisponível."));
    montar();

    expect(await screen.findByRole("alert")).toHaveTextContent("Serviço indisponível.");
  });

  it("atualiza o email e reflete o novo identificador na sessao", async () => {
    vi.mocked(atualizarEmail).mockResolvedValue({
      id: 7,
      tipo: "usuario",
      nome: "novo@exemplo.com",
    });
    montar();

    const email = screen.getByLabelText("Novo email");
    await userEvent.clear(email);
    await userEvent.type(email, "novo@exemplo.com");
    await userEvent.type(screen.getAllByLabelText("Senha atual")[0], "senha-atual-bem-longa");
    await userEvent.click(screen.getByRole("button", { name: "Atualizar email" }));

    expect(atualizarEmail).toHaveBeenCalledWith("novo@exemplo.com", "senha-atual-bem-longa");
    expect(atualizarNome).toHaveBeenCalledWith("novo@exemplo.com");
    expect(await screen.findByRole("status")).toHaveTextContent("Email atualizado");
  });

  it("nao envia senhas novas que nao coincidem", async () => {
    montar();

    await userEvent.type(screen.getAllByLabelText("Senha atual")[1], "senha-atual-bem-longa");
    await userEvent.type(screen.getByLabelText("Nova senha"), "uma-nova-senha-bem-longa");
    await userEvent.type(screen.getByLabelText("Confirme a nova senha"), "uma-senha-diferente");
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(alterarSenha).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("precisam ser iguais");
  });

  it("troca a senha e limpa os tres campos", async () => {
    vi.mocked(alterarSenha).mockResolvedValue("Senha atualizada com sucesso.");
    montar();

    const atual = screen.getAllByLabelText("Senha atual")[1];
    const nova = screen.getByLabelText("Nova senha");
    const confirmacao = screen.getByLabelText("Confirme a nova senha");
    await userEvent.type(atual, "senha-atual-bem-longa");
    await userEvent.type(nova, "uma-nova-senha-bem-longa");
    await userEvent.type(confirmacao, "uma-nova-senha-bem-longa");
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(alterarSenha).toHaveBeenCalledWith("senha-atual-bem-longa", "uma-nova-senha-bem-longa");
    expect(await screen.findByRole("status")).toHaveTextContent("Senha atualizada");
    expect(atual).toHaveValue("");
    expect(nova).toHaveValue("");
    expect(confirmacao).toHaveValue("");
  });

  it("traduz email repetido sem mostrar o detalhe interno", async () => {
    vi.mocked(atualizarEmail).mockRejectedValue(new ErroHttp(409, "detalhe interno"));
    montar();

    await userEvent.type(screen.getAllByLabelText("Senha atual")[0], "senha-atual-bem-longa");
    await userEvent.click(screen.getByRole("button", { name: "Atualizar email" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("já está em uso");
    expect(screen.getByRole("alert")).not.toHaveTextContent("detalhe interno");
  });
});
