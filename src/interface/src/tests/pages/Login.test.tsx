import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Login from "../../pages/Login/Login";
import { ErroHttp } from "../../services/api";
import { autenticar, registrar } from "../../services/authService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";

vi.mock("../../services/authService", () => ({
  autenticar: vi.fn(),
  registrar: vi.fn(),
}));

const entrou = vi.fn();

const EU = { id: 7, tipo: "usuario" as const, nome: "analista@exemplo.com" };

function montar() {
  render(<Login onEntrou={entrou} />);
}

async function preencher(senha = "uma-senha-longa-de-verdade") {
  await userEvent.type(screen.getByLabelText("Email"), "analista@exemplo.com");
  await userEvent.type(screen.getByLabelText("Senha"), senha);
}

beforeEach(() => {
  entrou.mockClear();
  vi.mocked(autenticar).mockReset();
  vi.mocked(registrar).mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

describe("Login", () => {
  it("entra e avisa quem cuida da sessao", async () => {
    vi.mocked(autenticar).mockResolvedValue(EU);
    montar();
    await preencher();

    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));

    expect(autenticar).toHaveBeenCalledWith("analista@exemplo.com", "uma-senha-longa-de-verdade");
    expect(entrou).toHaveBeenCalledTimes(1);
  });

  it("cria a conta antes de entrar quando o formulario esta em modo de registro", async () => {
    vi.mocked(registrar).mockResolvedValue(EU);
    vi.mocked(autenticar).mockResolvedValue(EU);
    montar();

    await userEvent.click(screen.getByRole("button", { name: "Criar uma conta" }));
    await preencher();
    await userEvent.click(screen.getByRole("button", { name: "Criar conta" }));

    expect(registrar).toHaveBeenCalledTimes(1);
    expect(autenticar).toHaveBeenCalledTimes(1);
  });

  it.each([
    [401, "Email ou senha incorretos."],
    [409, "Este email já tem conta. Entre em vez de criar."],
    [422, "Verifique o email e a senha e tente de novo."],
  ])("traduz o %i em mensagem para quem usa", async (status, mensagem) => {
    vi.mocked(autenticar).mockRejectedValue(new ErroHttp(status, "texto cru do servico"));
    montar();
    await preencher();

    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(mensagem);
    expect(entrou).not.toHaveBeenCalled();
  });

  it("mostra a mensagem de conexao quando a chamada nem sai", async () => {
    vi.mocked(autenticar).mockRejectedValue(new Error(ERROR_MESSAGES.NO_CONNECTION));
    montar();
    await preencher();

    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(ERROR_MESSAGES.NO_CONNECTION);
  });

  it("liga a mensagem de erro aos dois campos, para quem volta corrigir saber o motivo", async () => {
    vi.mocked(autenticar).mockRejectedValue(new ErroHttp(401, ""));
    montar();
    await preencher();

    await userEvent.click(screen.getByRole("button", { name: "Entrar" }));
    await screen.findByRole("alert");

    for (const campo of ["Email", "Senha"]) {
      const entrada = screen.getByLabelText(campo);
      expect(entrada).toHaveAttribute("aria-invalid", "true");
      expect(entrada.getAttribute("aria-describedby")).toContain("erro-login");
    }
  });

  it("declara o proposito de cada campo para o gerenciador de senhas", async () => {
    montar();

    expect(screen.getByLabelText("Email")).toHaveAttribute("autocomplete", "email");
    expect(screen.getByLabelText("Senha")).toHaveAttribute("autocomplete", "current-password");

    await userEvent.click(screen.getByRole("button", { name: "Criar uma conta" }));

    expect(screen.getByLabelText("Senha")).toHaveAttribute("autocomplete", "new-password");
  });
});
