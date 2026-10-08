import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ForgotPassword from "../../pages/ForgotPassword/ForgotPassword";
import ResetPassword from "../../pages/ResetPassword/ResetPassword";
import { ErroHttp } from "../../services/api";
import { redefinirSenha, solicitarRedefinicao } from "../../services/authService";

vi.mock("../../services/authService", () => ({
  solicitarRedefinicao: vi.fn(),
  redefinirSenha: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(solicitarRedefinicao).mockReset();
  vi.mocked(redefinirSenha).mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

describe("recuperacao de senha", () => {
  it("pede o link e mostra a resposta que nao enumera contas", async () => {
    vi.mocked(solicitarRedefinicao).mockResolvedValue(
      "Se o email estiver cadastrado, enviaremos um link de redefinicao.",
    );
    render(
      <MemoryRouter>
        <ForgotPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Email"), "pessoa@exemplo.com");
    await userEvent.click(screen.getByRole("button", { name: "Enviar link de recuperação" }));

    expect(solicitarRedefinicao).toHaveBeenCalledWith("pessoa@exemplo.com");
    expect(await screen.findByRole("status")).toHaveTextContent("Se o email estiver cadastrado");
  });

  it("explica quando o ambiente ainda nao envia email", async () => {
    vi.mocked(solicitarRedefinicao).mockRejectedValue(new ErroHttp(503));
    render(
      <MemoryRouter>
        <ForgotPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Email"), "pessoa@exemplo.com");
    await userEvent.click(screen.getByRole("button", { name: "Enviar link de recuperação" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "envio de email ainda não foi configurado",
    );
  });

  it("explica o limite de tentativas da recuperacao", async () => {
    vi.mocked(solicitarRedefinicao).mockRejectedValue(new ErroHttp(429));
    render(
      <MemoryRouter>
        <ForgotPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Email"), "pessoa@exemplo.com");
    await userEvent.click(screen.getByRole("button", { name: "Enviar link de recuperação" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Muitas tentativas seguidas");
  });

  it("recusa senhas diferentes antes de chamar o servico", async () => {
    render(
      <MemoryRouter initialEntries={["/reset-password?token=token-de-uso-unico-comprido"]}>
        <ResetPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Nova senha"), "uma-senha-nova-de-verdade");
    await userEvent.type(
      screen.getByLabelText("Confirme a nova senha"),
      "outra-senha-bem-diferente",
    );
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(screen.getByRole("alert")).toHaveTextContent("precisam ser iguais");
    expect(redefinirSenha).not.toHaveBeenCalled();
  });

  it("troca a senha com o token da URL", async () => {
    vi.mocked(redefinirSenha).mockResolvedValue("Senha redefinida.");
    render(
      <MemoryRouter initialEntries={["/reset-password?token=token-de-uso-unico-comprido"]}>
        <ResetPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Nova senha"), "uma-senha-nova-de-verdade");
    await userEvent.type(
      screen.getByLabelText("Confirme a nova senha"),
      "uma-senha-nova-de-verdade",
    );
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(redefinirSenha).toHaveBeenCalledWith(
      "token-de-uso-unico-comprido",
      "uma-senha-nova-de-verdade",
    );
    expect(await screen.findByRole("status")).toHaveTextContent("Senha atualizada");
  });

  it("traduz token expirado sem exibir o detalhe interno", async () => {
    vi.mocked(redefinirSenha).mockRejectedValue(new ErroHttp(400, "detalhe interno"));
    render(
      <MemoryRouter initialEntries={["/reset-password?token=token-de-uso-unico-comprido"]}>
        <ResetPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Nova senha"), "uma-senha-nova-de-verdade");
    await userEvent.type(
      screen.getByLabelText("Confirme a nova senha"),
      "uma-senha-nova-de-verdade",
    );
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("inválido, já foi usado ou expirou");
    expect(screen.getByRole("alert")).not.toHaveTextContent("detalhe interno");
  });

  it("orienta a pedir outro link quando o servico recusa o formato do token com 422", async () => {
    vi.mocked(redefinirSenha).mockRejectedValue(new ErroHttp(422, "detalhe interno"));
    render(
      <MemoryRouter initialEntries={["/reset-password?token=token-cortado"]}>
        <ResetPassword />
      </MemoryRouter>,
    );

    await userEvent.type(screen.getByLabelText("Nova senha"), "uma-senha-nova-de-verdade");
    await userEvent.type(
      screen.getByLabelText("Confirme a nova senha"),
      "uma-senha-nova-de-verdade",
    );
    await userEvent.click(screen.getByRole("button", { name: "Atualizar senha" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("incompleto");
    expect(screen.getByRole("alert")).not.toHaveTextContent("detalhe interno");
  });

  it("orienta a pedir outro link quando o token esta ausente", () => {
    render(
      <MemoryRouter initialEntries={["/reset-password"]}>
        <ResetPassword />
      </MemoryRouter>,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("não contém um token");
    expect(screen.getByRole("link", { name: "Solicitar um novo link" })).toHaveAttribute(
      "href",
      "/forgot-password",
    );
  });
});
