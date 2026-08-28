import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import fonteDoApp from "../App.tsx?raw";
import { autenticar } from "../services/authService";
import { guardarSessao } from "../services/sessao";
import { addCachedVideo } from "../services/videoCache";
import { getProcessedVideo, getVideoStatus } from "../services/videoService";

vi.mock("../services/videoService", async (original) => ({
  ...(await original<typeof import("../services/videoService")>()),
  getVideoStatus: vi.fn(),
  getProcessedVideo: vi.fn(),
  getRenderVideo: vi.fn(),
}));

vi.mock("../services/authService", () => ({
  autenticar: vi.fn(),
  registrar: vi.fn(),
}));

function abrir(caminho: string) {
  window.history.pushState({}, "", caminho);
  render(<App />);
}

function comSessao() {
  guardarSessao("token", 3600, 1);
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(getVideoStatus).mockClear();
  vi.mocked(getVideoStatus).mockResolvedValue({ video_status: "Finalizado", motivo: null });
  vi.mocked(getProcessedVideo).mockReset();
  vi.mocked(autenticar).mockReset();
});

describe("App", () => {
  it("sem sessao a arvore inteira e o login, e o cabecalho nao oferece sair", () => {
    abrir("/");

    expect(screen.getByRole("heading", { name: "Entrar" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sair" })).toBeNull();
  });

  it("o login substitui a rota sem trocar o endereco, e por isso ele volta depois de entrar", () => {
    abrir("/results/3");

    expect(screen.getByRole("heading", { name: "Entrar" })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/results/3");
  });

  it("com sessao valida abre a home e lista o que ja foi enviado", async () => {
    comSessao();
    addCachedVideo({ id: 3, name: "treino.mp4" });

    abrir("/");

    expect(await screen.findByText("Processamentos")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /treino\.mp4/ })).toHaveAttribute("href", "/results/3");
  });

  it("o cabecalho leva de volta ao inicio quando ha sessao", () => {
    comSessao();

    abrir("/");

    expect(screen.getByRole("link", { name: /ir para o início/ })).toHaveAttribute("href", "/");
  });

  it("sair derruba a sessao e devolve o login", async () => {
    comSessao();
    abrir("/");

    await userEvent.click(screen.getByRole("button", { name: "Sair" }));

    expect(screen.getByRole("heading", { name: "Entrar" })).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("endereco desconhecido cai na tela de nao encontrada, com saida para o inicio", () => {
    comSessao();

    abrir("/rota-que-nao-existe");

    expect(screen.getByRole("heading", { name: "Página não encontrada" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Voltar ao início" })).toHaveAttribute("href", "/");
  });

  it("cada tela nomeia a aba, que e o que a WCAG 2.4.2 cobra numa aplicacao de uma pagina so", async () => {
    abrir("/");
    expect(document.title).toBe("Entrar | XVision");

    comSessao();
    abrir("/");
    expect(document.title).toBe("Enviar vídeo | XVision");

    abrir("/rota-que-nao-existe");
    expect(document.title).toBe("Página não encontrada | XVision");
  });

  it("oferece pular o cabecalho e anuncia a mudanca de status sem recarregar a tela", async () => {
    comSessao();
    addCachedVideo({ id: 3, name: "treino.mp4" });

    abrir("/");

    const pular = screen.getByRole("link", { name: "Pular para o conteúdo" });
    expect(pular).toHaveAttribute("href", "#conteudo");
    expect(document.querySelector("main")).toHaveAttribute("id", "conteudo");

    await screen.findByText("Processamentos");
    expect(screen.getByText("Finalizado").closest("[aria-live]")).toHaveAttribute(
      "aria-live",
      "polite",
    );
  });

  it("o provedor de videos e montado dentro do ramo autenticado, e nao em volta dele", () => {
    const login = fonteDoApp.indexOf("<Login");
    const provedor = fonteDoApp.indexOf("<VideoProvider>");

    expect(login).toBeGreaterThan(-1);
    expect(provedor, "o provedor subiu para fora do ramo autenticado").toBeGreaterThan(login);
    expect(fonteDoApp.split("<VideoProvider>").length - 1).toBe(1);
  });

  it("sessao vencida volta ao login e nao consulta status nenhum", () => {
    comSessao();
    addCachedVideo({ id: 3, name: "treino.mp4" });
    guardarSessao("token", -1, 1);

    abrir("/");

    expect(screen.getByRole("heading", { name: "Entrar" })).toBeInTheDocument();
    expect(screen.queryByText("Processamentos")).toBeNull();
    expect(getVideoStatus).not.toHaveBeenCalled();
  });
});
