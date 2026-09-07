import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import fonteDoApp from "../App.tsx?raw";
import { autenticar, buscarEstatisticasConta } from "../services/authService";
import { guardarSessao } from "../services/sessao";
import { getProcessedVideo, getVideos } from "../services/videoService";
import { resumo } from "./fixtures";

const ZERO_STATISTICS = { pending: 0, processing: 0, completed: 0, failed: 0 };

vi.mock("../services/videoService", async (original) => ({
  ...(await original<typeof import("../services/videoService")>()),
  getVideos: vi.fn(),
  getVideoStatus: vi.fn(),
  getProcessedVideo: vi.fn(),
  getRenderVideo: vi.fn(),
}));

vi.mock("../services/authService", () => ({
  autenticar: vi.fn(),
  registrar: vi.fn(),
  solicitarRedefinicao: vi.fn(),
  redefinirSenha: vi.fn(),
  atualizarEmail: vi.fn(),
  alterarSenha: vi.fn(),
  buscarEstatisticasConta: vi.fn(),
}));

function abrir(caminho: string) {
  window.history.pushState({}, "", caminho);
  render(<App />);
}

function comSessao() {
  guardarSessao("token", 3600, 1, "analista@exemplo.com");
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(getVideos).mockReset();
  vi.mocked(getVideos).mockResolvedValue({ videos: [], total: 0, statistics: ZERO_STATISTICS });
  vi.mocked(getProcessedVideo).mockReset();
  vi.mocked(autenticar).mockReset();
  vi.mocked(buscarEstatisticasConta).mockReset();
  vi.mocked(buscarEstatisticasConta).mockResolvedValue({
    total: 0,
    pendentes: 0,
    em_processamento: 0,
    finalizadas: 0,
    falhas: 0,
  });
});

describe("App", () => {
  it("sem sessao mostra o login e nao oferece sair", () => {
    abrir("/");

    expect(screen.getByRole("heading", { name: "Bem-vindo de volta" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /sair/i })).toBeNull();
  });

  it("preserva o endereco solicitado durante o login", () => {
    abrir("/results/3");

    expect(screen.getByRole("heading", { name: "Bem-vindo de volta" })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/results/3");
  });

  it("deixa a documentacao da API publica e incorporada", () => {
    abrir("/api-docs");

    expect(
      screen.getByRole("heading", { name: "Automatize o envio de vídeos" }),
    ).toBeInTheDocument();
    expect(screen.getByTitle("Referência interativa da API XVision")).toHaveAttribute(
      "src",
      "http://api.test/docs",
    );
    expect(screen.getByRole("link", { name: "Baixar coleção Bruno" })).toHaveAttribute(
      "href",
      "http://api.test/bruno.zip",
    );
    const exemplo = document.querySelector("pre")?.textContent ?? "";
    expect(exemplo).toContain("$BASE_URL$UPLOAD_URL");
    expect(exemplo).toContain("VIDEO_ID=$(printf");
  });

  it("deixa recuperacao e definicao de nova senha publicas", () => {
    abrir("/forgot-password");
    expect(screen.getByRole("heading", { name: "Redefina sua senha" })).toBeInTheDocument();

    abrir("/reset-password?token=token-longo-de-recuperacao");
    expect(screen.getByRole("heading", { name: "Crie uma nova senha" })).toBeInTheDocument();
  });

  it("com sessao valida carrega a lista que vem do servidor", async () => {
    comSessao();
    vi.mocked(getVideos).mockResolvedValue({
      videos: [resumo()],
      total: 1,
      statistics: { ...ZERO_STATISTICS, completed: 1 },
    });

    abrir("/");

    expect(await screen.findByText("treino.mp4")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /treino\.mp4/ })).toHaveAttribute("href", "/results/3");
  });

  it("a marca leva de volta para as analises quando ha sessao", () => {
    comSessao();
    abrir("/");

    expect(screen.getAllByRole("link", { name: /ir para análises/ })[0]).toHaveAttribute(
      "href",
      "/",
    );
  });

  it("leva para as configuracoes pelo fim da barra lateral", () => {
    comSessao();
    abrir("/account");

    expect(screen.getByRole("heading", { name: "Configurações da conta" })).toBeInTheDocument();
    for (const link of screen.getAllByRole("link", { name: "Configurações da conta" })) {
      expect(link).toHaveAttribute("href", "/account");
    }
  });

  it("sair derruba a sessao e devolve o login", async () => {
    comSessao();
    abrir("/");

    await userEvent.click(screen.getAllByRole("button", { name: /sair/i })[0]);

    expect(screen.getByRole("heading", { name: "Bem-vindo de volta" })).toBeInTheDocument();
    expect(localStorage.getItem("sessao")).toBeNull();
  });

  it("endereco desconhecido cai na tela de nao encontrada", () => {
    comSessao();
    abrir("/rota-que-nao-existe");

    expect(screen.getByRole("heading", { name: "Página não encontrada" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Voltar ao início" })).toHaveAttribute("href", "/");
  });

  it("cada tela nomeia a aba", () => {
    abrir("/");
    expect(document.title).toBe("Entrar | XVision");

    comSessao();
    abrir("/");
    expect(document.title).toBe("Análises | XVision");

    abrir("/rota-que-nao-existe");
    expect(document.title).toBe("Página não encontrada | XVision");
  });

  it("oferece pular a navegacao e mantem o conteudo principal focalizavel", () => {
    comSessao();
    abrir("/");

    expect(screen.getByRole("link", { name: "Pular para o conteúdo" })).toHaveAttribute(
      "href",
      "#conteudo",
    );
    expect(document.querySelector("main")).toHaveAttribute("id", "conteudo");
    expect(document.querySelector("main")).toHaveAttribute("tabindex", "-1");
  });

  it("recolhe a navegacao lateral e preserva a preferencia", async () => {
    comSessao();
    abrir("/");

    await userEvent.click(screen.getByRole("button", { name: "Recolher menu lateral" }));

    expect(screen.getByRole("button", { name: "Expandir menu lateral" })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(localStorage.getItem("xvision:sidebar-collapsed")).toBe("true");
  });

  it("abre e fecha a navegacao mobile com botao e teclado", async () => {
    comSessao();
    abrir("/");

    await userEvent.click(screen.getByRole("button", { name: "Menu" }));
    expect(screen.getByRole("dialog", { name: "Menu lateral" })).toHaveAttribute(
      "aria-hidden",
      "false",
    );
    const dialog = screen.getByRole("dialog", { name: "Menu lateral" });
    const fechar = within(dialog).getByRole("button", { name: "Fechar menu" });
    const sair = within(dialog).getByRole("button", { name: "Sair da conta" });
    const inicio = within(dialog).getByRole("link", { name: "XVision, ir para análises" });
    expect(fechar).toHaveFocus();

    await userEvent.tab({ shift: true });
    expect(inicio).toHaveFocus();
    await userEvent.tab({ shift: true });
    expect(sair).toHaveFocus();
    await userEvent.tab();
    expect(inicio).toHaveFocus();

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Menu lateral" })).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Menu" }));
    await userEvent.click(screen.getByRole("button", { name: "Fechar menu" }));
    expect(screen.queryByRole("dialog", { name: "Menu lateral" })).toBeNull();
  });

  it("monta o provedor de videos somente no ramo autenticado", () => {
    const condicao = fonteDoApp.indexOf("{autenticado ? (");
    const login = fonteDoApp.indexOf("<Login");
    const provedor = fonteDoApp.indexOf("<VideoProvider>");

    expect(login).toBeGreaterThan(-1);
    expect(provedor).toBeGreaterThan(condicao);
    expect(login).toBeGreaterThan(provedor);
    expect(fonteDoApp.split("<VideoProvider>").length - 1).toBe(1);
  });

  it("sessao vencida volta ao login e nao consulta videos", () => {
    guardarSessao("token", -1, 1);
    abrir("/");

    expect(screen.getByRole("heading", { name: "Bem-vindo de volta" })).toBeInTheDocument();
    expect(getVideos).not.toHaveBeenCalled();
  });
});
