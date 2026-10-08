import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import Layout from "../../components/layout/Layout";
import { SessaoProvider } from "../../contexts/SessaoProvider";
import { guardarSessao } from "../../services/sessao";

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("Layout", () => {
  it("abre mesmo quando o navegador bloqueia a preferencia do menu lateral", () => {
    guardarSessao("t", 3600, 1, "analista@exemplo.com");
    const lerOriginal = localStorage.getItem.bind(localStorage);
    vi.spyOn(localStorage, "getItem").mockImplementation((chave: string) => {
      if (chave === "xvision:sidebar-collapsed") throw new Error("bloqueado");
      return lerOriginal(chave);
    });

    render(
      <MemoryRouter>
        <SessaoProvider>
          <Layout>
            <p>conteudo</p>
          </Layout>
        </SessaoProvider>
      </MemoryRouter>,
    );

    expect(screen.getByText("conteudo")).toBeInTheDocument();
  });
});
