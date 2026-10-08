import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { VideoContext } from "../../contexts/videoContext";
import VideoProcessingList from "../../pages/Home/components/VideoProcessingList";
import { resumo } from "../fixtures";

const recarregar = vi.fn().mockResolvedValue(undefined);

function montar({
  videos = [resumo()],
  loading = false,
  error = null,
  total = videos.length,
}: {
  videos?: ReturnType<typeof resumo>[];
  loading?: boolean;
  error?: string | null;
  total?: number;
} = {}) {
  render(
    <MemoryRouter>
      <VideoContext.Provider
        value={{
          videos,
          total,
          statistics: { pending: 0, processing: 0, completed: videos.length, failed: 0 },
          loading,
          error,
          recarregar,
        }}
      >
        <VideoProcessingList />
      </VideoContext.Provider>
    </MemoryRouter>,
  );
}

describe("VideoProcessingList", () => {
  it("mostra um estado vazio que explica onde o primeiro item aparecera", () => {
    montar({ videos: [] });
    expect(screen.getByText("Nenhuma análise ainda")).toBeInTheDocument();
  });

  it("filtra por nome e por situacao", async () => {
    montar({ videos: [resumo(), resumo({ id: 4, name: "erro.mp4", videoStatus: "Falhou" })] });

    await userEvent.type(screen.getByRole("searchbox", { name: "Buscar por arquivo" }), "erro");
    expect(screen.getByText("erro.mp4")).toBeInTheDocument();
    expect(screen.queryByText("treino.mp4")).toBeNull();

    await userEvent.clear(screen.getByRole("searchbox", { name: "Buscar por arquivo" }));
    await userEvent.click(screen.getByRole("button", { name: "Finalizado" }));
    expect(screen.getByText("treino.mp4")).toBeInTheDocument();
    expect(screen.queryByText("erro.mp4")).toBeNull();
  });

  it("oferece nova tentativa quando a lista falha", async () => {
    montar({ videos: [], error: "sem conexão" });
    expect(screen.getByRole("alert")).toHaveTextContent("sem conexão");
    await userEvent.click(screen.getByRole("button", { name: "Tentar de novo" }));
    expect(recarregar).toHaveBeenCalled();
  });

  it("anuncia mudancas automaticas na lista", () => {
    montar();
    expect(screen.getByText("treino.mp4").closest("[aria-live]")).toHaveAttribute(
      "aria-live",
      "polite",
    );
  });

  it("avisa quando a atualizacao falha mas ainda ha itens na tela", () => {
    montar({ error: "sem conexão" });

    expect(screen.getByText("treino.mp4")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("sem conexão");
  });

  it("avisa que a lista mostra so as mais recentes quando a conta tem mais analises", () => {
    montar({ total: 130 });

    expect(screen.getByText(/Mostrando 1 de 130 análises/)).toBeInTheDocument();
  });

  it("nao avisa nada quando a lista mostra todas as analises", () => {
    montar();

    expect(screen.queryByText(/Mostrando/)).toBeNull();
  });

  it("agrupa os filtros para leitores de tela", () => {
    montar();

    expect(screen.getByRole("group", { name: "Filtrar análises" })).toBeInTheDocument();
  });
});
