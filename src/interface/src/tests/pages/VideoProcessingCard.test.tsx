import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import VideoProcessingCard from "../../pages/Home/components/VideoProcessingCard";
import { resumo } from "../fixtures";

function montar(parcial = {}) {
  render(
    <MemoryRouter>
      <VideoProcessingCard video={resumo(parcial)} />
    </MemoryRouter>,
  );
}

describe("VideoProcessingCard", () => {
  it("leva ao resultado quando a analise terminou", () => {
    montar();
    expect(screen.getByRole("link", { name: /treino\.mp4/ })).toHaveAttribute("href", "/results/3");
  });

  it("tambem abre o resultado falho, onde mora o diagnostico", () => {
    montar({ videoStatus: "Falhou", reason: "bola ausente" });
    expect(screen.getByRole("link", { name: /treino\.mp4/ })).toHaveAttribute("href", "/results/3");
    expect(screen.getByText("bola ausente")).toBeInTheDocument();
  });

  it("mostra etapa e progresso sem oferecer link enquanto processa", () => {
    montar({ videoStatus: "Em processamento", currentStage: "ball", progress: 44 });

    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("Detecção da bola")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "44");
  });

  it("mostra a posicao na fila quando ainda esta pendente", () => {
    montar({ videoStatus: "Pendente", queuePosition: 2, progress: 0 });
    expect(screen.getByText("Posição 2 na fila")).toBeInTheDocument();
  });
});
