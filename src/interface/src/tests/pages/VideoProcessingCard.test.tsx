import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import VideoProcessingCard from "../../pages/Home/components/VideoProcessingCard";
import type { VideoStatus } from "../../services/videoService";

const tentarDeNovo = vi.fn();
const remover = vi.fn();

function montar(status: VideoStatus | null, statusError = false) {
  tentarDeNovo.mockClear();
  remover.mockClear();

  render(
    <MemoryRouter>
      <VideoProcessingCard
        id={3}
        name="treino.mp4"
        status={status}
        statusError={statusError}
        onTentarDeNovo={tentarDeNovo}
        onRemover={remover}
      />
    </MemoryRouter>,
  );
}

describe("VideoProcessingCard", () => {
  it("vira um link quando a analise terminou, alcancavel pelo teclado", () => {
    montar("Finalizado");

    expect(screen.getByRole("link")).toHaveAttribute("href", "/results/3");
  });

  it("tambem leva ao resultado quando a analise falhou, que e onde mora o motivo", () => {
    montar("Falhou");

    expect(screen.getByRole("link")).toHaveAttribute("href", "/results/3");
  });

  it.each<VideoStatus>(["Pendente", "Em processamento"])(
    "nao oferece link enquanto esta em %s",
    (status) => {
      montar(status);

      expect(screen.queryByRole("link")).toBeNull();
      expect(screen.getByText(status)).toBeInTheDocument();
    },
  );

  it("diz que nao conseguiu consultar, em vez de mostrar um estado inventado", () => {
    montar(null, true);

    expect(screen.getByText("Não foi possível consultar o status")).toBeInTheDocument();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("oferece saida para o cartao que nao consegue consultar, em vez de deixa-lo preso", async () => {
    montar(null, true);

    await userEvent.click(screen.getByRole("button", { name: "Tentar de novo" }));
    expect(tentarDeNovo).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByRole("button", { name: "Remover da lista" }));
    expect(remover).toHaveBeenCalledTimes(1);
  });

  it("nao oferece essas acoes quando o status chegou", () => {
    montar("Em processamento");

    expect(screen.queryByRole("button")).toBeNull();
  });
});
