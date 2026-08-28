import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VideoContext } from "../../contexts/videoContext";
import VideoProcessingList from "../../pages/Home/components/VideoProcessingList";

const recarregar = vi.fn().mockResolvedValue(undefined);
const remover = vi.fn();

const EM_ERRO = { id: 42, name: "treino.mp4", status: null, statusError: true };

function montar(videos = [EM_ERRO]) {
  render(
    <MemoryRouter>
      <VideoContext.Provider value={{ videos, recarregar, remover }}>
        <VideoProcessingList />
      </VideoContext.Provider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  recarregar.mockClear();
  remover.mockClear();
});

describe("VideoProcessingList", () => {
  it("nao ocupa espaco quando nada foi enviado", () => {
    montar([]);

    expect(screen.queryByText("Processamentos")).toBeNull();
  });

  it("liga o botao de tentar de novo ao provedor", async () => {
    montar();

    await userEvent.click(screen.getByRole("button", { name: "Tentar de novo" }));

    expect(recarregar).toHaveBeenCalledTimes(1);
  });

  it("remove o video que a pessoa descartou, e nao outro", async () => {
    montar();

    await userEvent.click(screen.getByRole("button", { name: "Remover da lista" }));

    expect(remover).toHaveBeenCalledWith(42);
  });

  it("anuncia a lista para quem usa leitor de tela, porque ela muda sozinha", () => {
    montar();

    expect(screen.getByText("treino.mp4").closest("[aria-live]")).toHaveAttribute(
      "aria-live",
      "polite",
    );
  });
});
