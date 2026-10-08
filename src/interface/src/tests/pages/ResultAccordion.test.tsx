import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ResultAccordion from "../../pages/Results/components/ResultAccordion";
import { RESULTADO_PRINCIPAL } from "../fixtures";

describe("ResultAccordion", () => {
  it("mostra o motivo da falha uma unica vez quando nao ha metricas", () => {
    render(
      <ResultAccordion
        principalMediu={false}
        result={{
          ...RESULTADO_PRINCIPAL,
          name: "ball",
          ok: false,
          state: "failed",
          reason: "a bola nao apareceu",
        }}
      />,
    );

    expect(screen.getAllByText("a bola nao apareceu")).toHaveLength(1);
  });

  it("explica que nao ha o que mostrar quando falhou sem motivo", () => {
    render(
      <ResultAccordion
        principalMediu={false}
        result={{ ...RESULTADO_PRINCIPAL, name: "ball", ok: false, state: "failed" }}
      />,
    );

    expect(screen.getByText(/não concluiu/)).toBeInTheDocument();
  });
});
