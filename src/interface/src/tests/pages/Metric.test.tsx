import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Metric from "../../pages/Results/components/Metric";
import type { VideoMetric } from "../../services/videoService";

function metrica(parcial: Partial<VideoMetric>): VideoMetric {
  return { name: "gol", unit: "", type: "float", value: 1, status: "ok", ...parcial };
}

describe("Metric", () => {
  it("mostra booleano em portugues, e nao 'true'", () => {
    render(<Metric metric={metrica({ type: "boolean", value: true })} />);

    expect(screen.getByText("Sim")).toBeInTheDocument();
    expect(screen.queryByText("true")).not.toBeInTheDocument();
  });

  it("mostra 'Não' para o booleano falso, que nao e ausencia de medida", () => {
    render(<Metric metric={metrica({ type: "boolean", value: false })} />);

    expect(screen.getByText("Não")).toBeInTheDocument();
    expect(screen.queryByText("indisponível")).not.toBeInTheDocument();
  });

  it("arredonda o float em vez de despejar as casas do modelo", () => {
    render(<Metric metric={metrica({ value: 12.3456789, unit: "m" })} />);

    expect(screen.getByText("12.35")).toBeInTheDocument();
  });

  it("marca a medida de baixa confianca, que a borda promete nao entregar como medida", () => {
    render(<Metric metric={metrica({ value: 3, status: "low_confidence" })} />);

    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText(/baixa confiança/i)).toBeInTheDocument();
  });

  it("nao marca baixa confianca no que veio medido", () => {
    render(<Metric metric={metrica({ value: 3, status: "ok" })} />);

    expect(screen.queryByText(/baixa confiança/i)).not.toBeInTheDocument();
  });

  it("diz indisponivel quando a feature nao pode ser medida", () => {
    render(<Metric metric={metrica({ value: null, status: "unavailable" })} />);

    expect(screen.getByText("indisponível")).toBeInTheDocument();
  });

  it("mostra a categoria medida, e nao 'indisponível'", () => {
    render(<Metric metric={metrica({ type: "text", value: "PERNA_ESQUERDA", unit: "" })} />);

    expect(screen.getByText("PERNA_ESQUERDA")).toBeInTheDocument();
    expect(screen.queryByText("indisponível")).toBeNull();
  });

  it("nao tenta arredondar texto", () => {
    expect(() =>
      render(<Metric metric={metrica({ type: "text", value: "CABECA" })} />),
    ).not.toThrow();
  });
});
