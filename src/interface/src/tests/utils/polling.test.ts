import { describe, expect, it } from "vitest";

import { segundosAteReconsultar } from "../../utils/polling";

describe("segundosAteReconsultar", () => {
  it("usa o intervalo padrao quando o servidor nao informa", () => {
    expect(segundosAteReconsultar(null)).toBe(3);
  });

  it("respeita o intervalo informado pelo servidor", () => {
    expect(segundosAteReconsultar(7)).toBe(7);
  });

  it.each([0, -2, Number.NaN])("nunca consulta em laco apertado com %s", (valor) => {
    expect(segundosAteReconsultar(valor)).toBe(1);
  });
});
