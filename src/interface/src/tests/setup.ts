import "@testing-library/jest-dom/vitest";

class MemoriaStorage implements Storage {
  private itens = new Map<string, string>();

  get length(): number {
    return this.itens.size;
  }

  key(indice: number): string | null {
    return [...this.itens.keys()][indice] ?? null;
  }

  getItem(chave: string): string | null {
    return this.itens.get(chave) ?? null;
  }

  setItem(chave: string, valor: string): void {
    this.itens.set(chave, String(valor));
  }

  removeItem(chave: string): void {
    this.itens.delete(chave);
  }

  clear(): void {
    this.itens.clear();
  }
}

Object.defineProperty(globalThis, "localStorage", {
  value: new MemoriaStorage(),
  configurable: true,
});

Object.defineProperty(globalThis, "scrollTo", {
  value: () => undefined,
  configurable: true,
  writable: true,
});

Object.defineProperty(globalThis, "requestAnimationFrame", {
  value: (callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(performance.now()), 0),
  configurable: true,
  writable: true,
});

Object.defineProperty(globalThis, "cancelAnimationFrame", {
  value: (id: number) => window.clearTimeout(id),
  configurable: true,
  writable: true,
});
