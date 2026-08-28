import { beforeEach, describe, expect, it } from "vitest";

import { guardarSessao } from "../../services/sessao";
import {
  addCachedVideo,
  getCachedVideo,
  getCachedVideos,
  removeCachedVideo,
} from "../../services/videoCache";

beforeEach(() => localStorage.clear());

describe("videoCache", () => {
  it("nao mostra para uma conta os videos da outra", () => {
    guardarSessao("t", 3600, 1);
    addCachedVideo({ id: 10, name: "treino-a.mp4" });

    guardarSessao("t", 3600, 2);

    expect(getCachedVideos()).toEqual([]);
  });

  it("devolve os videos da conta quando ela volta", () => {
    guardarSessao("t", 3600, 1);
    addCachedVideo({ id: 10, name: "treino-a.mp4" });
    guardarSessao("t", 3600, 2);
    addCachedVideo({ id: 20, name: "treino-b.mp4" });

    guardarSessao("t", 3600, 1);

    expect(getCachedVideos()).toEqual([{ id: 10, name: "treino-a.mp4" }]);
  });

  it("sem conta conhecida nao le nem escreve, em vez de criar um balde compartilhado", () => {
    guardarSessao("t", 3600);

    addCachedVideo({ id: 10, name: "treino-a.mp4" });

    expect(getCachedVideos()).toEqual([]);
    expect(localStorage.getItem("videos:anonimo")).toBeNull();
    expect(localStorage.getItem("videos:null")).toBeNull();
  });

  it("devolve lista vazia com storage corrompido, em vez de estourar no render", () => {
    guardarSessao("t", 3600, 1);
    localStorage.setItem("videos:1", "{nao e json");

    expect(() => getCachedVideos()).not.toThrow();
    expect(getCachedVideos()).toEqual([]);
  });

  it("descarta entrada com forma errada", () => {
    guardarSessao("t", 3600, 1);
    localStorage.setItem("videos:1", JSON.stringify([{ id: "x" }, { id: 3, name: "ok.mp4" }]));

    expect(getCachedVideos()).toEqual([{ id: 3, name: "ok.mp4" }]);
  });

  it("acha um video pelo identificador", () => {
    guardarSessao("t", 3600, 1);
    addCachedVideo({ id: 5, name: "c.mp4" });

    expect(getCachedVideo(5)?.name).toBe("c.mp4");
  });

  it("tira da lista o video que o servico nao conhece mais, sem levar os outros junto", () => {
    guardarSessao("t", 3600, 1);
    addCachedVideo({ id: 5, name: "c.mp4" });
    addCachedVideo({ id: 6, name: "d.mp4" });

    removeCachedVideo(5);

    expect(getCachedVideos()).toEqual([{ id: 6, name: "d.mp4" }]);
  });
});
