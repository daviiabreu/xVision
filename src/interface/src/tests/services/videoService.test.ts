import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  enviarVideo,
  getProcessedVideo,
  getRenderVideo,
  getVideoStatus,
  terminou,
} from "../../services/videoService";

const fetchMock = vi.fn();

// O corpo abaixo foi copiado de uma resposta real do servico, com a configuracao de exemplo que
// `src/tests/api/test_metricas.py` monta: uma feature que mede, uma sem leitura e uma categorica.
const RESULTADO = {
  status: 200,
  video_status: "Finalizado",
  resultados: [
    {
      nome_processo: "shot",
      status: true,
      frame: null,
      metricas: [
        {
          nome: "distancia_falsa",
          unidade_de_medida: "m",
          tipo: "float",
          valor: 11.5,
          status: "ok",
        },
        {
          nome: "sem_leitura",
          unidade_de_medida: "m",
          tipo: null,
          valor: null,
          status: "unavailable",
        },
        {
          nome: "parte_do_corpo",
          unidade_de_medida: "",
          tipo: "text",
          valor: "PERNA_ESQUERDA",
          status: "ok",
        },
      ],
    },
    { nome_processo: "ball", status: false, frame: null, metricas: [] },
  ],
  motivo: null,
  render_url: "/v1/videos/7/render",
};

function respondeJson(corpo: unknown, status = 200) {
  fetchMock.mockResolvedValue(
    new Response(JSON.stringify(corpo), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function chamada(indice = 0): [string, RequestInit] {
  return fetchMock.mock.calls[indice] as [string, RequestInit];
}

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

describe("videoService", () => {
  it("traduz o corpo do servico para os nomes que a tela usa", async () => {
    respondeJson(RESULTADO);

    const video = await getProcessedVideo(7);

    expect(chamada()[0]).toBe("http://api.test/v1/videos/7");
    expect(video.videoStatus).toBe("Finalizado");
    expect(video.reason).toBeNull();
    expect(video.renderUrl).toBe("/v1/videos/7/render");
    expect(video.results.map((r) => r.name)).toEqual(["shot", "ball"]);
    expect(video.results[1].ok).toBe(false);
  });

  it("traduz cada campo da metrica, inclusive o status que a borda promete", async () => {
    respondeJson(RESULTADO);

    const [medida, semLeitura, categoria] = (await getProcessedVideo(7)).results[0].metrics;

    expect(medida).toEqual({
      name: "distancia_falsa",
      unit: "m",
      type: "float",
      value: 11.5,
      status: "ok",
    });
    expect(semLeitura.value).toBeNull();
    expect(semLeitura.status).toBe("unavailable");
    expect(categoria.value).toBe("PERNA_ESQUERDA");
    expect(categoria.type).toBe("text");
  });

  it("aceita tipo nulo, que e o que o servico manda quando nao houve leitura", async () => {
    respondeJson({
      ...RESULTADO,
      resultados: [
        {
          nome_processo: "shot",
          status: true,
          frame: null,
          metricas: [
            {
              nome: "sem_leitura",
              unidade_de_medida: "m",
              tipo: null,
              valor: null,
              status: "unavailable",
            },
          ],
        },
      ],
    });

    const [metrica] = (await getProcessedVideo(7)).results[0].metrics;

    expect(metrica.type).toBeNull();
    expect(metrica.value).toBeNull();
  });

  it("traduz o quadro de demonstracao como veio, sem inventar prefixo", async () => {
    respondeJson({
      ...RESULTADO,
      resultados: [
        { nome_processo: "ball", status: true, frame: "QUJD", metricas: [] },
        {
          nome_processo: "field",
          status: true,
          frame: "data:image/webp;base64,QUJD",
          metricas: [],
        },
      ],
    });

    const etapas = (await getProcessedVideo(7)).results;

    expect(etapas[0].frame).toBe("QUJD");
    expect(etapas[1].frame).toBe("data:image/webp;base64,QUJD");
  });

  it("aguenta etapa sem a lista de metricas, que o contrato declara com valor padrao", async () => {
    respondeJson({
      ...RESULTADO,
      resultados: [{ nome_processo: "ball", status: true, frame: null }],
    });

    const [etapa] = (await getProcessedVideo(7)).results;

    expect(etapa.metrics).toEqual([]);
  });

  it("traduz o corpo que o servico devolve para um video que falhou", async () => {
    respondeJson({
      status: 200,
      video_status: "Falhou",
      resultados: [],
      motivo: "o arquivo enviado nao abriu como video",
      render_url: null,
    });

    const video = await getProcessedVideo(7);

    expect(video.videoStatus).toBe("Falhou");
    expect(video.results).toEqual([]);
    expect(video.reason).toBe("o arquivo enviado nao abriu como video");
    expect(video.renderUrl).toBeNull();
  });

  it("aguenta corpo sem a lista de etapas, que e defesa e nao contrato", async () => {
    respondeJson({ status: 200, video_status: "Falhou", motivo: "o clipe nao abriu" });

    expect((await getProcessedVideo(7)).results).toEqual([]);
  });

  it("reserva o endereco e manda o binario cru para ele, sem passar por base64", async () => {
    const arquivo = new File(["conteudo"], "c.mp4", { type: "video/mp4" });
    fetchMock
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ id: 7, uploadUrl: "/v1/videos/upload/tok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: 202, id: 7 }), { status: 202 }));

    await expect(enviarVideo(arquivo)).resolves.toBe(7);

    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      fileName: "c.mp4",
      contentType: "video/mp4",
    });
    expect(chamada(1)[0]).toBe("http://api.test/v1/videos/upload/tok");
    expect(chamada(1)[1].method).toBe("PUT");
    expect(chamada(1)[1].body).toBe(arquivo);
  });

  it("consulta o status no endereco que o servico publica", async () => {
    respondeJson({ status: 200, video_status: "Em processamento", motivo: null });

    await expect(getVideoStatus(3)).resolves.toMatchObject({
      video_status: "Em processamento",
      motivo: null,
    });
    expect(chamada()[0]).toBe("http://api.test/v1/videos/status/3");
  });

  it("devolve null quando o servico nao tem render, sem derrubar quem chamou", async () => {
    respondeJson({ detail: "sem render" }, 404);

    await expect(getRenderVideo("/v1/videos/7/render")).resolves.toBeNull();
  });

  it.each([
    ["Finalizado", true],
    ["Falhou", true],
    ["Pendente", false],
    ["Em processamento", false],
  ] as const)("terminou(%s) e %s", (status, esperado) => {
    expect(terminou(status)).toBe(esperado);
  });

  it("terminou(null) e falso: nao saber o estado nao e ter terminado", () => {
    expect(terminou(null)).toBe(false);
  });
});
