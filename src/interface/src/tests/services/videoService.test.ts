import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  enviarVideo,
  getProcessedVideo,
  getRenderVideo,
  getVideos,
  getVideoStatus,
  terminou,
} from "../../services/videoService";
import { XMLHttpRequestMock } from "../xhrMock";

const fetchMock = vi.fn();

const RESULTADO = {
  status: 200,
  video_status: "Finalizado",
  nome_arquivo: "c.mp4",
  content_type: "video/mp4",
  criado_em: "2026-09-01T12:00:00Z",
  atualizado_em: "2026-09-01T12:01:00Z",
  resultados: [
    {
      nome_processo: "shot",
      status: true,
      situacao: "ok",
      motivo: null,
      duracao_ms: 41,
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
    {
      nome_processo: "ball",
      status: false,
      situacao: "failed",
      motivo: "bola ausente",
      duracao_ms: 12,
      frame: null,
      metricas: [],
    },
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
  XMLHttpRequestMock.reset();
  vi.stubGlobal("XMLHttpRequest", XMLHttpRequestMock);
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
    expect(video.name).toBe("c.mp4");
    expect(video.results.map((r) => r.name)).toEqual(["shot", "ball"]);
    expect(video.results[1].ok).toBe(false);
    expect(video.results[1].reason).toBe("bola ausente");
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
          situacao: "ok",
          motivo: null,
          duracao_ms: 1,
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
        {
          nome_processo: "ball",
          status: true,
          situacao: "ok",
          motivo: null,
          duracao_ms: 1,
          frame: "QUJD",
          metricas: [],
        },
        {
          nome_processo: "field",
          status: true,
          situacao: "ok",
          motivo: null,
          duracao_ms: 1,
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
      resultados: [
        {
          nome_processo: "ball",
          status: true,
          situacao: "ok",
          motivo: null,
          duracao_ms: 1,
          frame: null,
        },
      ],
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
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ id: 7, uploadUrl: "/v1/videos/upload/tok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(enviarVideo(arquivo)).resolves.toBe(7);

    expect(JSON.parse(String(chamada(0)[1].body))).toEqual({
      fileName: "c.mp4",
      contentType: "video/mp4",
    });
    const envio = XMLHttpRequestMock.instances[0];
    expect(envio.url).toBe("http://api.test/v1/videos/upload/tok");
    expect(envio.method).toBe("PUT");
    expect(envio.body).toBe(arquivo);
  });

  it("consulta o status no endereco que o servico publica", async () => {
    respondeJson({
      status: 200,
      video_status: "Em processamento",
      motivo: null,
      etapa_atual: "ball",
      etapas_concluidas: 3,
      etapas_total: 9,
      progresso: 33.3,
      posicao_fila: null,
      reconsultar_em_s: 2,
    });

    await expect(getVideoStatus(3)).resolves.toMatchObject({
      videoStatus: "Em processamento",
      currentStage: "ball",
      completedStages: 3,
      progress: 33.3,
    });
    expect(chamada()[0]).toBe("http://api.test/v1/videos/status/3");
  });

  it("lista os videos da conta e traduz progresso, fila e metadados", async () => {
    respondeJson({
      total: 1,
      pendentes: 1,
      em_processamento: 0,
      finalizadas: 0,
      falhas: 0,
      videos: [
        {
          id: 8,
          nome_arquivo: "fila.mp4",
          content_type: "video/mp4",
          criado_em: "2026-09-01T12:00:00Z",
          atualizado_em: "2026-09-01T12:00:01Z",
          video_status: "Pendente",
          motivo: null,
          etapa_atual: null,
          etapas_concluidas: 0,
          etapas_total: 9,
          progresso: 0,
          posicao_fila: 2,
          reconsultar_em_s: 2,
        },
      ],
    });

    await expect(getVideos()).resolves.toMatchObject({
      total: 1,
      statistics: { pending: 1, processing: 0, completed: 0, failed: 0 },
      videos: [{ id: 8, name: "fila.mp4", queuePosition: 2, totalStages: 9 }],
    });
    expect(chamada()[0]).toBe("http://api.test/v1/videos?limite=100&deslocamento=0");
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
