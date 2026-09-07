import { Link } from "react-router-dom";

import { classesDeBotao } from "../../components/ui/classesDeBotao";
import Icon from "../../components/ui/Icon";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { apiPublicUrl } from "../../services/api";

const EXEMPLO = `# Dependências: curl e jq
# BASE_URL aponta para a origem do produto, por exemplo http://localhost:3000

# 1. Reserve o envio e leia os dois campos da resposta
RESERVA=$(curl -sS -X POST "$BASE_URL/v1/videos/upload-url" \\
  -H "Authorization: Bearer $XVISION_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"fileName":"finalizacao.mp4","contentType":"video/mp4"}')
VIDEO_ID=$(printf '%s' "$RESERVA" | jq -r '.id')
UPLOAD_URL=$(printf '%s' "$RESERVA" | jq -r '.uploadUrl')

# 2. uploadUrl é relativo à mesma origem
curl -X PUT "$BASE_URL$UPLOAD_URL" \\
  -H "Content-Type: video/mp4" \\
  --data-binary @finalizacao.mp4

# 3. Consulte o processamento e, ao final, o resultado
curl -H "Authorization: Bearer $XVISION_TOKEN" \\
  "$BASE_URL/v1/videos/status/$VIDEO_ID"
curl -H "Authorization: Bearer $XVISION_TOKEN" \\
  "$BASE_URL/v1/videos/$VIDEO_ID"`;

function ApiDocs() {
  useTituloDaPagina("Documentação da API");

  return (
    <div className="mx-auto w-full max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <DocsHeader />
      <IntegrationSteps />
      <WebhookNotice />
      <CurlExample />
      <InteractiveReference />
    </div>
  );
}

function DocsHeader() {
  return (
    <header className="max-w-3xl">
      <p className="text-sm font-semibold text-amber-700">Integração para desenvolvedores</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-stone-950 sm:text-4xl">
        Automatize o envio de vídeos
      </h1>
      <p className="mt-4 text-base leading-7 text-stone-600">
        Use a API REST para reservar o upload, enviar o vídeo e consultar status e resultado. Este é
        o mesmo fluxo usado pelo frontend, pronto para ser chamado por outro sistema.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <a
          href={apiPublicUrl("/docs")}
          className={classesDeBotao()}
          target="_blank"
          rel="noreferrer"
        >
          Abrir Swagger
          <Icon name="arrow-right" className="h-4 w-4" />
        </a>
        <a href={apiPublicUrl("/bruno.zip")} className={classesDeBotao("secundario")} download>
          Baixar coleção Bruno
        </a>
        <a
          href={apiPublicUrl("/openapi.json")}
          className={classesDeBotao("secundario")}
          target="_blank"
          rel="noreferrer"
        >
          OpenAPI JSON
        </a>
      </div>
    </header>
  );
}

function IntegrationSteps() {
  return (
    <section className="mt-10 grid gap-4 md:grid-cols-3" aria-label="Etapas da integração">
      {[
        [
          "1",
          "Receba uma credencial",
          "Peça ao operador um token de serviço. Ele pode ser revogado sem afetar contas de pessoas.",
        ],
        [
          "2",
          "Envie o arquivo",
          "Reserve uma URL, envie o MP4 como binário e guarde o identificador retornado.",
        ],
        [
          "3",
          "Consulte o resultado",
          "Consulte o status até a conclusão e então recupere métricas, etapas e evidências.",
        ],
      ].map(([numero, titulo, texto]) => (
        <article
          key={numero}
          className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-stone-950 text-sm font-bold text-white">
            {numero}
          </span>
          <h2 className="mt-4 text-lg font-bold text-stone-950">{titulo}</h2>
          <p className="mt-2 text-sm leading-6 text-stone-600">{texto}</p>
        </article>
      ))}
    </section>
  );
}

function WebhookNotice() {
  return (
    <section className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 p-5">
      <h2 className="font-bold text-amber-950">API hoje, webhook depois</h2>
      <p className="mt-2 text-sm leading-6 text-amber-900">
        Nesta versão, a integração consulta o status periodicamente. Ainda não existe webhook de
        conclusão. A documentação não promete esse contrato antes de ele estar implementado.
      </p>
    </section>
  );
}

function CurlExample() {
  return (
    <section className="mt-8 overflow-hidden rounded-2xl border border-stone-800 bg-stone-950">
      <div className="border-b border-stone-700 px-5 py-3 text-sm font-semibold text-stone-200">
        Exemplo completo com curl
      </div>
      <pre className="overflow-x-auto p-5 text-sm leading-6 text-stone-100">
        <code>{EXEMPLO}</code>
      </pre>
    </section>
  );
}

function InteractiveReference() {
  return (
    <section className="mt-10">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-amber-700">Referência interativa</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight text-stone-950">
            Teste os endpoints no Swagger
          </h2>
        </div>
        <Link
          to="/"
          className="text-sm font-semibold text-stone-600 underline underline-offset-4 hover:text-stone-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
        >
          Voltar ao produto
        </Link>
      </div>
      <iframe
        title="Referência interativa da API XVision"
        src={apiPublicUrl("/docs")}
        className="mt-5 h-[52rem] w-full rounded-2xl border border-stone-200 bg-white shadow-sm"
      />
    </section>
  );
}

export default ApiDocs;
