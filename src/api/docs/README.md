# API

Duas formas de ver o contrato, e as duas saem do mesmo lugar.

**Swagger**, gerado pelo código. Suba o serviço e abra `/docs`:

```bash
uv run --extra api uvicorn api:create_app --factory --reload
```

**Bruno**, para bater na API sem escrever cliente. Abra a coleção `src/api/docs/xvision/` e
selecione o ambiente `local`. As requisições estão numeradas na ordem em que se
encadeiam: a primeira guarda `uploadUrl` e `videoId` para as seguintes. Gere a
fixture antes com `uv run xvision clip`.

## O fluxo

O processamento é assíncrono porque um clipe leva segundos e um proxy corta a conexão
antes disso. Quem envia recebe **HTTP 202** com um identificador na hora e consulta o andamento
depois. O corpo repete o código, como o contrato do frontend pede, e os dois sempre batem.

`GET /v1/videos/{id}` responde 200 nos dois estados terminais: com `resultados` quando
`Finalizado`, e com `motivo` quando `Falhou`. Ele só recusa com 409 enquanto o vídeo de fato
não terminou. Devolver 409 para `Falhou` dizendo "ainda não terminou" seria falso duas vezes,
e mandaria o frontend consultar para sempre um estado que nunca muda.

| Rota | O que faz |
| --- | --- |
| `POST /v1/videos/upload-url` | reserva o identificador e devolve o endereço de envio |
| `PUT /v1/videos/upload/{token}` | recebe o binário cru e enfileira o processamento |
| `POST /v1/videos/processamento` | mesmo efeito, com o vídeo em base64 dentro do JSON |
| `GET /v1/videos/status/{id}` | responde um dos quatro valores do enum do frontend |
| `GET /v1/videos/{id}` | devolve uma linha por etapa, ou o motivo, quando terminou |
| `GET /health` | diz se o serviço subiu |
| `GET /` | lista as versões que o serviço responde |
| `POST /v1/auth/registro` | cria uma conta de usuário |
| `POST /v1/auth/login` | troca email e senha por uma sessão |
| `GET /v1/auth/eu` | diz quem o serviço acha que você é |

## Autenticação

Duas credenciais entram e uma coisa sai: um **principal**. O resto do serviço nunca pergunta se
veio sessão de usuário ou token de serviço, e é isso que impede a regra de posse de ter dois
jeitos de estar errada.

| Credencial | Para quem | Como se obtém |
| --- | --- | --- |
| Token de serviço | outro sistema (a borda, um pipeline) | `python -m api.tokens criar <nome>` |
| Sessão de usuário | pessoa, pelo frontend | `POST /v1/auth/login` |

Ambas viajam como `Authorization: Bearer <credencial>`.

**Chave de API compartilhada não existe, e a ausência é deliberada.** Uma chave é um token de
serviço que não está no banco: não dá para revogar sem redeploy, e não responde "quem mandou esse
vídeo". A comodidade de pôr um segredo na variável continua: se `XVISION_API_TOKEN` estiver
definida, o serviço semeia um token com ela na subida. Mas ele nasce na tabela, revogável e
atribuível, e semear é idempotente. O valor tem mínimo de 32 caracteres, porque ele vira
credencial exatamente como está escrito: aceitar `x` seria ter senha fraca com nome de token.
Variável em branco significa "não quero semente", e não derruba a subida.

**Todo vídeo pertence a quem o enviou, e ler exige ser o dono.** Sem isso, com identificador
sequencial, proteger só o envio deixaria o resultado público bastando contar de 1 em diante.
Vídeo de outro responde **404, e não 403**: com 403, o id vira um oráculo que diz quantos vídeos
existem e quando cada um entrou, sem ler nenhum.

**`PUT /v1/videos/upload/{token}` não exige Bearer**, e isso é decisão. O endereço de envio já é
uma credencial: 32 bytes sorteados, válido uma vez, preso a um vídeo, reivindicado por `UPDATE`
condicional. É o desenho de uma URL pré-assinada, que é o que `videoService.ts` já implementa.
Quem tem o endereço já passou por `upload-url`, que exige credencial.

O token de sessão é assinado com HMAC-SHA256 e **não é um JWT**. JWT carrega o algoritmo dentro
do próprio token, e a família de erros mais conhecida da especificação nasce daí: aceitar `none`,
confundir HS256 com RS256. Aqui os dois lados são o mesmo processo e o algoritmo é fixo, então
negociar algoritmo só acrescentaria superfície. Senha usa `hashlib.scrypt`, da biblioteca padrão,
com sal por senha: nenhuma dependência nova, e um vazamento do banco não entrega as senhas.

## Versionamento

O contrato mora sob `/v1`. Rota de operação fica fora dele: um orquestrador que consulta
`/health` não deveria precisar saber em que versão o contrato está, e publicar a `/v2` não
pode obrigar toda implantação a reconfigurar o healthcheck.

O prefixo vem de um `APIRouter`, não repetido em cada caminho, então publicar a `v2` é criar
um router novo e manter o antigo respondendo, em vez de editar seis strings e descobrir em
produção qual ficou para trás. Um teste recusa rota de contrato registrada fora do router, e
outro recusa a versão do documento discordar do prefixo da URL.

**Para o frontend isso não é mudança de código.** O `videoService.ts` monta as chamadas como
`${VITE_API_URL}/videos/upload-url`, então basta a variável de ambiente apontar para
`http://host/v1`. O endereço do `PUT` já vem pronto na resposta, com a versão dentro.

Quem chamar sem versão recebe 404, e `GET /` diz para onde ir, porque chamada sem versão é o
erro mais provável de quem integra.

Um endereço de envio serve uma vez só. A posse dele sai de um `UPDATE` condicional, e não de
ler o estado e depois gravar: medido, dois `PUT` no mesmo token passavam os dois, e com a fila
ocupada isso acontecia sempre, mesmo sem simultaneidade, porque a janela era a fila inteira.
Bastava um cliente repetir depois de um timeout de rede, e o segundo envio apagava o arquivo que
o worker do primeiro estava lendo. Envio interrompido no meio deixa o vídeo em `Falhou`; para
tentar de novo, peça outro endereço.

O envio tem dois caminhos porque o time do frontend especificou dois. O recomendado é
o `PUT`, que é o que `src/interface/src/services/videoService.ts` já chama e o que
transmite o arquivo direto para o disco. O base64 existe para atender o contrato
escrito e cobra o próprio preço: infla 33%, e o corpo inteiro precisa existir em
memória antes de virar bytes. Por isso o teto dele é de 32 MB, contra 128 MB do `PUT`,
que é o mesmo número que `fileValidation.ts` cobra no navegador.

O campo de tipo aceita `type` e `contentType`, os dois nomes que aparecem nas
especificações que chegaram, para a divergência não virar erro de integração.

## Que arquivo é aceito

Qualquer `video/*` passa na porta, e quem decide de verdade é o decodificador: logo depois de
receber os bytes, o serviço abre o arquivo e lê o cabeçalho. Custa milissegundos e a recusa
chega na própria resposta, não meia hora depois no status.

Lista fixa de container erra nas duas direções. Erra recusando: o decodificador desta imagem
abre `.mov`, `.avi`, `.mkv` e o `.mov` em HEVC que o iPhone grava, e uma lista com `video/mp4`
devolveria 415 para todos eles, incluindo o formato mais provável de um envio pelo celular. E
erra aceitando, porque `contentType` é afirmação do cliente: um `.txt` renomeado para `.mp4`
chega declarando `video/mp4` e passaria pela lista.

A desvantagem está nomeada: a prova é feita depois do upload, então um arquivo de 128 MB que
não é vídeo consome a banda antes de ser recusado. Evitar isso exigiria checar assinatura nos
primeiros bytes durante o streaming, o que resolveria o arquivo obviamente inválido e não o
vídeo corrompido no meio, que é o caso que mais aparece.

## Configuração

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `XVISION_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | origens que o navegador pode usar |
| `XVISION_API_MAX_UPLOAD_MB` | `128` | teto do `PUT` |
| `XVISION_API_MAX_JSON_MB` | `32` | teto do base64 |
| `XVISION_API_WORKERS` | `1` | clipes processados ao mesmo tempo |
| `XVISION_VAR_DIR` | `var` | onde ficam o banco e os arquivos enviados |
| `XVISION_CONFIG_DIR` | `config` | onde o serviço procura `pipeline.toml` |
| `XVISION_API_MAX_FILA` | `32` | trabalhos aceitos esperando a vez |
| `XVISION_LOG_LEVEL` | `INFO` | nível do log |
| `XVISION_AUTH_SECRET` | **obrigatória** | assina a sessão; sem ela o serviço não sobe |
| `XVISION_AUTH_TTL_S` | `28800` | validade da sessão, em segundos |
| `XVISION_API_TOKEN` | ausente | se definida, semeia um token de serviço na subida (mínimo 32 caracteres) |

O serviço **recusa a subir** com configuração quebrada, em vez de subir verde e falhar em toda
requisição. Um `XVISION_API_MAX_UPLOAD_MB=128MB`, com o sufixo a mais, passava por toda sonda de
prontidão e só aparecia no primeiro envio, como 500.

Quando a fila enche, o envio responde **503 com `Retry-After`**. Aceitar trabalho que não se
consegue fazer é pior que recusá-lo: quem chamou fica esperando um resultado que nunca vem.

Queda no meio do processamento é recuperada na subida seguinte: se o arquivo enviado sobreviveu,
o clipe é refeito; se não, o vídeo vai para `Falhou` com motivo. Trocar o nome do estado não é
recuperar o trabalho.

A fila é de um por padrão, e isso é decisão e não descuido. Memória é o recurso escasso
na borda: cada pipeline abre o próprio decodificador, e o pool de threads do Starlette
aceitaria quarenta chamadas bloqueantes ao mesmo tempo. Fila de um troca pico de memória
por espera, que é a troca certa aqui.

## O estado

O andamento mora em SQLite, em `var/api/jobs.db`, e não em Postgres. A necessidade
presente é uma requisição escrever e outra ler, num processo só, sem implantação. O
preço está nomeado: um arquivo não se replica, então no dia em que o serviço rodar em
mais de uma instância isto vira Postgres. A troca fica contida porque nenhum chamador
conhece SQL, só as funções de `src/api/store.py`.

O esquema é versionado por `PRAGMA user_version`, e `MIGRACOES` é uma tupla em que o índice é a
versão: acrescentar uma migração é acrescentar uma string no fim. `CREATE TABLE IF NOT EXISTS`
sozinho cria banco novo e não conserta banco velho, e sem versão a coluna nova só faltaria em
produção, nunca no teste de quem a escreveu. Migração já aplicada nunca é editada: editar uma
deixa dois bancos com o mesmo `user_version` e esquemas diferentes, que é pior que não versionar.

Não usa Alembic porque Alembic traz SQLAlchemy junto, e aqui não há ORM nem chamador que escreva
SQL: seriam duas dependências para versionar uma tabela.

## O contrato versionado

O `src/api/docs/openapi.json` é a cópia versionada, para a revisão ver o diff e para quem
escreve o frontend gerar cliente sem subir nada. Ele é gerado, nunca editado à mão:

```bash
uv run python -m api.export
```

Há teste que falha quando o arquivo versionado e o que o código gera divergem, e outro
que falha quando uma rota recusa com um código que o contrato não documenta.

## Ainda não implementado

O campo `frame` de cada resultado responde `null`: a imagem de demonstração depende da
visualização, que não entra nesta sprint. E `metricas` só é preenchida quando houver
feature declarada em `config/pipeline.toml`, o que acontece quando os modelos entrarem.
Até lá a lista vem vazia, e vem vazia de propósito, não por falha.
