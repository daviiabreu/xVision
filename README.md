<!-- TODO: nome do projeto -->
# [Nome do Projeto]

<!-- TODO: nome do grupo -->
## [Nome do Grupo]

<!-- TODO: adicionar a logo em docs/public/readme/logo.png -->
<p align="center">
  <img src="docs/public/readme/logo.png" alt="Logo do projeto" width="600">
</p>

## Integrantes:

<div align="center">
  <table>
    <tr>
      <td align="center">
        <!-- TODO: LinkedIn e foto (docs/public/readme/Lucas Nepomuceno.png) -->
        <a href="#">
          <img src="docs/public/readme/Lucas Nepomuceno.png" style="border-radius: 10%; width: 150px;" alt="Lucas Nepomuceno"/><br>
          <sub><b>Lucas Nepomuceno dos Santos</b></sub>
        </a>
      </td>
      <td align="center">
        <!-- TODO: LinkedIn e foto (docs/public/readme/Yasmim Marly.png) -->
        <a href="#">
          <img src="docs/public/readme/Yasmim Marly.png" style="border-radius: 10%; width: 150px;" alt="Yasmim Marly"/><br>
          <sub><b>Yasmim Marly Passos</b></sub>
        </a>
      </td>
      <td align="center">
        <!-- TODO: LinkedIn -->
        <a href="#">
          <img src="docs/public/readme/Matheus Jorge.png" style="border-radius: 10%; width: 150px;" alt="Matheus Jorge"/><br>
          <sub><b>Matheus Jorge Rosa</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://www.linkedin.com/in/davi-abreu-da-silveira/">
          <img src="docs/public/readme/Davi Abreu.png" style="border-radius: 10%; width: 150px;" alt="Davi Abreu"/><br>
          <sub><b>Davi Abreu da Silveira</b></sub>
        </a>
      </td>
    </tr>
    <tr>
      <td align="center">
        <!-- TODO: LinkedIn e foto (docs/public/readme/Gabriel Henrique.png) -->
        <a href="#">
          <img src="docs/public/readme/Gabriel Henrique.png" style="border-radius: 10%; width: 150px;" alt="Gabriel Henrique"/><br>
          <sub><b>Gabriel Henrique Martins Alves</b></sub>
        </a>
      </td>
      <td align="center">
        <!-- TODO: LinkedIn e foto (docs/public/readme/Roberto Diniz.png) -->
        <a href="#">
          <img src="docs/public/readme/Roberto Diniz.png" style="border-radius: 10%; width: 150px;" alt="Roberto Diniz"/><br>
          <sub><b>Roberto Diniz Branco Filho</b></sub>
        </a>
      </td>
      <td align="center">
        <!-- TODO: LinkedIn e foto (docs/public/readme/Daniel Parente.png) -->
        <a href="#">
          <img src="docs/public/readme/Daniel Parente.png" style="border-radius: 10%; width: 150px;" alt="Daniel Parente"/><br>
          <sub><b>Daniel Parente Pessoa Dias</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://www.linkedin.com/in/danilo-martins-merlo/">
          <img src="docs/public/readme/Danilo Martins.png" style="border-radius: 10%; width: 150px;" alt="Danilo Martins"/><br>
          <sub><b>Danilo Martins Merlo</b></sub>
        </a>
      </td>
    </tr>
  </table>
</div>

## Professores:

### Orientador(a)

<!-- TODO: LinkedIn -->
- Rodrigo Mangoni Nicola

### Instrutores

- [Fabiana Naomi Iegawa](https://www.linkedin.com/in/fabiana-iegawa/)
<!-- TODO: LinkedIn -->
- Bruno Grandchamp Rodilha
- [Marcelo de Paula do Desterro Gorini](https://www.linkedin.com/in/marcelodesterro/)
- [Marcelo Luiz do Amaral Gonçalves](https://www.linkedin.com/in/marcelo-gon%C3%A7alves-phd/)

## Descrição

<!-- TODO: descrição do projeto -->

## Estrutura de Pastas

```text
.
├── .gitlab/            # templates de issue/MR e componentes de CI
├── data/               # dado de entrada; não é versionado, e o README de dentro diz o que vai ali
├── docs/               # site de documentação (Fumadocs/Next.js), publicado no GitLab Pages
│   ├── content/docs/   # conteúdo em MDX, uma pasta por sprint
│   └── public/         # imagens referenciadas na documentação
├── infra/              # Terraform da infraestrutura
├── releases/           # notas de release; um arquivo novo aqui gera tag + release no GitLab
└── src/                # raiz dos projetos Python e frontend
    ├── .python-version # versão do Python usada pelo uv
    ├── api/            # serviço HTTP que a interface consome
    ├── cli/            # o comando `xvision`
    ├── config/         # pipeline.toml, labels.toml e split.toml
    ├── contracts/      # o esquema do evento que a borda produz
    ├── edge/           # o pipeline de percepção que roda na borda
    │   └── participants/  # detecção, tracking, identidade, equipe e marcação
    ├── interface/      # a interface em React
    ├── mlops/          # avaliação e divisão de dataset
    ├── models/         # scripts de treino e manifestos versionados dos pesos
    ├── pyproject.toml  # pacote, dependências e ferramentas Python
    ├── tests/          # a suíte de testes
    └── uv.lock         # resolução reproduzível das dependências
```

`src/` é a raiz Python. `pyproject.toml`, `uv.lock`, `.python-version`, os pacotes, os testes, as
configurações e os manifestos ficam juntos ali. `src/interface/` continua sendo um projeto
independente, com seu próprio `package.json`.

## Execução do Projeto

### Documentação

A documentação completa do projeto é publicada no GitLab Pages:
<https://g02-a0b03e.pages.git.inteli.edu.br>

Para executar a documentação localmente:

```bash
git clone <URL do repositório>
cd docs
npm install
npm run dev
```

### Pipeline de borda

O pipeline roda pela CLI `xvision`, instalada junto com o projeto:

```bash
cd src
uv sync --all-extras
uv run --frozen --all-extras xvision --help
```

O vídeo do parceiro não vem no repositório e os pesos de modelo também não. Os clipes ficam em
`../data/footstats/`, os pesos em `../var/models/` endereçados pelo conteúdo, e os dois caminhos são
ignorados pelo git. `xvision data status` e `xvision artifact status` dizem o que falta e onde.

Para processar um clipe, ou o catálogo inteiro:

```bash
uv run --frozen --all-extras xvision run --clip <arquivo.mp4>
uv run --frozen --all-extras xvision run --all --config config/pipeline.models.toml
```

Cada execução grava um diretório em `../var/runs/` com os eventos em JSONL, o relatório e a procedência
(commit, configuração, host, versões dos módulos).

### Medição

Três comandos medem elos diferentes da cadeia, e é proposital que sejam três: o de ponta mede a
feature e não diz onde ela quebrou.

```bash
uv run --frozen --all-extras xvision eval <execução>          # features contra os rótulos do parceiro
uv run --frozen --all-extras xvision eval-ball --dobra tudo   # a bola no instante do contato
uv run --frozen --all-extras xvision eval-instant <execução>  # o instante contra o gabarito manual
uv run --frozen --all-extras xvision placar --run-dir <execução> --bola <json do eval-ball>
```

O `placar` junta os três numa tabela só. Ele existe porque uma mudança pode subir a métrica da
percepção e derrubar a feature, e com os números em terminais separados isso passa por vitória.

### Gabarito manual do instante

O instante do chute não sai da marcação do parceiro, cujo grão é de um segundo. Ele é anotado a mão:

```bash
uv run --frozen --all-extras xvision label scrub --annotator <nome>   # janela, quadro a quadro
uv run --frozen --all-extras xvision label sheet                      # folha em PNG, sem janela
uv run --frozen --all-extras xvision label stats                      # quanto já tem, e a concordância
```

O `scrub` precisa de janela e roda onde o `opencv` foi compilado com uma; o `sheet` escreve PNG e
não depende de nada. A amostra é estratificada por partida com semente fixa, então rodar de novo
continua de onde parou.

## Histórico de Lançamentos

### 0.5.0 - Sprint 5

### 0.4.0 - Sprint 4

### 0.3.0 - Sprint 3

### 0.2.0 - Sprint 2

### 0.1.0 - Sprint 1

## Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="#">[Nome do Projeto]</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.inteli.edu.br/">Inteli</a>, Lucas Nepomuceno dos Santos, Yasmim Marly Passos, Matheus Jorge Rosa, Davi Abreu da Silveira, Gabriel Henrique Martins Alves, Roberto Diniz Branco Filho, Daniel Parente Pessoa Dias, Danilo Martins Merlo é licenciado sob a <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
