# Implementação e experimentos do TCC

Este documento complementa o [README.md](./README.md), que descreve a reprodução do IQLoc. O foco aqui é separar o fluxo de referência do projeto das extensões experimentais e dos utilitários presentes nesta cópia do repositório.

## 1. Escopo e critério de atribuição

O pipeline de referência descrito no README é o fluxo A→E de `src/IQLoc/`: recuperação inicial, geração de palavras-chave do relatório, pontuação com Cross-Encoder, geração de palavras-chave do código e localização/avaliação. `src/IR/`, `src/IR_Reretrieval/` e `src/Utils/` fornecem indexação, busca e utilitários desse fluxo.

As extensões experimentais mais claramente identificáveis nesta cópia são:

| Área | Onde está | O que contém |
|---|---|---|
| Extração alternativa de termos | `src/Modified_IQLoc/b/` e `src/Modified_IQLoc/d/` | Implementações RAKE, TextRank e YAKE, separadas para relatórios de bugs e código-fonte. |
| Seleção e pontuação de modelos | `src/IQLoc/cross_encoder_selector.py`, `src/IQLoc/CE_Method_Scorer.py` e `src/IQLoc/c_Generate_CE_Score.py` | Menu de modelos, carregamento com `sentence-transformers`, inferência em lotes e tratamento de falta de memória CUDA. |
| Configuração e execução de experimentos | `scripts/` | Seletores, executores, conversão/adaptação de datasets, correções e normalização de caminhos. |
| Evidências de execução | `Output/` | Caches das etapas, arquivos de localização e métricas. |

Essa classificação é baseada na organização atual do código e no README; **não é uma reconstrução completa do histórico de autoria**. O diretório fornecido não contém metadados `.git`, portanto não foi possível comparar commits nem confirmar, linha a linha, quais alterações foram feitas em relação ao repositório upstream. A pasta `src/IQLoc/old/` contém versões anteriores de alguns scripts, mas também não substitui o histórico Git.

## 2. Métodos de extração de palavras-chave

As variantes experimentais mantêm o contrato de dados do pipeline: recebem uma lista de relatórios JSON, colocam os termos extraídos no campo `keywords` e preservam os demais campos do registro. A entrada textual é formada por `bug_title + " . " + bug_description` nos relatórios; para código, a entrada é o corpo de cada método Java extraído na etapa A.

### 2.1. Relatórios de bugs (etapa B)

| Método | Implementação | Biblioteca e parâmetros observáveis |
|---|---|---|
| KeyBERT (referência) | `src/IQLoc/b_Generate_BR_Keywords.py` | `keybert.KeyBERT` com `flair.embeddings.TransformerDocumentEmbeddings("Salesforce/codet5-small")`; n-grama de 1 palavra, stop words em inglês, MMR ativado, `diversity=0.5` e `top_n=15`. |
| RAKE | `src/Modified_IQLoc/b/b_RAKE_Generate_BR_Keywords.py` | `rake-nltk.Rake(language="english", max_length=1)`; mantém até 15 frases ranqueadas. |
| TextRank | `src/Modified_IQLoc/b/b_TextRank_Generate_BR_Keywords.py` | `summa.keywords` em inglês, `ratio=1.0`, saída dividida em palavras; descarta termos com mais de uma palavra e limita a 15. |
| YAKE | `src/Modified_IQLoc/b/b_YAKE_Generate_BR_Keywords.py` | `yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=15)`. |

As alternativas RAKE, TextRank e YAKE **não** usam o modelo CodeT5/KeyBERT: são extratores lexicais/estatísticos. As declarações de alguns metadados (`n_gram_left`, `n_gram_right`, `stop_words`, `nr_candidates`) nas variantes são herdadas do formato do script original e não representam necessariamente parâmetros usados pelo extrator. Em particular, `nr_candidates` recebe o valor de diversidade da implementação original, embora não seja um parâmetro do RAKE, TextRank ou YAKE.

### 2.2. Código-fonte (etapa D)

| Método | Implementação | Aplicação e parâmetros |
|---|---|---|
| KeyBERT (referência) | `src/IQLoc/d_Generate_Source_Keywords.py` | Mesma combinação KeyBERT + `Salesforce/codet5-small`; n-grama unitário, stop words em inglês, MMR com diversidade 0.5 e até 15 termos. |
| RAKE | `src/Modified_IQLoc/d/d_Rake_Generate_Source_Keywords.py` | `rake-nltk.Rake(language="english", min_length=1, max_length=1)`; até 15 frases ranqueadas. |
| TextRank | `src/Modified_IQLoc/d/d_TextRank_Generate_Source_Keywords.py` | `summa.keywords` em inglês; mantém até 15 termos de uma palavra. |
| YAKE | `src/Modified_IQLoc/d/d_Yake_Generate_Source_Keywords.py` | `yake.KeywordExtractor(lan="en", n=1, top=15)`; até 15 termos. |

Os quatro métodos estão implementados para **os dois tipos de entrada**. Na etapa D, porém, só são processados métodos que tenham `method_body` textual e `ce_score` numérico maior ou igual a `0.5`. Os termos são gravados em `src_kw_15`; `src_kw_sim_15` recebe o próprio `ce_score`, e não uma similaridade calculada pelo extrator. A chave de saída e o limite são iguais para todas as variantes.

Os scripts alternativos usam caminhos fixos, não uma interface comum por argumentos. As variantes da etapa B gravam em `Output/Generation/Modified/Bug Reports/`; as da etapa D, em `Output/Generation/Modified/Source Code/`. Confirme e ajuste `INPUT_FILE`, `OUTPUT_DIR` e `OUTPUT_FILE_NAME` no script escolhido antes de combiná-lo com caches de outro experimento.

## 3. Comparação de Cross-Encoders

### 3.1. Modelos e nível de suporte

`src/IQLoc/cross_encoder_selector.py` lista os seguintes identificadores:

| Identificador técnico | Nome apresentado no menu | Situação no código atual |
|---|---|---|
| `Qwen/Qwen3-Reranker-0.6B` | Qwen3 Reranker 0.6B | Marcado como compatível e enviado ao `sentence-transformers.CrossEncoder`; não há arquivo de saída com esse nome na pasta `Output/`. |
| `Qwen/Qwen3-Reranker-4B` | Qwen3 Reranker 4B | Marcado como compatível e enviado ao `CrossEncoder`; não há saída correspondente no repositório. |
| `jinaai/jina-reranker-v3.5` | Jina Reranker v3.5 | Explicitamente marcado como incompatível com o scorer genérico; o seletor impede escolhê-lo. |
| `cross-encoder/ettin-reranker-400m-v1` | Ettin Reranker 400M | Marcado como compatível; não há saída correspondente na pasta `Output/`. |
| `BAAI/bge-reranker-v2-m3` | BGE Reranker v2 M3 | Marcado como compatível; existe cache/resultados com prefixo `BAAI_bge-reranker-v2-m3`. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | MS MARCO MiniLM L6 | Marcado como compatível; existe cache/resultados com prefixo `cross-encoder_ms-marco-MiniLM-L-6-v2`. |
| `cross-encoder/ms-marco-electra-base` | MS MARCO ELECTRA | Marcado como compatível e identificado como original no seletor; `src/IQLoc/old/` também o usa como padrão anterior. |

O carregamento efetivo ocorre em `src/IQLoc/CE_Method_Scorer.py`, por `sentence_transformers.CrossEncoder`. O scorer escolhe CUDA quando disponível, caso contrário CPU; o padrão é lote de 16 e comprimento máximo definido pelo modelo/tokenizador. Ele pontua cada par `[method_body, bug_title + " . " + bug_description]`, grava o número em `ce_score` e reduz o lote progressivamente após `torch.cuda.OutOfMemoryError`.

Embora o seletor declare configurações específicas de lote, comprimento e FP16 para Qwen, `c_Generate_CE_Score.py` instancia o scorer apenas com o identificador do modelo. Assim, essas configurações específicas não são passadas pelo caminho atual da etapa C. “Compatível” no menu indica a rota genérica de carregamento, não comprovação de execução bem-sucedida ou de equivalência de desempenho para todos os modelos.

`src/IQLoc/old/cross_encoder_selector.py` também lista `cross-encoder/ms-marco-MiniLM-L-12-v2`. Esse identificador é evidência de uma opção em código anterior, não de suporte no seletor atual nem de execução demonstrada. `src/IQLoc/Train_CE/CrossEncoder_Train.py` é um experimento separado de treinamento que usa `microsoft/codebert-base`, pares de diff/relatório e caminhos absolutos externos; não é uma opção da comparação de modelos da etapa C.

`scripts/set_ce_model.py` tenta alterar `DEFAULT_CE_MODEL` por substituição de texto. Como a etapa C atual define essa variável com o valor de `MODEL_NAME`, e não com uma string literal, a busca textual usada por esse utilitário não corresponde ao código atual.

### 3.2. Seleção, prefixos e risco de mistura

Há **dois mecanismos distintos** de seleção:

1. `src/IQLoc/cross_encoder_selector.py` seleciona o identificador do modelo para a pontuação.
2. `scripts/CrossEncoderConfig.py` solicita outra opção e define `CROSS_ENCODER_PREFIX`. As opções 1–3 são prefixos associados a ELECTRA, BGE e MiniLM; as opções 4–6 são `YAKE`, `RAKE` e `TextRank` — não são identificadores de Cross-Encoder.

O prefixo é incorporado aos nomes esperados por B, C, D e E, por exemplo `<prefixo>_Cached_CE_Scores.json` e `<prefixo>_Localization_Performance.json`. Como os seletores são independentes, escolha inconsistente pode ligar uma pontuação a arquivos com nomes que não correspondem ao modelo efetivamente carregado. Além disso, a etapa A grava `Cached_ES_Results.json` sem prefixo, enquanto B procura um cache de recuperação prefixado.

Para não sobrescrever nem misturar experimentos:

- mantenha uma cópia imutável do cache da etapa A e dos resultados de cada modelo;
- use nomes distintos para cada combinação de extrator B, modelo C e extrator D;
- confira antes de cada execução os caminhos `DEFAULT_INPUT_FILE`, `DEFAULT_OUTPUT_DIR` e `DEFAULT_*_FILE_NAME`;
- confirme que o nome selecionado no menu de prefixo corresponde ao conteúdo real do arquivo, não apenas ao rótulo;
- não considere os prefixos existentes uma garantia de isolamento: vários scripts alternativos gravam em nomes fixos.

`scripts/CrossEncoderConfig.py` solicita entrada durante a importação. A etapa C também importa o seletor de modelos. Isso torna os scripts interativos e explica por que seleções repetidas em etapas diferentes podem divergir.

## 4. Adaptação de datasets e código-fonte

### 4.1. O que foi encontrado para BeetleBox

Não foi encontrado no código, na documentação ou nos dados do projeto um conversor ou uma referência específica a **BeetleBox**. Portanto, esta versão não permite afirmar que BeetleBox esteja adaptado ou suportado. A conversão presente em `scripts/dataset.py` lê o formato de repositório do **Bench4BL**; não deve ser descrita como conversor BeetleBox.

### 4.2. Conversão de XML Bench4BL para JSON do IQLoc

`scripts/dataset.py` percorre `data/<sub_project>/<project>/bugrepo/repository/*.xml` e consulta também `data/<sub_project>/<project>/sources/`. Para cada `<bug>`, lê:

- atributo `bug/@id` → `bug_id`;
- `buginformation/summary` → `bug_title`;
- `buginformation/description` → `bug_description`;
- `buginformation/version` → versão candidata;
- `fixedFiles/file` → lista `fixed_files`;
- nomes dos diretórios → `project` e `sub_project`.

A versão de saída deve coincidir com o nome da pasta indexada em `sources/`. O script tenta correspondência exata; depois normaliza pontos e hífens para `_` e tenta o formato `<project>_<versão>`; por fim, procura uma pasta cujo nome termine com essa versão normalizada. Se não encontrar, deixa `version` vazia e emite aviso em vez de inventar uma correspondência.

O formato produzido é uma lista JSON de objetos com `bug_id`, `project`, `sub_project`, `version`, `bug_title`, `bug_description` e `fixed_files`, compatível com os campos que as etapas A e E consomem. A cópia atual de `scripts/dataset.py`, porém, termina com uma chamada de exemplo com caminhos absolutos Linux (`/home/josue/...`), sem guarda `if __name__ == "__main__"` nem argumentos CLI. Para reutilizá-la, é necessário ajustar os caminhos no próprio arquivo e verificar os dados antes de executar.

### 4.3. Indexação e correções auxiliares

- `scripts/load.py` percorre `<raiz>/<sub_project>/<project>/sources/<versão>/`, lê recursivamente arquivos `.java` e os envia ao indexador como `project`, `sub_project`, `version`, `source_code` e `file_url`. A raiz está fixada em `D:\Bench4BL_Selecionado\data`; ajuste-a para a instalação local. O caminho de versão enviado é o nome do diretório, que deve corresponder ao JSON.
- `scripts/repair_dataset.py` preenche apenas versões vazias em `Dataset/Bench4BLExtended.json`, consultando XMLs e usando a chave `(sub_project, project, bug_id)`. Escreve `Bench4BLExtended_repaired.json`; as raízes de entrada e saída de dados estão fixadas em caminhos Linux.
- `scripts/repair_empty_versions.py` trata versões vazias em cache de recuperação, pesquisa XMLs agrupados por projeto, mantém checkpoints e grava um cache reparado. Também usa caminhos absolutos Linux.
- `scripts/merge_cache.py` combina um cache com um checkpoint JSONL em streaming e valida contagens históricas fixas (`9459`, `4799`, `4660`). Não é um conversor genérico: esses valores precisam corresponder ao conjunto usado.
- `scripts/repair_dataset_new.py` seleciona registros de `Bench4BLExtended.json` que aparecem no cache de palavras-fonte, comparando `(project, sub_project, bug_id)` e ignorando `version`. O script move o JSON original para `Bench4BLExtended_old.json` antes de gravar o subconjunto; só deve ser usado com backup e revisão do critério de seleção.
- `scripts/normalize_es_file_urls.py` normaliza `file_url` no Elasticsearch para caminhos relativos em notação pontuada, reconhecendo layouts Java conhecidos; preserva o valor anterior em `file_url_original`. Aceita `--host`, `--port`, `--index`, `--batch-size` e `--dry-run`. Use `--dry-run` antes de atualizar documentos.

Esses utilitários resolvem problemas de compatibilidade entre nomes de versão, caminhos de arquivo e o esquema esperado pelo IQLoc/Bench4BL. Não fazem mapeamento de campos BeetleBox.

## 5. Pipeline experimental de ponta a ponta

O fluxo implementado é:

| Etapa | Operação | Entrada principal | Campos/resultados adicionados |
|---|---|---|---|
| A — recuperação | `src/IQLoc/a_Cache_ES_Results.py` consulta Elasticsearch com `project`, `sub_project` e `version`; usa `keywords`, se existente, ou título + descrição; busca até 100 arquivos e extrai métodos Java. | `Dataset/Bench4BLExtended.json`; índice normal configurado em `src/IR/config/IR_config.yaml`. | `Results[]`, contendo `file_url`, `source_code`, `bm25_score` e `methods[]` (`method_name`, `method_body`), além de `retrieval_query` e `retrieval_top_k`. |
| B — consulta do relatório | Script KeyBERT original ou alternativa em `src/Modified_IQLoc/b/`. | Cache da A. | `keywords` no nível do bug e metadados de extração. |
| C — pontuação semântica | `src/IQLoc/c_Generate_CE_Score.py` pontua cada corpo de método contra título + descrição do bug. | Saída B. | `ce_score` em cada método. |
| D — termos do código | Script KeyBERT original ou alternativa em `src/Modified_IQLoc/d/`; ignora métodos abaixo de `ce_score=0.5`. | Saída C. | `src_kw_15` e `src_kw_sim_15` em métodos elegíveis. |
| E — re-retrieval e avaliação | `src/IQLoc/e_Localize_Bug.py` seleciona os termos de até cinco métodos com maior pontuação, calcula a interseção de tokens, acrescenta os termos comuns à consulta e faz nova busca de até 10 arquivos. O `Searcher_RE` restringe a busca por `project`, `sub_project`, `version` e `bug_id`, além de consultar `source_code`. | Saída D; `fixed_files` como verdade-terreno; índice configurado em `src/IR_Reretrieval/config/IR_config_2.yaml`, cujo mapeamento inclui `bug_id`. | Registros `Localized_Bugs.json` e métricas `Localization_Performance.json`. |

Na etapa E, se não houver termos de código ou interseção entre os métodos selecionados, a consulta continua sendo apenas `keywords` do relatório. Os resultados são comparados a `fixed_files` pelos caminhos retornados. O avaliador implementa MAP, MRR, Recall@K e Hit@K, mas `evaluate_several` grava MAP, MRR e Hit@K; o script E atual solicita `K=1,5,10`. Não se deve esperar Recall@K no JSON atual.

## 6. Execução e estado da automação

### 6.1. Pré-requisitos

O [README.md](./README.md) descreve a instalação de Python, dependências de `requirements.txt`, Elasticsearch e os campos necessários no índice (`project`, `sub_project`, `version`, `source_code`, `file_url`). Para os utilitários de re-retrieval, a configuração correspondente também inclui `bug_id`. Antes de indexar, confirme que:

1. Elasticsearch está acessível em `localhost:9200` ou no endereço configurado;
2. os nomes de projeto, subprojeto e versão no JSON coincidem com os documentos indexados;
3. os `file_url` retornados são comparáveis aos valores em `fixed_files`;
4. os dois arquivos YAML apontam para os índices corretos.

### 6.2. Comandos das etapas

As chamadas abaixo representam os scripts de referência descritos no README:

```powershell
python src\IQLoc\a_Cache_ES_Results.py
python src\IQLoc\b_Generate_BR_Keywords.py
python src\IQLoc\c_Generate_CE_Score.py
python src\IQLoc\d_Generate_Source_Keywords.py
python src\IQLoc\e_Localize_Bug.py
```

No código atual, não trate essa sequência como execução pronta sem conferir os caminhos antes. A etapa A grava `Output/Caching/Cached_ES_Results.json`; B procura `Output/Caching/<CROSS_ENCODER_PREFIX>_Cached_ES_Results.json`. As etapas B–E importam o seletor interativo de prefixo, e C também solicita modelo. Portanto, alinhe manualmente as entradas/saídas e prefixos em cada script antes de executar uma cadeia completa. Preserve os caches anteriores para comparação.

### 6.3. Orquestradores

- `scripts/run_experiment.py` apresenta uma interface para combinar extratores B/D e Cross-Encoders, mas invoca scripts com `--input` e `--output`, argumentos que os scripts chamados não implementam de forma geral; também espera arquivos em diretórios `Experiments` que não são os caminhos fixos das variantes. Não é uma interface validada para reproduzir os experimentos sem correções.
- `scripts/ke_experiment.py` oferece um menu para extratores, mas seus caminhos do KeyBERT original apontam para subpastas `src/IQLoc/b/` e `src/IQLoc/d/` que não correspondem aos arquivos reais. As opções alternativas apontam para os scripts modificados.
- `scripts/run_pipeline.py` tenta preparar índices, executar A–E e normalizar caminhos. No estado atual, usa uma raiz de dados absoluta via `scripts/load.py`, altera os YAMLs do projeto, e sua substituição temporária do modelo C procura uma atribuição literal a `DEFAULT_CE_MODEL`, enquanto a etapa C atual deriva essa variável do seletor. Assim, revise os caminhos, o modelo e os efeitos sobre índices/configurações antes de utilizá-lo; não o considere uma execução reproduzível sem ajustes.

Para criar índices do zero, inspecione `src/IR/Indexer/Index_Creator.py` e `src/IR_Reretrieval/Indexer/Index_Creator.py` primeiro: os scripts de criação podem remover um índice existente quando executados com a opção de exclusão habilitada. Não execute contra índices com dados que devam ser preservados.

## 7. Arquivos de entrada, caches e resultados

### Estrutura esperada por etapa

- Entrada de avaliação: `Dataset/Bench4BLExtended.json`.
- Cache de A: `Output/Caching/Cached_ES_Results.json`.
- Palavras de relatório: `Output/Generation/`.
- Pontuações C: `Output/Caching/`.
- Palavras-fonte: `Output/Generation/`.
- Localizações e métricas finais: `Output/Localization/`.

Os JSONs são listas de bugs; cada item mantém campos originais como `bug_id`, `project`, `sub_project`, `version`, `bug_title`, `bug_description` e `fixed_files`, e acumula `Results`/`keywords`/pontuações ao longo do pipeline. As métricas são objetos JSON com chaves como `map`, `mrr`, `hit@1`, `hit@5` e `hit@10`.

### Artefatos presentes nesta cópia

- `Output/Caching/`: `Cached_ES_Results.json`, `Cached_CE_Scores.json` e caches com prefixo BAAI, MiniLM, RAKE, TextRank e YAKE.
- `Output/Generation/`: `Cached_BR_Keywords.json`, `Cached_Source_Keywords.json` e variantes KeyBERT/RAKE/TextRank/YAKE; há também cópias sob `Modified/Bug Reports/` e `Modified/Source Code/`.
- `Output/Localization/`: pares de arquivos `*_Localized_Bugs.json` e `*_Localization_Performance.json` para o fluxo base, BAAI, MiniLM, RAKE, TextRank e YAKE.

Os arquivos de localização contêm registros com `fixed_files`, consulta utilizada e `localized_results`; os arquivos de performance contêm as métricas agregadas. O conjunto `Dataset/Bench4BLExtended.json` atualmente presente tem **10 bugs**, e os caches principais examinados também têm 10 registros. Esses arquivos são úteis para inspecionar formato e execução em pequena escala, mas **não demonstram resultados sobre o Bench4BL completo nem sobre BeetleBox**.

## 8. Dependências e limitações de reprodutibilidade

Além das dependências listadas em `requirements.txt`, as etapas usam `keybert`, `flair`, `summa`, `rake-nltk`, `yake`, `sentence-transformers`, PyTorch e o cliente Python do Elasticsearch. A primeira execução pode baixar pesos de modelos do Hugging Face; confira espaço em disco, memória e acesso à rede. A indexação e a busca dependem de versões compatíveis entre Elasticsearch e o cliente instalado.

Limitações relevantes observadas:

- ausência de histórico Git acessível para comprovar a autoria temporal das alterações;
- nenhuma implementação BeetleBox identificável nesta cópia;
- caminhos absolutos Linux/Windows ainda presentes em scripts de conversão e indexação;
- scripts de palavras-chave e seleção de modelo usam caminhos fixos e prompts em importação;
- prefixo de experimento, modelo selecionado e nome do cache não são centralizados em uma única configuração;
- alguns executores não correspondem às interfaces reais dos scripts;
- artefatos atuais representam apenas 10 bugs, não uma avaliação completa;
- scripts de reparo, filtragem e criação de índice podem mover arquivos ou alterar/remover dados; faça cópias e valide a saída antes de usá-los.

Esses pontos devem ser resolvidos ou controlados antes de comparar métricas entre extratores/modelos ou reportar resultados como reprodução integral.
