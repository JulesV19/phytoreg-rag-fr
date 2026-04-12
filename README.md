# LLM Phytosanitaire

Assistant conversationnel local pour aider les agriculteurs à naviguer la réglementation phytosanitaire française. Fonctionne entièrement en local — aucune donnée envoyée vers une API externe.

## Stack

| Brique | Rôle | Outil |
|---|---|---|
| Embedding | Texte → vecteur | Ollama + `mxbai-embed-large` |
| Vector store | Stockage et recherche | Qdrant (Docker) |
| LLM | Génération de réponse | Ollama + `mistral:7b` |
| Framework | Orchestration | LangChain |

## Prérequis

- [Ollama](https://ollama.com) installé et lancé
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé
- Python 3.11+

```bash
ollama pull mxbai-embed-large
ollama pull mistral:7b
```

## Installation

```bash
git clone https://github.com/TON_USER/llm-phyto.git
cd llm-phyto
pip install -r requirements.txt
```

## Données sources

Le dossier `Data/` n'est pas versionné. Créer le dossier et y placer les fichiers suivants :

```bash
mkdir Data
```

| Fichier | Source | Taille |
|---|---|---|
| `20170504_AM_Utilisation_pdt_Phyto.pdf` | [Légifrance](https://www.legifrance.gouv.fr) | ~200 KB |
| `2026-168_final (1).pdf` | [DGAL](https://info.agriculture.gouv.fr) | ~700 KB |
| `ActiveSubstanceExport_11-04-2026.xlsx` | [Pesticides DB CE](https://ec.europa.eu/food/plant/pesticides/eu-pesticides-database) | ~230 KB |
| `decisionamm-intrant-format-xml-*/` | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/decisions-damm-intrant/) | ~120 MB |

## Utilisation

**1. Démarrer Qdrant**
```bash
docker compose up -d
```

**2. Ingestion des données** (Environ 1h sur Mac, CPU local)
```bash
python3 -m src.ingestion.ingest
# Depuis zéro : python3 -m src.ingestion.ingest --reset
```

**3. Lancer le chatbot**
```bash
python3 -m src.app.main              # streaming (défaut)
python3 -m src.app.main --no-stream  # réponse complète + sources
```

## Architecture

```
src/
├── embeddings.py          ← MxbaiEmbeddings (préfixe query obligatoire)
├── ingestion/
│   ├── ingest.py          ← Pipeline principal
│   ├── parse_xml.py       ← XML AMM en streaming (iterparse)
│   ├── parse_pdfs.py      ← Arrêté 2017, note biocontrôle
│   └── parse_xlsx.py      ← Substances actives CE
└── rag/
    └── chain.py           ← Query analyzer + retriever hybride + LLM
```

### Sources indexées

- **XML AMM** — base nationale e-phy (~200k–400k usages). Un chunk par usage `culture × nuisible × méthode`, en phrase naturelle + données structurées
- **Arrêté du 4 mai 2017** — réglementation nationale. Un chunk par sous-paragraphe (Art. 3-I, Art. 3-II…)
- **Note biocontrôle DGAL** — liste L.253-6. Un chunk par section + une ligne du tableau = un chunk
- **Substances actives CE** — statuts d'approbation européens. Une substance = un chunk

### Embedding : préfixe de requête mxbai

`mxbai-embed-large` exige un préfixe sur les **requêtes uniquement** :
```
"Represent this sentence for searching relevant passages: {question}"
```
Géré automatiquement par `src/embeddings.py`. Les documents indexés restent sans préfixe.

## Commandes utiles

```bash
# Vérifier l'état de la base
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
print('Documents indexés :', c.count('phyto_docs').count)
"

# Dashboard Qdrant
open http://localhost:6333/dashboard

# Logs Qdrant
docker compose logs qdrant
```
