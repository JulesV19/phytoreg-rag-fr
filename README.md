# ⚠️ Avertissement Important

**Ce projet est un prototype expérimental utilisant une technologie d'Intelligence Artificielle (RAG). Il n'est en aucun cas un outil de conseil réglementaire homologué.**

### 1. Absence de garantie
Les réponses fournies par ce chatbot sont générées automatiquement. Malgré l'utilisation de sources officielles (E-Phy, Légifrance), l'IA peut commettre des erreurs d'interprétation, omettre des mises à jour récentes ou produire des "hallucinations" (réponses factuellement fausses mais crédibles).

### 2. Responsabilité de l'utilisateur
L'utilisateur reste seul responsable de l'application des produits phytopharmaceutiques sur ses cultures. L'utilisation de cet outil ne dispense en aucun cas de la consultation :
* De l'étiquette du produit (qui est l'unique document faisant foi).
* Du site officiel [E-Phy (Anses)](https://www.anses.fr/).
* Des textes de loi sur [Légifrance](https://www.legifrance.gouv.fr/).

### 3. Usage strictement pédagogique
Ce dépôt est une preuve de concept technique destinée à explorer les capacités des LLM dans le domaine agricole. **En aucun cas l'auteur de ce projet ne pourra être tenu responsable des dommages directs ou indirects (sanctions administratives, pertes de récoltes, dommages environnementaux) résultant de l'utilisation des informations fournies par ce chatbot.**

---

# LLM Phytosanitaire

Assistant conversationnel local pour aider les agriculteurs à naviguer la réglementation phytosanitaire française. Fonctionne entièrement en local — aucune donnée envoyée vers une API externe.

## Stack

| Brique | Rôle | Outil |
|---|---|---|
| Embedding dense | Texte → vecteur sémantique (1024 dims) | Ollama + `mxbai-embed-large` |
| Embedding sparse | BM25 pour matching exact (termes techniques) | FastEmbed + `Qdrant/bm25` |
| Vector store | Stockage et recherche hybride (dense + BM25) | Qdrant (Docker) |
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

### Recherche hybride (dense + BM25)

Chaque chunk est indexé avec deux vecteurs complémentaires :

- **Dense** (`mxbai-embed-large`, 1024 dims) — similarité sémantique cosinus
- **Sparse BM25** (`Qdrant/bm25` via FastEmbed) — matching exact sur les termes rares du domaine (numéros AMM, noms de produits, molécules)

Le retriever route selon le type de question :
- `usage_check`, `substance_check` → hybride dense + BM25 (RRF) — le matching exact est critique pour les numéros AMM et noms de produits
- `product_list`, `regulation` → dense seul — la query sémantique focalisée suffit

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
