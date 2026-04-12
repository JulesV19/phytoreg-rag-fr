"""
Parser des fichiers PDF réglementaires et des décisions AMM individuelles.

- Arrêté du 4 mai 2017 : chunking par article
- Décisions AMM individuelles : chunking par section
- Note de service biocontrôle : chunking par section + lignes de tableau
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber


@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Arrêté du 4 mai 2017
# ─────────────────────────────────────────────

def parse_arrete_2017(pdf_path: str) -> list[TextChunk]:
    """
    Parse l'arrêté du 4 mai 2017 avec une granularité fine :
    - Un chunk par SOUS-PARAGRAPHE (I., II., III., IV.) à l'intérieur de chaque article.
    - Quand un article n'a pas de sous-paragraphes, un chunk par article.
    - Chaque chunk répète explicitement l'article et le sous-paragraphe en en-tête
      pour ancrer l'embedding sur la règle précise.

    Cela permet à "délai de rentrée par défaut" de trouver Art. 3-II (6 heures)
    plutôt que l'article 3 entier dilué sémantiquement.
    """
    full_text = _extract_full_text(pdf_path)
    chunks = []

    # Découpage par article
    article_pattern = re.compile(
        r"(Art\.\s*\d+[a-z\s]*[\.]\s*[–—-]|TITRE\s+[IVX]+\b[^\n]*)",
        re.MULTILINE,
    )
    sections = _split_by_pattern(full_text, article_pattern)

    # Chunk d'intro
    if sections and sections[0][0] == "__intro__":
        intro_text = sections[0][1].strip()
        if len(intro_text) > 100:
            chunks.append(TextChunk(
                text=intro_text[:1400],
                metadata={
                    "source": "arrete_2017",
                    "type": "reglementation",
                    "section": "introduction",
                    "article": "",
                    "paragraphe": "",
                    "titre": "Arrêté du 4 mai 2017 - Présentation générale",
                    "date": "2017-05-04",
                },
            ))

    # Pattern de sous-paragraphes : "I. –", "II. –", "III. –", "IV. –" etc.
    para_pattern = re.compile(
        r"((?:^|\n)\s*([IVX]+)\.\s*[–—-])",
        re.MULTILINE,
    )

    for header, body in sections:
        if header == "__intro__":
            continue

        article_num = _extract_article_number(header)
        titre = _extract_titre(header)
        label_article = f"Article {article_num}" if article_num else titre or header.strip()
        prefix_article = (
            f"Arrêté du 4 mai 2017 — {label_article} "
            f"relatif à l'utilisation des produits phytopharmaceutiques.\n"
        )

        # Tenter de découper par sous-paragraphes
        sub_sections = _split_by_pattern(body, para_pattern)
        has_sub = len(sub_sections) > 1 or (sub_sections and sub_sections[0][0] != "__intro__")

        if has_sub:
            for sub_header, sub_body in sub_sections:
                if sub_header == "__intro__":
                    # Texte avant le premier paragraphe (ex: "Sauf dispositions contraires...")
                    if sub_body.strip():
                        text = f"{prefix_article}{header}\n{sub_body}".strip()
                        chunks.append(TextChunk(
                            text=text[:1400],
                            metadata={
                                "source": "arrete_2017",
                                "type": "reglementation",
                                "section": header.strip(),
                                "article": article_num,
                                "paragraphe": "intro",
                                "titre": f"{label_article} — Arrêté 4 mai 2017",
                                "date": "2017-05-04",
                            },
                        ))
                    continue

                # Extraire le numéro de paragraphe romain (I, II, III...)
                para_num_match = re.search(r"([IVX]+)\.", sub_header)
                para_num = para_num_match.group(1) if para_num_match else ""

                text = (
                    f"{prefix_article}"
                    f"{label_article} — Paragraphe {para_num}\n"
                    f"{sub_header} {sub_body}"
                ).strip()

                chunks.append(TextChunk(
                    text=text[:1400],
                    metadata={
                        "source": "arrete_2017",
                        "type": "reglementation",
                        "section": header.strip(),
                        "article": article_num,
                        "paragraphe": para_num,
                        "titre": f"{label_article} §{para_num} — Arrêté 4 mai 2017",
                        "date": "2017-05-04",
                    },
                ))
        else:
            # Article sans sous-paragraphes : un seul chunk
            text = f"{prefix_article}{header}\n{body}".strip()
            if len(text) < 50:
                continue
            chunks.append(TextChunk(
                text=text[:1400],
                metadata={
                    "source": "arrete_2017",
                    "type": "reglementation",
                    "section": header.strip(),
                    "article": article_num,
                    "paragraphe": "",
                    "titre": titre or f"{label_article} — Arrêté 4 mai 2017",
                    "date": "2017-05-04",
                },
            ))

    return chunks


# ─────────────────────────────────────────────
# Décisions AMM individuelles
# ─────────────────────────────────────────────

def parse_decision_amm(pdf_path: str) -> list[TextChunk]:
    """
    Parse une décision AMM individuelle avec un découpage à trois niveaux :

    1. Informations générales — 1 chunk (identité du produit)
    2. Conditions d'emploi   — 1 chunk (EPI, délais, restrictions)
    3. Usages autorisés      — 1 chunk par ligne de tableau (culture × nuisible)
       Avantage : chaque usage a son propre vecteur, très proche des questions
       du type "puis-je utiliser X sur Y ?" plutôt qu'un bloc dilué de 5 000 chars.

    Si le tableau des usages n'est pas extractible via pdfplumber (PDF image/scan),
    on repasse sur un découpage par blocs de lignes (~10 lignes par chunk).
    """
    full_text = _extract_full_text(pdf_path)
    chunks = []

    # Extraire les métadonnées générales
    nom_produit = _regex_extract(full_text, r"Nom commercial\s+(.+?)(?:\n|Numéro)")
    numero_amm = _regex_extract(full_text, r"Numéro d.AMM\s+(\d+)")
    substance = _regex_extract(full_text, r"Substance\(s\) active\(s\)\s+(.+?)(?:\n|Concentration)")
    titulaire = _regex_extract(full_text, r"Titulaire de l.autorisation\s+(.+?)(?:\n\n|\d-)")
    etat = "AUTORISE"

    base_meta = {
        "source": "decision_amm_individuelle",
        "type": "decision_amm",
        "nom_produit": nom_produit,
        "numero_amm": numero_amm,
        "substances_actives": substance,
        "etat_produit": etat,
        "fichier": Path(pdf_path).name,
    }

    # ── Section 1 : Informations générales ───────────────────────────────────
    section_general = _extract_section(full_text, r"1-\s*Informations générales", r"2-\s*Conditions")
    if section_general:
        chunks.append(TextChunk(
            text=(
                f"Décision AMM — Informations générales\n"
                f"Produit : {nom_produit} (AMM {numero_amm}) | Substance : {substance} | "
                f"Titulaire : {titulaire}\n{section_general}"
            ),
            metadata={**base_meta, "section": "informations_generales"},
        ))

    # ── Section 2 : Conditions d'emploi ──────────────────────────────────────
    section_emploi = _extract_section(full_text, r"2-\s*Conditions d.emploi", r"3-\s*Usage")
    if section_emploi:
        chunks.append(TextChunk(
            text=(
                f"Conditions d'emploi du produit {nom_produit} (AMM {numero_amm}) — "
                f"Décision AMM officielle\n{section_emploi}"
            ),
            metadata={**base_meta, "section": "conditions_emploi"},
        ))

    # ── Section 3 : Usages autorisés — découpage fin ─────────────────────────
    usage_chunks = _parse_usages_table(pdf_path, nom_produit, numero_amm, base_meta)

    if usage_chunks:
        chunks.extend(usage_chunks)
    else:
        # Fallback texte brut : on découpe la section en blocs de ~10 lignes
        section_usages = _extract_section(full_text, r"3-\s*Usage", None)
        if section_usages:
            chunks.extend(
                _split_usages_text(section_usages, nom_produit, numero_amm, base_meta)
            )

    # ── Fallback global ───────────────────────────────────────────────────────
    if not chunks:
        chunks.append(TextChunk(
            text=full_text[:4000],
            metadata={**base_meta, "section": "complet"},
        ))

    return chunks


def _parse_usages_table(
    pdf_path: str,
    nom_produit: str,
    numero_amm: str,
    base_meta: dict,
) -> list[TextChunk]:
    """
    Tente d'extraire les usages depuis les tableaux pdfplumber.
    Retourne une liste de chunks (1 par ligne d'usage) ou [] si aucun tableau trouvé.
    """
    chunks = []
    usage_keywords = {"culture", "nuisible", "organisme", "dose", "dar", "znt", "usage", "méthode"}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                if not table or len(table) < 2:
                    continue
                header = [str(c).lower().strip() if c else "" for c in table[0]]
                # Garder uniquement les tables qui ressemblent à une table d'usages
                if not any(kw in h for kw in usage_keywords for h in header):
                    continue

                col_names = [str(c).strip() if c else f"col{i}" for i, c in enumerate(table[0])]

                for row in table[1:]:
                    if not row or not any(row):
                        continue
                    # Construire un texte naturel à partir des colonnes
                    cells = {col_names[i]: str(v).strip() for i, v in enumerate(row) if v}
                    if len(cells) < 2:
                        continue

                    # Phrase d'accroche naturelle
                    culture = cells.get("Culture", cells.get("culture", ""))
                    nuisible = cells.get("Nuisible", cells.get("nuisible", cells.get("Organisme nuisible", "")))
                    dose = cells.get("Dose", cells.get("dose", cells.get("Dose retenue", "")))
                    dar = cells.get("DAR", cells.get("Délai avant récolte", ""))

                    intro_parts = []
                    if culture:
                        intro_parts.append(f"sur {culture}")
                    if nuisible:
                        intro_parts.append(f"contre {nuisible}")
                    intro = f"Le produit {nom_produit} (AMM {numero_amm}) est autorisé " + " ".join(intro_parts) + "."

                    detail = " | ".join(f"{k} : {v}" for k, v in cells.items() if v)
                    row_text = f"{intro}\n{detail}"

                    chunks.append(TextChunk(
                        text=row_text,
                        metadata={
                            **base_meta,
                            "section": "usages_autorises",
                            "culture": culture,
                            "nuisible": nuisible,
                        },
                    ))

    return chunks


def _split_usages_text(
    section_text: str,
    nom_produit: str,
    numero_amm: str,
    base_meta: dict,
    lines_per_chunk: int = 10,
) -> list[TextChunk]:
    """
    Fallback : découpe la section usages en blocs de lignes quand
    le tableau n'est pas extractible.
    """
    chunks = []
    lines = [l for l in section_text.splitlines() if l.strip()]
    for i in range(0, len(lines), lines_per_chunk):
        block = "\n".join(lines[i : i + lines_per_chunk])
        if len(block) < 20:
            continue
        chunks.append(TextChunk(
            text=f"Usages autorisés — {nom_produit} (AMM {numero_amm})\n{block}",
            metadata={**base_meta, "section": "usages_autorises", "bloc_index": i // lines_per_chunk},
        ))
    return chunks


# ─────────────────────────────────────────────
# Note de service biocontrôle
# ─────────────────────────────────────────────

def parse_note_biocontrole(pdf_path: str) -> list[TextChunk]:
    """
    Parse la note DGAL sur les produits de biocontrôle.
    Sections I, II, III + tableau annexe (une ligne = un chunk).
    """
    full_text = _extract_full_text(pdf_path)
    chunks = []
    fichier = Path(pdf_path).name

    # Sections principales
    sections = [
        ("I-", "II-", "Réglementation applicable aux produits de biocontrôle"),
        ("II-", "III-", "Inscription des produits de biocontrôle sur la liste"),
        ("III-", "ANNEXE", "Actualisation de la liste"),
    ]

    for start, end, titre in sections:
        content = _extract_section(full_text, re.escape(start), re.escape(end) if end else None)
        if content and len(content) > 100:
            chunks.append(TextChunk(
                text=f"{titre}\n{content[:3000]}",
                metadata={
                    "source": "note_biocontrole",
                    "type": "biocontrole",
                    "section": titre,
                    "fichier": fichier,
                    "date": "2026-03-27",
                },
            ))

    # Annexe : tableau des produits de biocontrôle
    # On extrait le tableau ligne par ligne depuis le PDF pour garder les données structurées
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                # Détecter les tables du tableau annexe (ont des colonnes AMM)
                header = [str(c).lower() if c else "" for c in table[0]]
                if not any("amm" in h or "substance" in h or "nom commercial" in h.lower() for h in header):
                    continue

                for row in table[1:]:
                    if not row or not any(row):
                        continue
                    row_text = " | ".join(str(c).strip() for c in row if c)
                    if len(row_text) < 10:
                        continue

                    # Extraire N° AMM depuis la ligne si possible
                    amm_match = re.search(r"\b(\d{7})\b", row_text)
                    amm = amm_match.group(1) if amm_match else ""

                    chunks.append(TextChunk(
                        text=f"Produit de biocontrôle : {row_text}",
                        metadata={
                            "source": "note_biocontrole",
                            "type": "biocontrole_produit",
                            "numero_amm": amm,
                            "fichier": fichier,
                            "date": "2026-03-27",
                        },
                    ))

    return chunks


# ─────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────

def _extract_full_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _split_by_pattern(text: str, pattern) -> list[tuple[str, str]]:
    """Découpe le texte par pattern, retourne (header, body) pairs."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [("__intro__", text)]

    results = []
    # Texte avant le premier match
    intro = text[: matches[0].start()].strip()
    if intro:
        results.append(("__intro__", intro))

    for i, match in enumerate(matches):
        header = match.group(0)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        results.append((header, body))

    return results


def _extract_section(text: str, start_pattern: str, end_pattern: str | None) -> str:
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    if not start_match:
        return ""
    start = start_match.start()
    if end_pattern:
        end_match = re.search(end_pattern, text[start:], re.IGNORECASE)
        end = start + end_match.start() if end_match else len(text)
    else:
        end = len(text)
    return text[start:end].strip()


def _extract_article_number(header: str) -> str:
    match = re.search(r"Art\.\s*(\d+)", header)
    return match.group(1) if match else ""


def _extract_titre(header: str) -> str:
    match = re.search(r"TITRE\s+([IVX]+)\s+(.+)", header)
    return f"Titre {match.group(1)} - {match.group(2).strip()}" if match else ""


def _regex_extract(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:200]
    return ""
