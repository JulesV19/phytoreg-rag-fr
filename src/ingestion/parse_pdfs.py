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
import fitz
import numpy as np


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

    Le tableau annexe s'étend sur 16+ pages. pdfplumber traite chaque page
    comme un tableau indépendant sans en-tête (seule la page 8 a la ligne
    ['Substance active', 'Nom commercial', 'N°AMM', ...]). On détecte les
    lignes produits par la présence d'un numéro AMM à 7 chiffres en col 2,
    ce qui est robuste quelle que soit la page.
    """
    full_text = _extract_full_text(pdf_path)
    chunks = []
    fichier = Path(pdf_path).name

    # ── Sections réglementaires : découpage fin par sous-section ─────────────
    # Le texte réglementaire couvre 6 pages avec des thématiques distinctes.
    # On split par tous les marqueurs de niveau (I-/II-/III-, 1./2./3., A./B./C.,
    # "Mesures applicables", "Mesures complémentaires") pour obtenir un chunk
    # sémantiquement cohérent par thème.
    _SUBSECTION_RE = re.compile(
        r"(?:^|\n)"
        r"([IVX]+[-]\..*"
        r"|(?:\d+)\.\s+[A-ZÉÈÀÂ].*"
        r"|[A-D]\.\s+[A-ZÉÈÀÂ].*"
        r"|Mesures (?:applicables|complémentaires).*)",
        re.MULTILINE,
    )

    # Titres lisibles pour chaque marqueur
    _SECTION_TITLES = {
        "I-": "Définition du biocontrôle (Art. L.253-6 CRPM)",
        "Mesures applicables": "Avantages réglementaires — produits de biocontrôle (Art. L.253-6)",
        "Mesures complémentaires": "Avantages complémentaires — produits sur liste de biocontrôle (CEPP, publicité, JEVI…)",
        "II-": "Critères d'inscription sur la liste de biocontrôle — présentation",
        "1.": "Critères liste : statut réglementaire (AMM en cours de validité)",
        "2.": "Critères liste : nature des substances actives (micro-organismes, médiateurs chimiques, substances naturelles)",
        "3.": "Critères liste : sécurité pour la santé et l'environnement — présentation",
        "A.": "Critères liste : exclusion des substances candidates à la substitution",
        "B.": "Critères liste : exclusion par mentions de danger (santé publique — H300, H317, H340, H360…)",
        "C.": "Critères liste : exclusion par dangers environnementaux",
        "III-": "Actualisation mensuelle de la liste officielle de biocontrôle",
    }

    def _titre_for(header_line: str) -> str:
        for key, titre in _SECTION_TITLES.items():
            if header_line.strip().startswith(key):
                return titre
        return header_line.strip()[:80]

    # Extraire le corps réglementaire (I- jusqu'à ANNEXE:)
    # On ne peut pas utiliser _extract_section ici : il est case-insensitive et
    # s'arrêterait au mot "annexe" dans "liste en annexe" (texte courant) avant
    # d'atteindre le vrai header "ANNEXE: Liste des produits…" en début de ligne.
    _reg_start = re.search(r"I-\.", full_text)
    _reg_end   = re.search(r"(?m)^ANNEXE:", full_text)
    reg_body   = full_text[_reg_start.start():_reg_end.start()].strip() if (_reg_start and _reg_end) else ""

    if reg_body:
        sub_sections = _split_by_pattern(reg_body, _SUBSECTION_RE)
        for header, body in sub_sections:
            if header == "__intro__":
                continue
            text_chunk = f"{_titre_for(header)}\n{header.strip()}\n{body.strip()}"
            if len(text_chunk) < 80:
                continue
            chunks.append(TextChunk(
                text=text_chunk[:1400],
                metadata={
                    "source": "note_biocontrole",
                    "type": "biocontrole",
                    "section": _titre_for(header),
                    "fichier": fichier,
                    "date": "2026-03-27",
                },
            ))

    # ── Tableau annexe : scan de toutes les pages ────────────────────────────
    # Colonnes attendues (index fixes dans le tableau pdfplumber) :
    # 0=substance_active  1=nom_commercial  2=numero_amm  3=mention_eaj
    # 4=remarque  5=production_bio  6=cepp  7=faible_risque
    COL_SUBSTANCE = 0
    COL_NOM       = 1
    COL_AMM       = 2
    COL_EAJ       = 3
    COL_REMARQUE  = 4
    COL_BIO       = 5
    COL_FAIBLE    = 7

    def _cell(row, idx):
        if idx >= len(row):
            return ""
        v = row[idx]
        return str(v).strip() if v else ""

    seen_amm_nom = set()  # éviter les doublons inter-pages

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                for row in table:
                    if not row or len(row) < 3:
                        continue

                    amm = _cell(row, COL_AMM)
                    # Garder uniquement les lignes avec un numéro AMM valide (7 chiffres)
                    if not re.fullmatch(r"\d{7}", amm):
                        continue

                    nom       = _cell(row, COL_NOM)
                    substance = _cell(row, COL_SUBSTANCE)
                    eaj       = _cell(row, COL_EAJ)
                    remarque  = _cell(row, COL_REMARQUE).replace("\n", " ")
                    bio       = _cell(row, COL_BIO)
                    faible    = _cell(row, COL_FAIBLE)

                    # Dédoublonnage : même AMM + même nom commercial
                    key = (amm, nom)
                    if key in seen_amm_nom:
                        continue
                    seen_amm_nom.add(key)

                    # ── Chunk en langage naturel ──────────────────────────
                    intro = f"Le produit {nom} (AMM {amm}) est un produit de biocontrôle"
                    if substance:
                        intro += f", contenant {substance}"
                    intro += "."

                    lines = [intro, f"Substance active : {substance}"]

                    flags = []
                    if bio.strip().lower() in ("oui", "yes", "x"):
                        flags.append("autorisé en production biologique")
                    if eaj.strip().lower() in ("oui", "yes", "x"):
                        flags.append("emploi autorisé dans les jardins (EAJ)")
                    if faible.strip().lower() in ("oui", "yes", "x"):
                        flags.append("substance à faible risque (Art. 47)")
                    if flags:
                        lines.append("Statut : " + " | ".join(flags))

                    if remarque:
                        lines.append(f"Remarque : {remarque}")

                    chunk_text = "\n".join(lines)

                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={
                            "source": "note_biocontrole",
                            "type": "biocontrole_produit",
                            "nom_produit": nom,
                            "numero_amm": amm,
                            "substance_active": substance,
                            "production_bio": bio.strip().lower() in ("oui", "yes", "x"),
                            "mention_eaj": eaj.strip().lower() in ("oui", "yes", "x"),
                            "faible_risque": faible.strip().lower() in ("oui", "yes", "x"),
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


# ─────────────────────────────────────────────
# ARVALIS — Fiches produits (fongicides, etc.)
# ─────────────────────────────────────────────

_DISEASE_KW = [
    'rhyncho', 'helminth', 'oïdium', 'rouille', 'ramulariose',
    'fusariose', 'septoriose', 'piétin', 'brunissure', 'carie',
    'sclérotiniose', 'mildiou', 'alternariose', 'botrytis',
]
_STADE_NAMES = [
    "Plein tallage", "Fin tallage", "Epis 1cm", "1 nœud",
    "2 nœuds", "Dern. feuille", "Epiaison", "Floraison", "Grain laiteux",
]
# x-center (PDF points) of the 10 stade table columns (col 0 = disease name, 1–9 = stages)
_STADE_COL_X = [113, 167, 209, 250, 291, 332, 374, 416, 457, 498]
_SCALE = 3  # render scale for pixel sampling


def _parse_produit_filename(stem: str) -> dict:
    """Extract produit name, type, culture from the Arvalis filename stem."""
    _TYPES = {'fongicide', 'herbicide', 'insecticide', 'molluscicide', 'acaricide', 'nematicide'}
    parts = stem.split('-')
    type_idx = next((i for i, p in enumerate(parts) if p in _TYPES), None)
    if type_idx is None:
        return {'produit': stem.upper(), 'type': '', 'culture': ''}
    produit = ' '.join(parts[:type_idx]).upper()
    produit_type = parts[type_idx]
    # After type: -homologation-...-arvalis-{year}-{culture}--{hash}
    year_idx = next((i for i in range(type_idx + 1, len(parts)) if re.fullmatch(r'\d{4}', parts[i])), None)
    culture = parts[year_idx + 1] if year_idx and year_idx + 1 < len(parts) else ''
    return {'produit': produit, 'type': produit_type, 'culture': culture}


def _is_disease_row(row) -> bool:
    if not row or not row[0]:
        return False
    text = str(row[0])
    # Header/legend rows have embedded newlines and lots of text — exclude them
    if '\n' in text or len(text) > 60:
        return False
    return any(kw in text.lower() for kw in _DISEASE_KW)


def _find_efficacy_table(tables):
    """Return disease rows from the 2-column disease→dose table."""
    for t in tables:
        if not t or not t[0] or len(t[0]) != 2:
            continue
        rows = [r for r in t if _is_disease_row(r)]
        if len(rows) >= 2:
            return rows
    return []


def _find_stade_table(tables):
    """Return disease rows from the 10-column stades table."""
    for t in tables:
        if not t or not t[0] or len(t[0]) < 8:
            continue
        rows = [r for r in t if _is_disease_row(r)]
        if rows:
            return rows
    return []


def _stades_from_bbch(stade_rows) -> dict:
    """Build disease→stade-string map when BBCH text is present."""
    result = {}
    for row in stade_rows:
        disease = row[0].replace('\n', ' ').strip()
        stages = [
            f"{_STADE_NAMES[i]} ({cell})"
            for i, cell in enumerate(row[1:10])
            if cell and 'BBCH' in str(cell)
        ]
        result[disease] = " → ".join(stages) if stages else ""
    return result


def _classify_pixel(r, g, b) -> str | None:
    if r > 240 and g > 240 and b > 240:
        return None
    if int(g) - int(r) > 50 and int(g) > 120:
        return "optimal"
    if int(g) - int(r) > 15 and int(g) > 150:
        return "suboptimal"
    return None


def _stades_from_pixels(pdf_path: str, stade_rows, page_height: float) -> dict:
    """
    Build disease→stade-string map by sampling pixel colors for each disease row.
    Used when no BBCH text is present (color-only format).
    """
    # Find y-center of each disease row in the stades section (bottom ~45% of page)
    mid_y = page_height * 0.55
    with pdfplumber.open(pdf_path) as pdf:
        words = pdf.pages[1].extract_words()

    stade_words = [
        w for w in words
        if w['top'] > mid_y and w['x0'] < 120
        and any(kw in w['text'].lower() for kw in _DISEASE_KW)
    ]

    # Cluster consecutive words within 5pt → one row, collect center y in order
    row_centers: list[float] = []
    prev_top = None
    for w in sorted(stade_words, key=lambda x: x['top']):
        if prev_top is None or abs(w['top'] - prev_top) > 5:
            row_centers.append((w['top'] + w['bottom']) / 2)
        prev_top = w['top']

    # Render page to RGB array
    doc = fitz.open(pdf_path)
    pix = doc[1].get_pixmap(matrix=fitz.Matrix(_SCALE, _SCALE))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    result = {}
    for row, center_y in zip(stade_rows, row_centers):
        disease = row[0].replace('\n', ' ').strip()
        py = int(center_y * _SCALE)
        optimal, suboptimal = [], []
        for i, x in enumerate(_STADE_COL_X[1:], 0):
            px = int(x * _SCALE)
            if py < arr.shape[0] and px < arr.shape[1]:
                r, g, b = arr[py, px]
                cls = _classify_pixel(r, g, b)
                if cls == "optimal":
                    optimal.append(_STADE_NAMES[i])
                elif cls == "suboptimal":
                    suboptimal.append(_STADE_NAMES[i])
        parts = []
        if optimal:
            parts.append("Optimum : " + " | ".join(optimal))
        if suboptimal:
            parts.append("Suboptimal : " + " | ".join(suboptimal))
        result[disease] = " — ".join(parts)
    return result


def _bbch_window(stade_rows) -> dict:
    """Extract regulatory window (first→last BBCH cell) per disease from stade table."""
    result = {}
    for row in stade_rows:
        disease = str(row[0]).replace('\n', ' ').strip()
        bbch_indices = [i for i in range(9) if i + 1 < len(row) and row[i + 1] and 'BBCH' in str(row[i + 1])]
        if bbch_indices:
            start = _STADE_NAMES[bbch_indices[0]]
            end = _STADE_NAMES[bbch_indices[-1]]
            result[disease] = start if start == end else f"{start} → {end}"
    return result


def _parse_produit_regulatory(pdf_path: str) -> dict:
    """Extract regulatory fields from page 3 text."""
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[2].extract_text() if len(pdf.pages) > 2 else ""
    if not text:
        return {}
    fields = {
        'dar': _regex_extract(text, r"DAR\s*:\s*(.+?)(?:\n|$)"),
        'znt': _regex_extract(text, r"Zone Non Traitée\s*:\s*(.+?)(?:\n|$)"),
        'delai_rentree': _regex_extract(text, r"Délai de Rentrée\s*:\s*(.+?)(?:\n|$)"),
        'cmr': _regex_extract(text, r"CMR\s*:\s*(.+?)(?:\n|$)"),
        'ab': _regex_extract(text, r"Agriculture biologique\s*:\s*(.+?)(?:\n|$)"),
        'formulation': _regex_extract(text, r"Formulation\s*:\s*(.+?)(?:\n|$)"),
        'amm': _regex_extract(text, r"AMM\s*:\s*(\d+)"),
        'h_codes': ', '.join(re.findall(r'H\d{3}', text)),
    }
    return {k: v for k, v in fields.items() if v}


def _parse_single_produit(pdf_path: str) -> list[TextChunk]:
    """Parse one Arvalis fiche produit → 2 TextChunks (_efficacite, _reglementation)."""
    meta_fn = _parse_produit_filename(Path(pdf_path).stem)
    produit = meta_fn['produit']
    culture = meta_fn['culture']
    produit_type = meta_fn['type']
    fichier = Path(pdf_path).name

    base_meta = {
        'source': 'arvalis_produits',
        'produit': produit,
        'culture': culture,
        'type_produit': produit_type,
        'fichier': fichier,
    }

    with pdfplumber.open(pdf_path) as pdf:
        page2 = pdf.pages[1] if len(pdf.pages) > 1 else pdf.pages[0]
        page2_height = page2.height
        tables = page2.extract_tables() or []

        # Matières actives: Table 1, row 1, col 0
        matieres_actives = ""
        for t in tables:
            if t and len(t) > 1 and t[0] and 'Matière active' in str(t[0][0]):
                matieres_actives = str(t[1][0] or '').replace('\n', ' | ').strip()
                break

    eff_rows = _find_efficacy_table(tables)   # 2-col disease→dose
    stade_rows = _find_stade_table(tables)     # 10-col stades

    # Build stade lookup — always use pixel sampling.
    # BBCH text in cells marks regulatory window boundaries, not optimal stades.
    # Colors in cells encode optimal vs suboptimal windows.
    if stade_rows:
        stade_map = _stades_from_pixels(pdf_path, stade_rows, page2_height)
        window_map = _bbch_window(stade_rows)
    else:
        stade_map = {}
        window_map = {}

    # ── Chunk 1 : efficacité ──────────────────────────────────────────────────
    header = f"Fiche ARVALIS — {produit} ({produit_type}) | Culture : {culture}"
    lines = [header]
    if matieres_actives:
        lines.append(f"Matières actives : {matieres_actives}")
    lines.append("Efficacités et doses homologuées :")

    for row in eff_rows:
        disease = str(row[0] or '').replace('\n', ' ').strip()
        dose = str(row[1] or '').strip()

        def _lookup(m, d):
            v = m.get(d, '')
            if not v:
                for k, val in m.items():
                    if d.lower()[:12] in k.lower() or k.lower()[:12] in d.lower():
                        return val
            return v

        stade_str = _lookup(stade_map, disease)
        window_str = _lookup(window_map, disease)

        dose_label = f"dose : {dose} l/ha" if dose else "non autorisé"
        stade_parts = []
        if stade_str:
            stade_parts.append(stade_str)
        if window_str:
            stade_parts.append(f"Réglementaire : {window_str}")
        stade_label = " — " + " — ".join(stade_parts) if stade_parts else ""
        lines.append(f"  - {disease} : {dose_label}{stade_label}")

    chunk_eff = TextChunk(
        text='\n'.join(lines)[:1400],
        metadata={**base_meta, 'section': 'efficacite'},
    )

    # ── Chunk 2 : réglementation ──────────────────────────────────────────────
    reg = _parse_produit_regulatory(pdf_path)
    reg_lines = [f"Réglementation ARVALIS — {produit} ({produit_type}) | Culture : {culture}"]
    for label, key in [
        ("Délai de Rentrée", 'delai_rentree'),
        ("Zone Non Traitée (ZNT)", 'znt'),
        ("DAR (Délai Avant Récolte)", 'dar'),
        ("CMR", 'cmr'),
        ("Agriculture biologique", 'ab'),
        ("Formulation", 'formulation'),
        ("N° AMM", 'amm'),
        ("Mentions de danger", 'h_codes'),
    ]:
        if key in reg:
            reg_lines.append(f"  {label} : {reg[key]}")

    chunk_reg = TextChunk(
        text='\n'.join(reg_lines)[:1400],
        metadata={**base_meta, 'section': 'reglementation', **{k: reg.get(k, '') for k in ('dar', 'znt', 'ab')}},
    )

    return [chunk_eff, chunk_reg]


def parse_arvalis_produits(produits_dir: str) -> list[TextChunk]:
    """Parse all Arvalis fiche produit PDFs in produits_dir → 2 chunks per PDF."""
    chunks = []
    for pdf_path in sorted(Path(produits_dir).glob("*.pdf")):
        try:
            chunks.extend(_parse_single_produit(str(pdf_path)))
        except Exception as e:
            chunks.append(TextChunk(
                text=f"[Erreur parsing {pdf_path.name}: {e}]",
                metadata={'source': 'arvalis_produits', 'fichier': pdf_path.name, 'section': 'error'},
            ))
    return chunks


# ─────────────────────────────────────────────
# ARVALIS — Fiches variétés
# ─────────────────────────────────────────────

def parse_arvalis_varietes(varietes_dir: str) -> list[TextChunk]:
    """Parse all Arvalis fiche variété PDFs → 1 chunk per PDF."""
    chunks = []
    _SKIP_LINES = {
        'AddThis est désactivé.', 'Autoriser', 'Cliquer sur la carte pour',
        'choisir un département', 'Département selectionné :', '-',
        'Veuillez sélectionner un département', 'Accueil | Poser une question | CGU',
        'Sources des données : GEVES et ARVALIS', 'Accédez à l\'outil',
        'Densité de semis', 'Choix des variétés', 'Rendements inscription',
        'Rendements post-inscription', 'Choisissez une zone',
        'Fiches accidents', 'Fiches fertilisants', 'Fiches biodiversité',
    }

    for pdf_path in sorted(Path(varietes_dir).glob("*.pdf")):
        try:
            lines = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages[:2]:  # pages 1 and 2 have the data
                    text = page.extract_text()
                    if not text:
                        continue
                    for line in text.splitlines():
                        s = line.strip()
                        if not s or any(skip in s for skip in _SKIP_LINES):
                            continue
                        lines.append(s)

            # Extract name and culture from filename (more reliable than page text)
            stem = pdf_path.stem
            # Pattern: {variete-name}-variete-de-{culture-words}-{section}-...-{hash}
            _SECTION_KW = r'(?:caracteristiques|resistances|rendement|resultats)'
            match = re.match(rf'^(.+?)-variete-de-(.+?)-{_SECTION_KW}', stem)
            if match:
                variete_name = match.group(1).replace('-', ' ').title()
                culture_var = match.group(2).replace('-', ' ')
            else:
                variete_name = stem.split('-variete')[0].replace('-', ' ').title()
                culture_var = ''

            body = '\n'.join(lines)
            header = f"Fiche variété ARVALIS — {variete_name}"
            if culture_var:
                header += f" ({culture_var})"

            chunks.append(TextChunk(
                text=f"{header}\n{body}"[:1400],
                metadata={
                    'source': 'arvalis_varietes',
                    'variete': variete_name,
                    'culture': culture_var,
                    'fichier': pdf_path.name,
                },
            ))
        except Exception as e:
            chunks.append(TextChunk(
                text=f"[Erreur parsing {pdf_path.name}: {e}]",
                metadata={'source': 'arvalis_varietes', 'fichier': pdf_path.name},
            ))
    return chunks


# ─────────────────────────────────────────────
# ARVALIS — Fiches fertilisants
# ─────────────────────────────────────────────

def parse_arvalis_fertilisants(fertilisants_dir: str) -> list[TextChunk]:
    """Parse all Arvalis fiche fertilisant PDFs → 1 chunk per PDF."""
    _SKIP = {'AddThis est désactivé.', 'Autoriser', 'Accueil | Poser une question | CGU',
             'Sources des données : ARVALIS', 'Fiches accidents', 'Fiches variétés',
             'Fiches fertilisants', 'Fiches biodiversité', 'Fiches produits', 'Fiches couverts'}

    chunks = []
    for pdf_path in sorted(Path(fertilisants_dir).glob("*.pdf")):
        try:
            lines = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    for line in text.splitlines():
                        s = line.strip()
                        if not s or any(skip in s for skip in _SKIP):
                            continue
                        lines.append(s)

            # Product name: first substantive line (before "Identité")
            name = pdf_path.stem.split('--')[0].replace('-', ' ').title()
            body = '\n'.join(lines)
            header = f"Fiche fertilisant ARVALIS — {name}"

            chunks.append(TextChunk(
                text=f"{header}\n{body}"[:1400],
                metadata={
                    'source': 'arvalis_fertilisants',
                    'produit': name,
                    'fichier': pdf_path.name,
                },
            ))
        except Exception as e:
            chunks.append(TextChunk(
                text=f"[Erreur parsing {pdf_path.name}: {e}]",
                metadata={'source': 'arvalis_fertilisants', 'fichier': pdf_path.name},
            ))
    return chunks


# ─────────────────────────────────────────────
# ARVALIS — Fiches couverts
# ─────────────────────────────────────────────

def parse_arvalis_couverts(couverts_dir: str) -> list[TextChunk]:
    """Parse all Arvalis fiche couvert PDFs → 1–2 chunks per PDF."""
    _SKIP = {'AddThis est désactivé.', 'Autoriser', 'Accueil | Poser une question | CGU',
             'Sources des données : ARVALIS', 'Fiches accidents', 'Fiches variétés',
             'Fiches fertilisants', 'Fiches biodiversité', 'Fiches produits', 'Fiches couverts',
             '+ +', 'Légende :'}
    # Symbols used in adaptation tables that aren't useful as standalone lines
    _SKIP_RE = re.compile(r'^[+\-/]{1,3}$')

    chunks = []
    for pdf_path in sorted(Path(couverts_dir).glob("*.pdf")):
        try:
            all_lines = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    for line in text.splitlines():
                        s = line.strip()
                        if not s or any(skip in s for skip in _SKIP) or _SKIP_RE.match(s):
                            continue
                        all_lines.append(s)

            name = pdf_path.stem.split('--')[0].replace('-', ' ').title()
            header = f"Fiche couvert ARVALIS — {name}"
            body = '\n'.join(all_lines)

            # Split into 2 chunks if content is long (characteristics vs rotation/adaptation)
            rotation_idx = next(
                (i for i, l in enumerate(all_lines) if 'rotation' in l.lower() or 'adaptation' in l.lower()),
                None
            )

            base_meta = {'source': 'arvalis_couverts', 'produit': name, 'fichier': pdf_path.name}

            if rotation_idx and rotation_idx > 5 and len(body) > 800:
                carac_text = '\n'.join(all_lines[:rotation_idx])
                rota_text = '\n'.join(all_lines[rotation_idx:])
                chunks.append(TextChunk(
                    text=f"{header} — Caractéristiques\n{carac_text}"[:1400],
                    metadata={**base_meta, 'section': 'caracteristiques'},
                ))
                chunks.append(TextChunk(
                    text=f"{header} — Rotation et adaptation\n{rota_text}"[:1400],
                    metadata={**base_meta, 'section': 'rotation'},
                ))
            else:
                chunks.append(TextChunk(
                    text=f"{header}\n{body}"[:1400],
                    metadata={**base_meta, 'section': 'general'},
                ))
        except Exception as e:
            chunks.append(TextChunk(
                text=f"[Erreur parsing {pdf_path.name}: {e}]",
                metadata={'source': 'arvalis_couverts', 'fichier': pdf_path.name},
            ))
    return chunks
