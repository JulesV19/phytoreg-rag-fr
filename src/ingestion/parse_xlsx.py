"""
Parser du fichier XLSX des substances actives européennes (Pesticides Database - CE).
Chaque substance active devient un chunk textuel avec ses métadonnées.
"""

from dataclasses import dataclass, field
from pathlib import Path
import openpyxl


@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def parse_substances_actives(xlsx_path: str) -> list[TextChunk]:
    """
    Parse le fichier XLSX des substances actives.
    Les vraies colonnes démarrent à la ligne 3 (index 2).
    Retourne un TextChunk par substance active.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    chunks = []

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Ligne 3 = headers réels (index 2)
    if len(rows) < 3:
        return chunks

    headers = [str(h).strip() if h else "" for h in rows[2]]

    # Index des colonnes clés
    col = {h: i for i, h in enumerate(headers) if h}

    def get(row, key):
        idx = col.get(key)
        if idx is None:
            return ""
        val = row[idx] if idx < len(row) else None
        return str(val).strip() if val is not None else ""

    for row in rows[3:]:
        if not any(row):
            continue

        substance = get(row, "Substance")
        if not substance or substance == "None":
            continue

        status = get(row, "Status under Reg. (EC) No 1107/2009")
        date_approbation = get(row, "Date of approval")
        expiration = get(row, "Expiration of approval")
        cas = get(row, "CAS Number")
        legislation = get(row, "Legislation")
        candidat_substitution = get(row, "Candidate for substitution")
        substance_base = get(row, "Basic substance")
        faible_risque = get(row, "Low-risk a.s.")
        fonctions = get(row, "Functions")
        classification = get(row, "Classification (Reg. 1272/2008)")

        # Construction du chunk textuel
        lines = [f"Substance active : {substance}"]
        if cas:
            lines.append(f"Numéro CAS : {cas}")
        if status:
            lines.append(f"Statut réglementaire (Règl. 1107/2009) : {status}")
        if date_approbation:
            lines.append(f"Date d'approbation : {date_approbation}")
        if expiration:
            lines.append(f"Expiration de l'approbation : {expiration}")
        if legislation:
            lines.append(f"Législation : {legislation}")
        if candidat_substitution and candidat_substitution.lower() not in ("no", "none", ""):
            lines.append(f"Candidate à la substitution : {candidat_substitution}")
        if substance_base and substance_base.lower() not in ("no", "none", ""):
            lines.append(f"Substance de base : {substance_base}")
        if faible_risque and faible_risque.lower() not in ("no", "none", ""):
            lines.append(f"Substance à faible risque : {faible_risque}")
        if fonctions:
            lines.append(f"Fonction(s) : {fonctions}")
        if classification:
            lines.append(f"Classification CLP : {classification}")

        text = "\n".join(lines)

        chunks.append(TextChunk(
            text=text,
            metadata={
                "source": "substances_actives_xlsx",
                "type": "substance_active",
                "substance": substance,
                "cas_number": cas,
                "statut": status,
                "date_approbation": date_approbation,
                "expiration": expiration,
                "candidat_substitution": candidat_substitution.lower() not in ("no", "none", ""),
                "faible_risque": faible_risque.lower() not in ("no", "none", ""),
                "fonctions": fonctions,
                "fichier": Path(xlsx_path).name,
            },
        ))

    return chunks
