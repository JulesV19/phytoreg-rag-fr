"""
Parser du fichier XML des décisions AMM (SI-Intrant / e-phy).
Produit un chunk textuel par usage de produit, enrichi de métadonnées structurées.

Stratégie mémoire : iterparse avec clear() pour ne jamais charger tout le fichier.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class UsageChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _get_text(elem, tag: str, default: str = "") -> str:
    node = elem.find(tag)
    return (node.text or default).strip() if node is not None else default


def _parse_usage_label(label: str) -> tuple[str, str, str]:
    """Découpe 'Artichaut*Trt Part.Aer.*Pucerons' en (culture, méthode, nuisible)."""
    parts = label.split("*")
    culture = parts[0].strip() if len(parts) > 0 else ""
    methode = parts[1].strip() if len(parts) > 1 else ""
    nuisible = parts[2].strip() if len(parts) > 2 else ""
    return culture, methode, nuisible


def _build_usage_text(product: dict, usage: dict) -> str:
    """
    Construit le texte du chunk en deux blocs :
    1. Une phrase d'accroche en langage naturel — pour rapprocher l'embedding
       des questions agriculteurs ("puis-je utiliser X sur Y contre Z ?")
    2. Un bloc structuré avec toutes les valeurs techniques — pour que le LLM
       puisse citer des chiffres précis (dose, DAR, ZNT…)
    """
    usage_label = usage.get("label", "")
    culture, methode, nuisible = _parse_usage_label(usage_label)
    nom = product["nom"]
    amm = product["amm"]
    etat = product["etat"]
    gamme = product["gamme"]

    # ── Phrase d'accroche en langage naturel ──────────────────────────────────
    etat_phrase = "est autorisé" if "AUTORIS" in etat.upper() else f"a l'état : {etat}"
    substance_phrase = ""
    if product["substances"]:
        substance_phrase = f", contenant {', '.join(product['substances'])},"
    culture_phrase = f"sur {culture}" if culture else ""
    nuisible_phrase = f"contre {nuisible}" if nuisible else ""
    methode_phrase = f"par {methode}" if methode else ""
    gamme_phrase = f"Usage {gamme.lower()}." if gamme else ""

    intro = (
        f"Le produit {nom} (AMM {amm}){substance_phrase} {etat_phrase} "
        f"{culture_phrase} {nuisible_phrase} {methode_phrase}. {gamme_phrase}"
    ).strip()

    # ── Bloc structuré avec les valeurs techniques ────────────────────────────
    lines = [
        intro,
        "",  # ligne vide de séparation
        f"Produit : {nom} | AMM : {amm} | État : {etat} | Gamme : {gamme}",
    ]
    if product["substances"]:
        lines.append(f"Substance(s) active(s) : {', '.join(product['substances'])}")
    if product["titulaire"]:
        lines.append(f"Titulaire : {product['titulaire']}")

    lines.append(
        f"Culture : {culture} | Nuisible : {nuisible} | Méthode : {methode}"
    )
    etat_usage = usage.get("etat_usage", "")
    if etat_usage:
        lines.append(f"État de l'usage : {etat_usage}")

    if usage.get("dose"):
        lines.append(f"Dose retenue : {usage['dose']} {usage.get('dose_unite', '')}")
    if usage.get("dar"):
        lines.append(f"Délai avant récolte (DAR) : {usage['dar']} jours")
    if usage.get("nb_apports"):
        lines.append(f"Nombre d'apports maximum : {usage['nb_apports']}")
    if usage.get("znt_aquatique"):
        lines.append(f"ZNT aquatique : {usage['znt_aquatique']} {usage.get('znt_unite', 'm')}")
    if usage.get("znt_arthropodes"):
        lines.append(f"ZNT arthropodes non cibles : {usage['znt_arthropodes']}")
    if usage.get("condition_emploi"):
        lines.append(f"Conditions d'emploi : {usage['condition_emploi'][:300]}")
    if usage.get("date_fin_utilisation"):
        lines.append(f"Date de fin d'utilisation : {usage['date_fin_utilisation']}")

    # Conditions générales du produit (EPI, délai de rentrée, etc.)
    if product["conditions"]:
        resume_conditions = " | ".join(product["conditions"][:3])
        lines.append(f"Conditions générales : {resume_conditions[:400]}")

    return "\n".join(lines)


def _build_product_metadata(product: dict, usage: dict) -> dict:
    usage_label = usage.get("label", "")
    culture, methode, nuisible = _parse_usage_label(usage_label)
    return {
        "source": "amm_xml",
        "type": "amm_usage",
        "numero_amm": product["amm"],
        "nom_produit": product["nom"],
        "etat_produit": product["etat"],
        "gamme": product["gamme"],
        "type_produit": product["type_produit"],
        "substances_actives": ", ".join(product["substances"]),
        "culture": culture,
        "methode": methode,
        "nuisible": nuisible,
        "etat_usage": usage.get("etat_usage", ""),
        "dar_jours": int(usage["dar"]) if usage.get("dar") else None,
        "znt_metres": float(usage["znt_aquatique"]) if usage.get("znt_aquatique") else None,
    }


def parse_amm_xml(xml_path: str) -> Generator[UsageChunk, None, None]:
    """
    Parse le fichier XML AMM en streaming et génère un UsageChunk par usage de produit.
    Pour les produits sans usage, génère un chunk de description générale.
    """
    # État de parsing
    product = {}
    current_usage = {}
    in_ppp = False
    in_usage = False
    in_substance = False
    in_composition = False
    current_tag_stack = []

    for event, elem in ET.iterparse(xml_path, events=["start", "end"]):
        tag = elem.tag

        if event == "start":
            current_tag_stack.append(tag)

            if tag == "PPP":
                in_ppp = True
                product = {
                    "nom": "", "amm": "", "etat": "", "gamme": "",
                    "type_produit": "PPP", "titulaire": "",
                    "substances": [], "conditions": [], "usages_count": 0
                }

            elif tag == "usage" and in_ppp:
                in_usage = True
                current_usage = {}

            elif tag in ("composition-integrale",) and in_ppp:
                in_composition = True

        elif event == "end":
            if tag == "PPP":
                # Produit sans usage → chunk général
                if product.get("usages_count", 0) == 0 and product.get("nom"):
                    text = (
                        f"Produit : {product['nom']} (AMM {product['amm']}) | "
                        f"État : {product['etat']} | Gamme : {product['gamme']}\n"
                        f"Type : {product['type_produit']}\n"
                        f"Substance(s) active(s) : {', '.join(product['substances'])}\n"
                        f"Titulaire : {product['titulaire']}\n"
                        "Ce produit ne comporte pas d'usage spécifique enregistré."
                    )
                    yield UsageChunk(
                        text=text,
                        metadata={
                            "source": "amm_xml",
                            "type": "amm_produit_general",
                            "numero_amm": product["amm"],
                            "nom_produit": product["nom"],
                            "etat_produit": product["etat"],
                            "gamme": product["gamme"],
                            "type_produit": product["type_produit"],
                            "substances_actives": ", ".join(product["substances"]),
                        },
                    )
                in_ppp = False
                product = {}
                elem.clear()

            elif tag == "usage" and in_ppp:
                if current_usage.get("label") and product.get("nom"):
                    text = _build_usage_text(product, current_usage)
                    metadata = _build_product_metadata(product, current_usage)
                    yield UsageChunk(text=text, metadata=metadata)
                    product["usages_count"] = product.get("usages_count", 0) + 1
                in_usage = False
                current_usage = {}
                elem.clear()

            # Champs du produit
            elif tag == "nom-produit" and in_ppp and not in_usage and not in_composition:
                product["nom"] = (elem.text or "").strip()

            elif tag == "numero-AMM" and in_ppp:
                product["amm"] = (elem.text or "").strip()

            elif tag == "etat-produit" and in_ppp:
                product["etat"] = (elem.text or "").strip()

            elif tag == "gamme-usage" and in_ppp:
                product["gamme"] = (elem.text or "").strip()

            elif tag == "type-produit" and in_ppp and not in_usage:
                product["type_produit"] = (elem.text or "").strip()

            elif tag == "titulaire" and in_ppp:
                product["titulaire"] = (elem.text or "").strip()

            # Substances actives (dans composition-integrale)
            elif tag == "substance" and in_ppp and in_composition:
                sa_name = (elem.text or "").strip()
                if sa_name and sa_name not in product["substances"]:
                    product["substances"].append(sa_name)

            elif tag == "composition-integrale":
                in_composition = False

            # Conditions d'emploi du produit
            elif tag == "description" and in_ppp and not in_usage:
                desc = (elem.text or "").strip()
                if desc:
                    product["conditions"].append(desc)

            # Champs d'un usage
            elif tag == "identifiant-usage" and in_usage:
                current_usage["label"] = (elem.text or "").strip()

            elif tag == "etat-usage" and in_usage:
                current_usage["etat_usage"] = (elem.text or "").strip()

            elif tag == "dose-retenue" and in_usage:
                current_usage["dose"] = (elem.text or "").strip()
                current_usage["dose_unite"] = elem.get("unite", "")

            elif tag == "delai-avant-recolte-jour" and in_usage:
                current_usage["dar"] = (elem.text or "").strip()

            elif tag == "nombre-apport-max" and in_usage:
                current_usage["nb_apports"] = (elem.text or "").strip()

            elif tag == "condition-emploi" and in_usage:
                current_usage["condition_emploi"] = (elem.text or "").strip()

            elif tag == "ZNT-aquatique" and in_usage:
                current_usage["znt_aquatique"] = (elem.text or "").strip()
                current_usage["znt_unite"] = elem.get("unite", "m")

            elif tag == "ZNT-arthropodes-non-cibles" and in_usage:
                current_usage["znt_arthropodes"] = (elem.text or "").strip()

            elif tag == "date-fin-utilisation" and in_usage:
                current_usage["date_fin_utilisation"] = (elem.text or "").strip()

            if current_tag_stack:
                current_tag_stack.pop()
