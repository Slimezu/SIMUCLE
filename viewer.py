import json
import re
import math
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from urllib.parse import quote

# Load periodic table data
URL = "https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json"
try:
    data = requests.get(URL).json()
    ELEMENTS = {el["number"]: el for el in data["elements"]}
except:
    ELEMENTS = {}  # fallback if network fails
    print("Failed to load data. Using minimal fallback.")

# Reverse lookup: element symbol -> atomic number (used by the compound/fusion system)
SYMBOL_TO_Z = {el["symbol"]: z for z, el in ELEMENTS.items()}
SYMBOL_TO_Z.setdefault("D", 1)  # deuterium (²H) - same proton count as hydrogen
SYMBOL_TO_Z.setdefault("T", 1)  # tritium (³H)


# ---------------------------------------------------------------------------
# Compound system
#
# PubChem (the NIH's public chemistry database) holds 100M+ compound records.
# No desktop app can ship a dropdown with all of them, so instead of faking
# completeness with a hardcoded list, this app talks to PubChem directly:
# search by name, get back real candidates, pick the right one. That's the
# only honest way to give access to "every discovered molecule."
# ---------------------------------------------------------------------------

_ELEMENT_TOKEN = re.compile(r'[A-Z][a-z]?')


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula (handles nested parentheses, e.g. 'Ca(OH)2',
    '(NH4)2SO4') into a dict of {element_symbol: count}.
    """
    formula = formula.replace(" ", "").replace("·", "").replace(".", "")  # strip hydrate dots etc.

    def read_number(s, i):
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        return (i, 1) if j == i else (j, int(s[i:j]))

    def parse_group(s, i):
        counts: Dict[str, int] = {}
        while i < len(s) and s[i] != ')':
            if s[i] == '(':
                sub_counts, i = parse_group(s, i + 1)
                i += 1  # skip ')'
                i, mult = read_number(s, i)
                for el, c in sub_counts.items():
                    counts[el] = counts.get(el, 0) + c * mult
            else:
                m = _ELEMENT_TOKEN.match(s, i)
                if not m:
                    raise ValueError(f"Could not parse formula '{formula}' near position {i}")
                el = m.group(0)
                i = m.end()
                i, mult = read_number(s, i)
                counts[el] = counts.get(el, 0) + mult
        return counts, i

    counts, end = parse_group(formula, 0)
    if end != len(formula):
        raise ValueError(f"Unbalanced parentheses in formula '{formula}'")
    return counts


def compute_molar_mass(atoms: Dict[str, int]):
    total = 0.0
    for symbol, count in atoms.items():
        z = SYMBOL_TO_Z.get(symbol)
        mass = ELEMENTS.get(z, {}).get("atomic_mass") if z else None
        if mass is None:
            return None
        total += mass * count
    return total


@dataclass
class Compound:
    name: str
    formula: str
    atoms: Dict[str, int]
    molar_mass: Optional[float] = None
    source: str = "built-in"

    @property
    def total_protons(self) -> int:
        """Sum of atomic numbers across every atom in the molecule - the
        simplified 'total nuclear charge' the particle accelerator uses when
        a compound is fired as a projectile."""
        total = 0
        for symbol, count in self.atoms.items():
            z = SYMBOL_TO_Z.get(symbol)
            if z is None:
                raise ValueError(f"Unknown element symbol '{symbol}' in formula '{self.formula}'")
            total += z * count
        return total

    @property
    def total_mass_number(self) -> int:
        total = 0
        for symbol, count in self.atoms.items():
            z = SYMBOL_TO_Z.get(symbol)
            mass = ELEMENTS.get(z, {}).get("atomic_mass") if z else None
            a = round(mass) if mass else round((z or 1) * 2.5)
            total += a * count
        return total

    @property
    def atom_count(self) -> int:
        return sum(self.atoms.values())

    @property
    def label(self) -> str:
        return f"{self.name} ({self.formula})"


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def pubchem_search_by_name(query: str, max_results: int = 8):
    """
    Search PubChem's live database by name and return matching Compound
    candidates (there can be more than one for an ambiguous name). Requires
    an internet connection. This is the real "every discovered molecule"
    feature - it talks to the actual database instead of a static list.
    """
    name_url = f"{PUBCHEM_BASE}/compound/name/{quote(query)}/cids/JSON"
    resp = requests.get(name_url, timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"PubChem found no compound matching '{query}'.")
    cids = resp.json().get("IdentifierList", {}).get("CID", [])[:max_results]
    if not cids:
        raise ValueError(f"PubChem found no compound matching '{query}'.")

    cid_list = ",".join(str(c) for c in cids)
    prop_url = (f"{PUBCHEM_BASE}/compound/cid/{cid_list}/property/"
                f"MolecularFormula,MolecularWeight,IUPACName/JSON")
    prop_resp = requests.get(prop_url, timeout=10)
    prop_resp.raise_for_status()
    rows = prop_resp.json()["PropertyTable"]["Properties"]

    candidates = []
    for row in rows:
        formula = row.get("MolecularFormula", "")
        try:
            atoms = parse_formula(formula)
        except Exception:
            continue  # skip anything our simple parser can't handle
        molar_mass = float(row["MolecularWeight"]) if row.get("MolecularWeight") else None
        name = (row.get("IUPACName") or query).strip()
        candidates.append(Compound(
            name=(name.title() if name else query.title()),
            formula=formula,
            atoms=atoms,
            molar_mass=molar_mass,
            source=f"PubChem CID {row.get('CID')}",
        ))
    if not candidates:
        raise ValueError(f"PubChem found matches for '{query}' but none had a parseable formula.")
    return candidates


# A small starter pack so the app has something to show before any search -
# NOT meant as "the library." Search PubChem above for anything else.
_STARTER_COMPOUND_DEFS = [
    ("Water", "H2O"),
    ("Carbon Dioxide", "CO2"),
    ("Oxygen Gas", "O2"),
    ("Nitrogen Gas", "N2"),
    ("Methane", "CH4"),
    ("Ammonia", "NH3"),
    ("Sodium Chloride", "NaCl"),
    ("Glucose", "C6H12O6"),
    ("Ethanol", "C2H6O"),
    ("Sulfuric Acid", "H2SO4"),
    ("Calcium Carbonate", "CaCO3"),
    ("Carbon Monoxide", "CO"),
    ("Hydrogen Peroxide", "H2O2"),
    ("Sodium Hydroxide", "NaOH"),
    ("Hydrochloric Acid", "HCl"),
]


def _build_starter_library() -> Dict[str, Compound]:
    library = {}
    for name, formula in _STARTER_COMPOUND_DEFS:
        try:
            atoms = parse_formula(formula)
            compound = Compound(name=name, formula=formula, atoms=atoms,
                                 molar_mass=compute_molar_mass(atoms), source="starter pack")
            library[compound.label] = compound
        except Exception as ex:
            print(f"Skipping starter compound '{name}' ({formula}) - {ex}")
    return library


# Session-wide compound library: starts with the starter pack, grows as the
# user searches PubChem. Shared by the Compound Creator and the Particle
# Accelerator's compound dropdown.
COMPOUND_LIBRARY: Dict[str, Compound] = _build_starter_library()

# Highest atomic number with an officially IUPAC-ratified name/symbol (Oganesson, 2016).
# Anything above this does not have a permanent name yet, so we fall back to the
# systematic IUPAC naming convention below.
HIGHEST_NAMED_ELEMENT = 118

# IUPAC systematic element naming (used for undiscovered/unnamed elements, Z > 118)
# Each digit of the atomic number maps to a numerical root.
_DIGIT_ROOTS = {
    "0": "nil",
    "1": "un",
    "2": "bi",
    "3": "tri",
    "4": "quad",
    "5": "pent",
    "6": "hex",
    "7": "sept",
    "8": "oct",
    "9": "enn",
}
_DIGIT_ABBR = {
    "0": "n",
    "1": "u",
    "2": "b",
    "3": "t",
    "4": "q",
    "5": "p",
    "6": "h",
    "7": "s",
    "8": "o",
    "9": "e",
}


def systematic_name_and_symbol(z: int):
    """
    Generate the IUPAC systematic name and symbol for an atomic number.
    E.g. 119 -> ('Ununennium', 'Uue'), 120 -> ('Unbinilium', 'Ubn')
    Follows IUPAC recommendations: digit roots are concatenated and an
    '-ium' ending is appended, with elision of a double 'i' down to a
    single 'i' (e.g. 'bi' + 'ium' -> 'bium').
    """
    digits = str(z)
    roots = [_DIGIT_ROOTS[d] for d in digits]
    stem = "".join(roots)
    name = stem + "ium"
    name = name.replace("iiu", "iu")  # elision for roots ending in "i" (bi, tri, etc.)
    name = name.capitalize()
    symbol = "".join(_DIGIT_ABBR[d] for d in digits).capitalize()
    return name, symbol


def get_element_identity(z: int):
    """
    Returns (name, symbol) for any atomic number, with no upper limit.
    Uses the real/official name if it's loaded in ELEMENTS, otherwise
    generates the IUPAC systematic placeholder name/symbol.
    """
    if z in ELEMENTS:
        el = ELEMENTS[z]
        return el["name"], el["symbol"]
    if z <= HIGHEST_NAMED_ELEMENT:
        # Should be a known element but our data source failed to load -
        # be honest about that rather than guessing.
        return f"Unknown (Z={z})", "?"
    return systematic_name_and_symbol(z)


@dataclass
class Atom:
    atomic_number: int
    electrons: int
    name: str
    symbol: str
    data: Dict[str, Any]

    @property
    def is_ion(self):
        return self.electrons != self.atomic_number

    @property
    def is_systematic_name(self):
        return self.atomic_number > HIGHEST_NAMED_ELEMENT or self.atomic_number not in ELEMENTS


def get_atom(z: int, custom_electrons: int = None) -> Atom:
    if z < 1:
        raise ValueError("Atomic number must be at least 1")
    name, symbol = get_element_identity(z)
    el = ELEMENTS.get(z, {})  # may be empty for unnamed/superheavy elements
    electrons = custom_electrons if custom_electrons is not None else z
    return Atom(z, electrons, name, symbol, el)


def estimate_shells(z: int):
    """
    Fallback electron shell estimate (2n^2 filling, capped at typical
    chemistry-class capacities) for atoms that have no shell data in
    the source JSON - i.e. anything beyond Z=118 or any custom/hypothetical atom.
    """
    capacities = [2, 8, 18, 32, 32, 18, 8, 2]
    shells = []
    remaining = z
    for cap in capacities:
        if remaining <= 0:
            break
        take = min(cap, remaining)
        shells.append(take)
        remaining -= take
    if remaining > 0:
        shells.append(remaining)  # overflow bucket for very large hypothetical atoms
    return shells


# ---------------------------------------------------------------------------
# Particle Accelerator: states of matter, beam speed defaults
# ---------------------------------------------------------------------------

# A representative set of recognized states/phases of matter - classical and
# exotic/quantum. New exotic phases get proposed in physics papers
# fairly often, so "every state of matter ever discovered" is a moving
# target, but this covers the well-established ones.
STATES_OF_MATTER = [
    "(unspecified)",
    "Solid",
    "Liquid",
    "Gas",
    "Plasma",
    "Bose–Einstein Condensate",
    "Fermionic Condensate",
    "Degenerate Matter",
    "Quark–Gluon Plasma",
    "Quark Matter (Strange Matter)",
    "Superfluid",
    "Supersolid",
    "Liquid Crystal",
    "Amorphous Solid (Glass)",
    "Plasma Crystal",
    "Photonic Matter",
    "Rydberg Matter",
    "Neutron-Degenerate Matter",
    "Electron-Degenerate Matter",
    "Supercritical Fluid",
    "Quantum Spin Liquid",
]

SPEED_OF_LIGHT_MPS = 299_792_458


def default_beam_speed_fraction_c(atomic_number: int) -> float:
    """
    A representative beam speed (as a fraction of c) for a projectile of the
    given atomic number, loosely modeled on real accelerator practice:
    light particles (protons, alphas) are routinely run at relativistic
    speeds, while the heavy-ion beams used for superheavy-element fusion
    (e.g. the 48-Ca beams at JINR Dubna) run around ~10% c - enough to clear
    the Coulomb barrier without needing to be relativistic. This is a
    simplified average for flavor, not a literal accelerator spec.
    """
    if atomic_number <= 2:
        return 0.85
    elif atomic_number <= 10:
        return 0.5
    elif atomic_number <= 30:
        return 0.2
    else:
        return 0.1


# Coulomb-barrier-style speed requirement, calibrated so that the real
# 48-Ca + 249-Cf -> Oganesson reaction (Z1*Z2 = 1960) just clears the bar at
# the default beam speeds above (~0.1c). Heavier pairs need more speed;
# lighter pairs need less. This is a simplified flavor heuristic, not a
# rigorous nuclear-physics calculation.
COULOMB_CONST = 0.00226


@dataclass
class Projectile:
    label: str
    z_total: int        # total protons contributed (sum of Z for compounds)
    a_total: int         # approximate total mass number contributed
    amount: int          # how many atoms/molecules are being fired
    kind: str = "element"
    state: Optional[str] = None
    temperature_k: Optional[float] = None
    speed_fraction_c: Optional[float] = None


# Known, publicly documented superheavy-element discovery reactions (all
# real history - JINR Dubna, GSI Darmstadt, RIKEN). Keyed by sorted (Z1, Z2).
# This is standard nuclear-chemistry curriculum content (see e.g. the
# Wikipedia articles on each element / "Island of stability") - none of
# these elements have any practical or weapons application; most have
# half-lives measured in milliseconds and are produced one atom at a time.
KNOWN_FUSION_REACTIONS = {
    (20, 94): {"product_z": 114, "beam_a": 48, "target_a": 244, "neutrons": 4,
               "label": "⁴⁸Ca + ²⁴⁴Pu → Flerovium-288 + 4n"},
    (20, 95): {"product_z": 115, "beam_a": 48, "target_a": 243, "neutrons": 3,
               "label": "⁴⁸Ca + ²⁴³Am → Moscovium-288 + 3n"},
    (20, 96): {"product_z": 116, "beam_a": 48, "target_a": 248, "neutrons": 4,
               "label": "⁴⁸Ca + ²⁴⁸Cm → Livermorium-292 + 4n"},
    (20, 97): {"product_z": 117, "beam_a": 48, "target_a": 249, "neutrons": 4,
               "label": "⁴⁸Ca + ²⁴⁹Bk → Tennessine-293 + 4n"},
    (20, 98): {"product_z": 118, "beam_a": 48, "target_a": 249, "neutrons": 3,
               "label": "⁴⁸Ca + ²⁴⁹Cf → Oganesson-294 + 3n"},
    (24, 83): {"product_z": 107, "beam_a": 54, "target_a": 209, "neutrons": 1,
               "label": "⁵⁴Cr + ²⁰⁹Bi → Bohrium-262 + n"},
    (26, 82): {"product_z": 108, "beam_a": 58, "target_a": 208, "neutrons": 1,
               "label": "⁵⁸Fe + ²⁰⁸Pb → Hassium-265 + n"},
    (26, 83): {"product_z": 109, "beam_a": 58, "target_a": 209, "neutrons": 1,
               "label": "⁵⁸Fe + ²⁰⁹Bi → Meitnerium-266 + n"},
    (28, 82): {"product_z": 110, "beam_a": 62, "target_a": 208, "neutrons": 1,
               "label": "⁶²Ni + ²⁰⁸Pb → Darmstadtium-269 + n"},
    (28, 83): {"product_z": 111, "beam_a": 64, "target_a": 209, "neutrons": 1,
               "label": "⁶⁴Ni + ²⁰⁹Bi → Roentgenium-272 + n"},
    (30, 82): {"product_z": 112, "beam_a": 70, "target_a": 208, "neutrons": 1,
               "label": "⁷⁰Zn + ²⁰⁸Pb → Copernicium-277 + n"},
    (30, 83): {"product_z": 113, "beam_a": 70, "target_a": 209, "neutrons": 1,
               "label": "⁷⁰Zn + ²⁰⁹Bi → Nihonium-278 + n"},
}


def simulate_fusion(p1: Projectile, p2: Projectile) -> Dict[str, Any]:
    """
    Combine two projectiles (elements and/or compounds) into a new nucleus.
    Recognizes real historical synthesis reactions by matching proton counts;
    otherwise falls back to a simplified general fusion model (sum protons,
    evaporate a few neutrons). Includes a Coulomb-barrier-style speed check.
    """
    z1, z2 = p1.z_total, p2.z_total
    a1, a2 = p1.a_total, p2.a_total
    key = tuple(sorted((z1, z2)))

    speed1 = p1.speed_fraction_c if p1.speed_fraction_c is not None else default_beam_speed_fraction_c(z1)
    speed2 = p2.speed_fraction_c if p2.speed_fraction_c is not None else default_beam_speed_fraction_c(z2)
    combined_speed = max(speed1, speed2)

    threshold_speed = COULOMB_CONST * math.sqrt(max(z1 * z2, 1))
    fused = combined_speed >= threshold_speed

    matched = key in KNOWN_FUSION_REACTIONS
    if matched:
        reaction = KNOWN_FUSION_REACTIONS[key]
        product_z = reaction["product_z"]
        product_a = reaction["beam_a"] + reaction["target_a"] - reaction["neutrons"]
        neutrons = reaction["neutrons"]
        reaction_label = reaction["label"]
    else:
        product_z = z1 + z2
        neutrons = max(1, round((a1 + a2) * 0.02))
        product_a = max(product_z, a1 + a2 - neutrons)
        reaction_label = None

    product_name, product_symbol = get_element_identity(product_z)

    yield_estimate = 0
    if fused:
        base_pairs = min(p1.amount, p2.amount)
        severity = max(1, (z1 + z2) // 10)
        yield_estimate = max(0, base_pairs // (severity ** 2))

    return {
        "fused": fused,
        "combined_speed_c": combined_speed,
        "threshold_speed_c": threshold_speed,
        "matched_known_reaction": matched,
        "reaction_label": reaction_label,
        "product_z": product_z,
        "product_a": product_a,
        "product_name": product_name,
        "product_symbol": product_symbol,
        "neutrons_emitted": neutrons,
        "yield_estimate": yield_estimate,
    }


def display_info(atom: Atom):
    el = atom.data
    half_life = el.get("half_life") or ("Stable" if atom.atomic_number < 84 or atom.atomic_number in [90, 92] else "Radioactive (very short)")
    naming_note = ""
    if atom.is_systematic_name:
        naming_note = "\n(Note: this element has no permanent IUPAC name yet, so a systematic placeholder name is shown.)"
    summary = el.get('summary')
    summary_text = (summary[:500] + "...") if summary else "N/A (no data available for this element)"
    info = f"""
Element: {atom.name} ({atom.symbol}){naming_note}
Atomic Number (Z): {atom.atomic_number}
Electrons: {atom.electrons} {'(Ion)' if atom.is_ion else ''}
Atomic Mass: {el.get('atomic_mass', 'N/A')}
Electron Configuration: {el.get('electron_configuration', 'N/A')}
Shells: {el.get('shells', estimate_shells(atom.atomic_number))}
Electronegativity: {el.get('electronegativity_pauling', 'N/A')}
Atomic Radius (pm): {el.get('atomic_radius', 'N/A')}
Phase: {el.get('phase', 'N/A')}
Density: {el.get('density', 'N/A')} g/cm³
Melting Point (K): {el.get('melt', 'N/A')}
Boiling Point (K): {el.get('boil', 'N/A')}
Half-life: {half_life}
Summary: {summary_text}
    """
    print(info)
    return info


# ---------------------------------------------------------------------------
# 3D Visualization (beautified)
# ---------------------------------------------------------------------------

SHELL_COLORS = [
    "#00e5ff",  # shell 1 - cyan
    "#7c4dff",  # shell 2 - violet
    "#ff4081",  # shell 3 - pink
    "#ffd740",  # shell 4 - amber
    "#69f0ae",  # shell 5 - green
    "#ff6e40",  # shell 6 - orange
    "#40c4ff",  # shell 7 - blue
    "#e040fb",  # shell 8 - magenta
]


def visualize_atom(atom: Atom):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9), facecolor="#0a0a14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a14")

    naming_tag = " (systematic name)" if atom.is_systematic_name else ""
    fig.suptitle(
        f"{atom.name} ({atom.symbol}){naming_tag}",
        color="#f5f5f5", fontsize=20, fontweight="bold", y=0.97,
    )
    ax.set_title(
        f"Z = {atom.atomic_number}   •   e\u207b = {atom.electrons}"
        f"{'   •   ION' if atom.is_ion else ''}",
        color="#9aa0b4", fontsize=12, pad=14,
    )

    # Subtle starfield background for atmosphere
    rng = np.random.default_rng(atom.atomic_number)
    star_r = rng.uniform(8, 22, 120)
    star_theta = rng.uniform(0, np.pi, 120)
    star_phi = rng.uniform(0, 2 * np.pi, 120)
    ax.scatter(
        star_r * np.sin(star_theta) * np.cos(star_phi),
        star_r * np.sin(star_theta) * np.sin(star_phi),
        star_r * np.cos(star_theta),
        s=2, color="white", alpha=0.15, depthshade=False,
    )

    # --- Nucleus: layered glow effect ---
    nucleus_glow_sizes = [1400, 900, 500, 220]
    nucleus_glow_alphas = [0.08, 0.14, 0.22, 1.0]
    nucleus_colors = ["#ff8a65", "#ff7043", "#ff5722", "#ffab91"]
    for s, a, c in zip(nucleus_glow_sizes, nucleus_glow_alphas, nucleus_colors):
        ax.scatter([0], [0], [0], color=c, s=s, alpha=a, edgecolors="none", depthshade=False)
    ax.text(0, 0, 0.55, "nucleus", color="#ffccbc", fontsize=8, ha="center", style="italic")

    shells = atom.data.get("shells") or estimate_shells(atom.atomic_number)
    if not shells:
        shells = [atom.electrons]

    placed = 0
    for n, capacity in enumerate(shells, 1):
        radius = n * 1.9
        remaining = max(atom.electrons - placed, 0)
        num_e = min(capacity, remaining)
        placed += num_e
        color = SHELL_COLORS[(n - 1) % len(SHELL_COLORS)]

        # Orbit ring (drawn as a smooth great-circle-ish ellipse, slightly tilted per shell)
        u = np.linspace(0, 2 * np.pi, 120)
        tilt = (n - 1) * 0.35
        x_ring = radius * np.cos(u)
        y_ring = radius * np.sin(u) * np.cos(tilt)
        z_ring = radius * np.sin(u) * np.sin(tilt)
        ax.plot(x_ring, y_ring, z_ring, color=color, alpha=0.35, linewidth=1.3)

        if num_e <= 0:
            continue

        # Electron cloud: evenly distributed points around the (tilted) shell
        theta = np.linspace(0, 2 * np.pi, num_e, endpoint=False)
        jitter = rng.uniform(-0.06, 0.06, num_e)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta) * np.cos(tilt) + jitter
        z = radius * np.sin(theta) * np.sin(tilt) + jitter

        # Soft glow halo beneath each electron, then a bright core on top
        ax.scatter(x, y, z, color=color, s=160, alpha=0.18, edgecolors="none", depthshade=False)
        ax.scatter(
            x, y, z, color=color, s=42, alpha=0.95,
            edgecolors="white", linewidths=0.3, depthshade=False,
            label=f"Shell {n}  ({num_e} e\u207b)",
        )

    limit = max(len(shells) * 1.9 + 2, 4)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    # Clean, minimal axes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color("#33334d")
        axis._axinfo["grid"]["color"] = (0.15, 0.15, 0.25, 0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(colors="#33334d")

    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=9,
        facecolor="#14142a", edgecolor="#33334d", labelcolor="#e8e8f0",
        framealpha=0.85,
    )

    fig.text(
        0.5, 0.015,
        "drag to rotate  •  scroll to zoom",
        ha="center", color="#5a5a72", fontsize=9, style="italic",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    plt.show()


# ---------------------------------------------------------------------------
# GUI (beautified)
# ---------------------------------------------------------------------------

BG = "#0f0f1a"
PANEL = "#181830"
ACCENT = "#00e5ff"
ACCENT_DIM = "#0a8fa3"
TEXT = "#e8e8f0"
SUBTEXT = "#8a8aa0"
ENTRY_BG = "#22223d"
DANGER = "#ff5252"


class AtomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚛  Atomic Simulator")
        self.root.configure(bg=BG)
        self.root.geometry("440x560")
        self.root.resizable(False, False)

        self._build_style()
        self._build_layout()

    # -- styling -----------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)

        style.configure(
            "Title.TLabel", background=BG, foreground=TEXT,
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel", background=BG, foreground=SUBTEXT,
            font=("Segoe UI", 9),
        )
        style.configure(
            "FieldLabel.TLabel", background=PANEL, foreground=SUBTEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Hint.TLabel", background=PANEL, foreground=SUBTEXT,
            font=("Segoe UI", 8),
        )

        style.configure(
            "TEntry", fieldbackground=ENTRY_BG, foreground=TEXT,
            insertcolor=TEXT, borderwidth=0, relief="flat",
            padding=8,
        )
        style.map("TEntry", fieldbackground=[("focus", ENTRY_BG)])

        style.configure(
            "Accent.TButton", font=("Segoe UI", 10, "bold"),
            foreground="#001014", background=ACCENT,
            borderwidth=0, padding=(14, 10), relief="flat",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#33eeff"), ("pressed", ACCENT_DIM)],
        )

        style.configure(
            "Ghost.TButton", font=("Segoe UI", 10),
            foreground=TEXT, background=PANEL,
            borderwidth=1, padding=(14, 10), relief="flat",
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#26264a")],
            bordercolor=[("!disabled", "#33334d")],
        )

    # -- layout --------------------------------------------------------------
    def _build_layout(self):
        outer = ttk.Frame(self.root, style="TFrame", padding=24)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="⚛ Atomic Simulator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Explore real elements or build a hypothetical atom — any atomic number works.",
            style="Subtitle.TLabel", wraplength=380, justify="left",
        ).pack(anchor="w", pady=(2, 18))

        panel = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        panel.pack(fill="x")

        ttk.Label(panel, text="ATOMIC NUMBER (Z)", style="FieldLabel.TLabel").pack(anchor="w")
        self.z_var = tk.IntVar(value=1)
        z_entry = ttk.Entry(panel, textvariable=self.z_var, font=("Segoe UI", 12))
        z_entry.pack(fill="x", pady=(6, 2))
        ttk.Label(
            panel, text="Try 119+ for systematically-named superheavy elements",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        ttk.Label(panel, text="CUSTOM ELECTRON COUNT (optional)", style="FieldLabel.TLabel").pack(anchor="w")
        self.e_var = tk.IntVar()
        e_entry = ttk.Entry(panel, textvariable=self.e_var, font=("Segoe UI", 12))
        e_entry.pack(fill="x", pady=(6, 2))
        ttk.Label(
            panel, text="Leave blank for a neutral atom — set this to model an ion",
            style="Hint.TLabel",
        ).pack(anchor="w")

        btn_row = ttk.Frame(outer, style="TFrame")
        btn_row.pack(fill="x", pady=(20, 8))
        ttk.Button(
            btn_row, text="✨  Load & Visualize", style="Accent.TButton",
            command=self.load_atom,
        ).pack(fill="x")

        ttk.Button(
            outer, text="🧪  Build a Custom Hypothetical Atom", style="Ghost.TButton",
            command=self.custom_atom,
        ).pack(fill="x", pady=(8, 0))

        ttk.Button(
            outer, text="🧬  Compound Creator", style="Ghost.TButton",
            command=self.open_compound_creator,
        ).pack(fill="x", pady=(8, 0))

        ttk.Button(
            outer, text="💥  Particle Accelerator", style="Ghost.TButton",
            command=self.open_accelerator,
        ).pack(fill="x", pady=(8, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            outer, textvariable=self.status_var, style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(18, 0))

    # -- actions -------------------------------------------------------------
    def load_atom(self):
        try:
            z = self.z_var.get()
            e = self.e_var.get() if self.e_var.get() > 0 else None
            atom = get_atom(z, e)
            self.status_var.set(f"Loaded {atom.name} ({atom.symbol}) — Z={atom.atomic_number}")
            info = display_info(atom)
            messagebox.showinfo("Element Data", info)
            visualize_atom(atom)
        except Exception as ex:
            self.status_var.set("Error — see dialog.")
            messagebox.showerror("Error", str(ex))

    def custom_atom(self):
        z = simpledialog.askinteger("Custom Atom", "Atomic number (protons):", minvalue=1, maxvalue=10000)
        if z:
            e = simpledialog.askinteger("Custom Atom", "Number of electrons:", minvalue=0, maxvalue=10000)
            if e is not None:
                name, symbol = get_element_identity(z)
                atom = Atom(z, e, name, symbol, {"shells": estimate_shells(z)})
                self.status_var.set(f"Built hypothetical {name} ({symbol}) — Z={z}")
                display_info(atom)
                visualize_atom(atom)

    def open_compound_creator(self):
        CompoundCreatorWindow(self.root)

    def open_accelerator(self):
        ParticleAcceleratorWindow(self.root)


# ---------------------------------------------------------------------------
# Compound Creator window
# ---------------------------------------------------------------------------

class CompoundCreatorWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.transient(master)
        self.title("🧬 Compound Creator")
        self.configure(bg=BG)
        self.geometry("500x560")
        self.resizable(False, False)

        outer = ttk.Frame(self, style="TFrame", padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="🧬 Compound Creator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=("Searches PubChem's live database directly — 100M+ compounds, "
                  "not a fixed list. Type a name below and hit Search."),
            style="Subtitle.TLabel", wraplength=440, justify="left",
        ).pack(anchor="w", pady=(2, 16))

        search_panel = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        search_panel.pack(fill="x")

        ttk.Label(search_panel, text="SEARCH PUBCHEM (requires internet)", style="FieldLabel.TLabel").pack(anchor="w")
        search_row = ttk.Frame(search_panel, style="Panel.TFrame")
        search_row.pack(fill="x", pady=(6, 4))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_row, textvariable=self.search_var, font=("Segoe UI", 11))
        search_entry.pack(side="left", fill="x", expand=True)
        search_entry.bind("<Return>", lambda e: self.search_pubchem())
        ttk.Button(search_row, text="🔍 Search", style="Accent.TButton", command=self.search_pubchem).pack(side="left", padx=(8, 0))
        ttk.Label(search_panel, text="e.g. 'caffeine', 'ibuprofen', 'sodium benzoate', or any formula",
                  style="Hint.TLabel").pack(anchor="w")

        ttk.Label(outer, text="MATCHES (pick one — names can be ambiguous)", style="FieldLabel.TLabel").pack(anchor="w", pady=(16, 4))
        list_frame = ttk.Frame(outer, style="Panel.TFrame", padding=4)
        list_frame.pack(fill="x")
        self.results_list = tk.Listbox(
            list_frame, height=5, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 10),
            relief="flat", selectbackground=ACCENT, selectforeground="#001014",
            highlightthickness=0, activestyle="none",
        )
        self.results_list.pack(fill="x")
        self.results_list.bind("<<ListboxSelect>>", self._on_result_select)
        self._search_results = []

        ttk.Label(outer, text="SAVED THIS SESSION", style="FieldLabel.TLabel").pack(anchor="w", pady=(16, 4))
        self.saved_var = tk.StringVar()
        self.saved_box = ttk.Combobox(
            outer, textvariable=self.saved_var, state="readonly",
            values=sorted(COMPOUND_LIBRARY.keys()), font=("Segoe UI", 10),
        )
        self.saved_box.pack(fill="x")
        self.saved_box.bind("<<ComboboxSelected>>", lambda e: self._show_detail(COMPOUND_LIBRARY.get(self.saved_var.get())))

        self.detail_text = tk.Text(
            outer, height=10, bg=ENTRY_BG, fg=TEXT, font=("Consolas", 10),
            relief="flat", padx=10, pady=10, wrap="word",
        )
        self.detail_text.pack(fill="both", expand=True, pady=(16, 0))
        self._set_detail_text("Search above, or pick something from Saved This Session, to see details here.")

    def _set_detail_text(self, text):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _show_detail(self, compound: Optional["Compound"]):
        if not compound:
            return
        atoms_str = ", ".join(f"{c}×{s}" for s, c in compound.atoms.items())
        mass_str = f"{compound.molar_mass:.3f} g/mol" if compound.molar_mass else "N/A"
        text = (
            f"Name: {compound.name}\n"
            f"Formula: {compound.formula}\n"
            f"Source: {compound.source}\n"
            f"Atoms: {atoms_str}\n"
            f"Total atoms in molecule: {compound.atom_count}\n"
            f"Total protons (Z, summed): {compound.total_protons}\n"
            f"Molar mass: {mass_str}\n"
            f"\nSaved as '{compound.label}' — available in the Particle Accelerator's compound dropdown."
        )
        self._set_detail_text(text)

    def search_pubchem(self):
        query = self.search_var.get().strip()
        if not query:
            return
        self.results_list.delete(0, "end")
        self._search_results = []
        try:
            candidates = pubchem_search_by_name(query)
        except Exception as ex:
            messagebox.showerror("PubChem Search Failed", str(ex))
            return
        self._search_results = candidates
        for c in candidates:
            self.results_list.insert("end", f"{c.label}  [{c.source}]")
        if len(candidates) == 1:
            self.results_list.selection_set(0)
            self._on_result_select(None)

    def _on_result_select(self, event):
        sel = self.results_list.curselection()
        if not sel:
            return
        compound = self._search_results[sel[0]]
        COMPOUND_LIBRARY[compound.label] = compound
        self.saved_box.configure(values=sorted(COMPOUND_LIBRARY.keys()))
        self.saved_var.set(compound.label)
        self._show_detail(compound)


# ---------------------------------------------------------------------------
# Particle Accelerator window
# ---------------------------------------------------------------------------

class ProjectilePanel(ttk.Frame):
    """One side of the collision: element or compound, amount, and optional physics."""

    def __init__(self, master, heading):
        super().__init__(master, style="Panel.TFrame", padding=16)
        ttk.Label(self, text=heading, style="FieldLabel.TLabel").pack(anchor="w")

        self.kind_var = tk.StringVar(value="element")
        toggle_row = ttk.Frame(self, style="Panel.TFrame")
        toggle_row.pack(fill="x", pady=(8, 10))
        ttk.Radiobutton(toggle_row, text="Element", value="element", variable=self.kind_var,
                        command=self._on_kind_change).pack(side="left")
        ttk.Radiobutton(toggle_row, text="Compound", value="compound", variable=self.kind_var,
                        command=self._on_kind_change).pack(side="left", padx=(16, 0))

        # Element dropdown (named elements, 1-118) -----------------------------
        self.element_label_to_z = {}
        for z in range(1, HIGHEST_NAMED_ELEMENT + 1):
            name, symbol = get_element_identity(z)
            self.element_label_to_z[f"{z} — {name} ({symbol})"] = z
        self.element_var = tk.StringVar()
        self.element_box = ttk.Combobox(
            self, textvariable=self.element_var, state="readonly",
            values=list(self.element_label_to_z.keys()), font=("Segoe UI", 10),
        )
        self.element_box.pack(fill="x")

        ttk.Label(
            self, text="...or type any atomic number (overrides dropdown — works for hypothetical elements too):",
            style="Hint.TLabel", wraplength=300,
        ).pack(anchor="w", pady=(6, 2))
        self.custom_z_var = tk.StringVar()
        self.custom_z_entry = ttk.Entry(self, textvariable=self.custom_z_var, font=("Segoe UI", 10))
        self.custom_z_entry.pack(fill="x")

        # Compound dropdown -------------------------------------------------------
        self.compound_var = tk.StringVar()
        self.compound_box = ttk.Combobox(
            self, textvariable=self.compound_var, state="readonly",
            values=sorted(COMPOUND_LIBRARY.keys()), font=("Segoe UI", 10),
        )

        # Amount ----------------------------------------------------------------
        ttk.Label(self, text="AMOUNT (atoms/molecules fired)", style="FieldLabel.TLabel").pack(anchor="w", pady=(14, 0))
        self.amount_var = tk.StringVar(value="1")
        ttk.Entry(self, textvariable=self.amount_var, font=("Segoe UI", 10)).pack(fill="x", pady=(6, 0))

        # Optional physical parameters --------------------------------------------
        ttk.Label(self, text="STATE OF MATTER (optional)", style="FieldLabel.TLabel").pack(anchor="w", pady=(14, 0))
        self.state_var = tk.StringVar(value=STATES_OF_MATTER[0])
        ttk.Combobox(self, textvariable=self.state_var, state="readonly",
                     values=STATES_OF_MATTER, font=("Segoe UI", 10)).pack(fill="x", pady=(6, 0))

        ttk.Label(self, text="TEMPERATURE IN KELVIN (optional)", style="FieldLabel.TLabel").pack(anchor="w", pady=(14, 0))
        self.temp_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.temp_var, font=("Segoe UI", 10)).pack(fill="x", pady=(6, 0))

        ttk.Label(
            self, text="SPEED, as a fraction of c (optional — blank = realistic auto default)",
            style="Hint.TLabel", wraplength=300,
        ).pack(anchor="w", pady=(14, 0))
        self.speed_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.speed_var, font=("Segoe UI", 10)).pack(fill="x", pady=(6, 0))

        self._on_kind_change()

    def _on_kind_change(self):
        if self.kind_var.get() == "element":
            self.compound_box.pack_forget()
            self.element_box.pack(fill="x")
            self.custom_z_entry.pack(fill="x")
        else:
            self.element_box.pack_forget()
            self.custom_z_entry.pack_forget()
            self.compound_box.pack(fill="x")

    def refresh_compound_list(self):
        self.compound_box.configure(values=sorted(COMPOUND_LIBRARY.keys()))

    def build_projectile(self) -> Projectile:
        amount_str = self.amount_var.get().strip()
        if not amount_str:
            raise ValueError("Amount is required for both projectiles.")
        amount = int(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be a positive whole number.")

        speed = float(self.speed_var.get().strip()) if self.speed_var.get().strip() else None
        temp = float(self.temp_var.get().strip()) if self.temp_var.get().strip() else None
        state = self.state_var.get() if self.state_var.get() != STATES_OF_MATTER[0] else None

        if self.kind_var.get() == "element":
            if self.custom_z_var.get().strip():
                z = int(self.custom_z_var.get().strip())
            elif self.element_var.get():
                z = self.element_label_to_z[self.element_var.get()]
            else:
                raise ValueError("Pick an element from the dropdown or type a custom atomic number.")
            if z < 1:
                raise ValueError("Atomic number must be at least 1.")
            name, symbol = get_element_identity(z)
            mass_data = ELEMENTS.get(z, {}).get("atomic_mass")
            a = round(mass_data) if mass_data else round(z * 2.5)
            return Projectile(label=f"{name} ({symbol})", z_total=z, a_total=a, amount=amount,
                               kind="element", state=state, temperature_k=temp, speed_fraction_c=speed)
        else:
            label = self.compound_var.get()
            compound = COMPOUND_LIBRARY.get(label)
            if not compound:
                raise ValueError("Pick a compound from the dropdown (add more via Compound Creator).")
            return Projectile(label=compound.label, z_total=compound.total_protons,
                               a_total=compound.total_mass_number, amount=amount,
                               kind="compound", state=state, temperature_k=temp, speed_fraction_c=speed)


class ParticleAcceleratorWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.transient(master)
        self.title("💥 Particle Accelerator")
        self.configure(bg=BG)
        self.geometry("780x800")
        self.resizable(False, False)

        outer = ttk.Frame(self, style="TFrame", padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="💥 Particle Accelerator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=("Pick two projectiles (elements or compounds) and amounts, then fire. Real historical "
                  "syntheses (like ⁴⁸Ca + ²⁴⁹Cf → Oganesson) are recognized automatically; any other "
                  "combination uses a simplified general fusion model."),
            style="Subtitle.TLabel", wraplength=720, justify="left",
        ).pack(anchor="w", pady=(2, 16))

        panels_row = ttk.Frame(outer, style="TFrame")
        panels_row.pack(fill="x")
        self.panel1 = ProjectilePanel(panels_row, "PROJECTILE 1")
        self.panel1.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.panel2 = ProjectilePanel(panels_row, "PROJECTILE 2")
        self.panel2.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Button(outer, text="🔥  Fire!", style="Accent.TButton", command=self.fire).pack(fill="x", pady=(16, 8))

        self.result_text = tk.Text(
            outer, height=11, bg=ENTRY_BG, fg=TEXT, font=("Consolas", 10),
            relief="flat", padx=10, pady=10, wrap="word",
        )
        self.result_text.pack(fill="both", expand=True)
        self._set_result_text("Set up both projectiles above and hit Fire to run the collision.")

        # Keep compound dropdowns in sync if compounds are added via Compound Creator
        self.bind("<FocusIn>", lambda e: (self.panel1.refresh_compound_list(), self.panel2.refresh_compound_list()))

    def _set_result_text(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def fire(self):
        try:
            p1 = self.panel1.build_projectile()
            p2 = self.panel2.build_projectile()
        except Exception as ex:
            messagebox.showerror("Invalid Input", str(ex))
            return

        result = simulate_fusion(p1, p2)
        self._write_result(p1, p2, result)

        if result["fused"]:
            atom = Atom(result["product_z"], result["product_z"], result["product_name"],
                        result["product_symbol"], {"shells": estimate_shells(result["product_z"])})
            try:
                visualize_atom(atom)
            except Exception:
                pass  # visualization is a bonus - never block the result on it

    def _write_result(self, p1: Projectile, p2: Projectile, result: Dict[str, Any]):
        lines = []
        lines.append(f"PROJECTILE 1: {p1.label}  (Z={p1.z_total}, A≈{p1.a_total}, ×{p1.amount:,})")
        lines.append(f"PROJECTILE 2: {p2.label}  (Z={p2.z_total}, A≈{p2.a_total}, ×{p2.amount:,})")
        lines.append("")
        if result["matched_known_reaction"]:
            lines.append("✅ Matches a real, historical synthesis reaction:")
            lines.append(f"   {result['reaction_label']}")
            lines.append("")
        if not result["fused"]:
            lines.append("❌ NO FUSION — Coulomb repulsion wins.")
            lines.append(
                f"   At this speed ({result['combined_speed_c']:.4f}c) the nuclei don't have enough "
                f"energy to get close enough to fuse (estimated threshold ≈ {result['threshold_speed_c']:.4f}c)."
            )
            lines.append("   They scatter off each other instead. Try increasing the speed on either side.")
        else:
            lines.append(
                f"✅ FUSION SUCCESSFUL at {result['combined_speed_c']:.4f}c "
                f"(threshold ≈ {result['threshold_speed_c']:.4f}c)"
            )
            lines.append("")
            lines.append(f"   Product: {result['product_name']} ({result['product_symbol']})")
            lines.append(f"   Atomic number (Z): {result['product_z']}")
            lines.append(f"   Approx. mass number (A): {result['product_a']}")
            lines.append(f"   Neutrons evaporated: {result['neutrons_emitted']}")
            if result["product_z"] > HIGHEST_NAMED_ELEMENT:
                lines.append("   (No permanent IUPAC name yet — systematic name shown above.)")
            lines.append("")
            lines.append(f"   Estimated yield from this shot: {result['yield_estimate']} atom(s)")
            lines.append(
                "   Reality check: real superheavy-element synthesis has cross-sections so small that "
                "experiments run for weeks to produce a handful of atoms — this simulator scales things "
                "up for fun, it isn't a literal yield prediction."
            )
        lines.append("")
        lines.append(
            "Note: real nuclear fusion happens between bare nuclei, not whole molecules. When a compound "
            "is used as a projectile, its total proton count (summed across every atom in the molecule) "
            "is used as a simplified stand-in for game purposes."
        )
        self._set_result_text("\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = AtomApp(root)
    root.mainloop()
