"""Damage levels and management guidance shown by the web app.

Damage levels are DESCRIPTIVE bands defined for this tool from the image-level estimate of
damaged visible leaf area. They are NOT validated economic/action thresholds. Field decisions
follow scouting-based thresholds from:
  Prasanna B.M., Huesing J.E., Eddy R., Peschke V.M. (eds) 2018. Fall Armyworm in Africa:
  A Guide for Integrated Pest Management, 1st ed. CIMMYT/USAID, Mexico CDMX.
"""

SOURCE = ("Prasanna, B.M., Huesing, J.E., Eddy, R., Peschke, V.M. (eds), 2018. Fall Armyworm in Africa: "
          "A Guide for Integrated Pest Management. CIMMYT/USAID.")

LEVELS = [  # (upper bound of % visible leaf damaged, key, label)
    (0.05, "none", "No visible feeding damage detected"),
    (2.0, "low", "Low visible damage"),
    (5.0, "moderate", "Moderate visible damage"),
    (float("inf"), "high", "High visible damage"),
]

ALWAYS = [
    "This photo-based estimate covers only the leaves visible in this image. Base any control decision on field "
    "scouting, not on a single photo.",
    "The tool detects visible feeding damage; it does not confirm fall armyworm. Check the whorl for larvae and "
    "fresh frass (sawdust-like droppings) to confirm the pest.",
    "Scouting method (CIMMYT guide): enter about 5 m into the field, avoid edge rows, stop at 5 locations in a 'W' "
    "pattern and examine 10–20 plants at each stop; repeat weekly during seedling and early-whorl stages.",
    "Action thresholds from the same guide are based on the % of plants infested: early whorl (VE–V6) about 20% "
    "(range 10–30%); late whorl (V7–VT) about 40% (range 30–50%).",
]

BY_LEVEL = {
    "none": [
        "Keep scouting weekly; pheromone traps can show when moths arrive, but never decide on spraying from moth "
        "counts alone.",
        "Keep the crop vigorous (timely weeding, fertiliser and water): healthy plants tolerate pest attack better.",
    ],
    "low": [
        "Crush egg masses and hand-pick larvae from the whorls where practical.",
        "Protect natural enemies (ants, earwigs, spiders, predatory beetles, parasitoid wasps): avoid broad-spectrum "
        "sprays at this level.",
        "Re-scout the field within a week to see whether damage is spreading.",
    ],
    "moderate": [
        "Scout now and compare the % of infested plants with the threshold for the crop's growth stage.",
        "If the threshold is exceeded, prefer low-toxicity options that spare natural enemies, e.g. registered "
        "biopesticides based on FAW viruses (SfMNPV), entomopathogenic fungi or nematodes, or releases of egg "
        "parasitoids (Trichogramma, Telenomus remus) where available. Ask your local extension office which "
        "products are registered.",
        "Continue hand-picking larvae and crushing egg masses.",
    ],
    "high": [
        "Scout urgently: high visible damage may mean the infestation has passed the action threshold.",
        "If the threshold is exceeded and a chemical insecticide is needed, use only products registered in your "
        "country, follow the label rate and intervals, wear protective equipment, rotate modes of action (do not "
        "use the same active ingredient on successive pest generations), and observe the pre-harvest interval.",
        "The CIMMYT guide advises smallholders not to spray insecticides at or after tasselling (VT) because of the "
        "risk to the applicator and family.",
        "Contact your local agricultural extension officer for advice specific to your area.",
    ],
}


def level_for(pct_leaf):
    for ub, key, label in LEVELS:
        if pct_leaf < ub:
            return key, label
    return LEVELS[-1][1], LEVELS[-1][2]


def recommendations(pct_leaf):
    key, label = level_for(pct_leaf)
    return {"level": key, "label": label, "actions": BY_LEVEL[key], "general": ALWAYS, "source": SOURCE,
            "note": "Damage levels are descriptive bands used by this tool (none < 0.05%, low < 2%, moderate 2–5%, "
                    "high ≥ 5% of visible leaf area); they are not validated economic thresholds."}
