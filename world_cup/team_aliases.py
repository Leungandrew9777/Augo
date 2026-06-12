"""
National-team name normalization for the World Cup pipeline.
"""

from __future__ import annotations

# Display / fixture name -> canonical training-data name
TEAM_ALIASES: dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "IR Iran ": "Iran",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Republic of Ireland": "Ireland",
    "Northern Ireland": "Northern Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Chinese Taipei": "Taiwan",
    "Korea DPR": "North Korea",
    "Democratic People's Republic of Korea": "North Korea",
}

# football-data.org -> canonical
FOOTBALL_DATA_TO_CANONICAL: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
}


def canonical_name(name: str) -> str:
    if not name:
        return name  # type: ignore[return-value]
    s = str(name).strip()
    return TEAM_ALIASES.get(s, FOOTBALL_DATA_TO_CANONICAL.get(s, s))


def fixture_lookup_key(name: str) -> str:
    return canonical_name(name)


# Teams commonly appearing in recent international tournaments (for UI custom predictor).
WC_TEAMS: list[str] = sorted(
    {
        "Argentina",
        "Australia",
        "Austria",
        "Belgium",
        "Brazil",
        "Cameroon",
        "Canada",
        "Chile",
        "Colombia",
        "Costa Rica",
        "Croatia",
        "Czech Republic",
        "Denmark",
        "Ecuador",
        "Egypt",
        "England",
        "France",
        "Germany",
        "Ghana",
        "Greece",
        "Iran",
        "Italy",
        "Ivory Coast",
        "Japan",
        "Mexico",
        "Morocco",
        "Netherlands",
        "Nigeria",
        "Norway",
        "Poland",
        "Portugal",
        "Qatar",
        "Saudi Arabia",
        "Scotland",
        "Senegal",
        "Serbia",
        "South Africa",
        "South Korea",
        "Spain",
        "Switzerland",
        "Tunisia",
        "Turkey",
        "Ukraine",
        "United States",
        "Uruguay",
        "Wales",
    }
)
