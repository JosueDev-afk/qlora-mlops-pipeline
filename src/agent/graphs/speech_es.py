"""What the validate_contact agent says, in es-MX, and how it reads values back.

Kept apart from the graph so code that only needs the words (the telemetry
simulator, which also runs inside the Airflow image) does not pull LangGraph.
Agent speech is Spanish by convention; identifiers stay English.
"""

from __future__ import annotations

FIELDS = ("name", "phone", "email")

FIELD_NAMES = {"name": "su nombre", "phone": "su teléfono", "email": "su correo"}
ASK = {
    "name": "¿Me podría decir su nombre completo, por favor?",
    "phone": "¿Me podría dictar su número de celular a diez dígitos?",
    "email": "¿Me podría decir su correo electrónico?",
}
# Guided: the second try asks for the form the normalizers read best.
REPROMPT = {
    "name": "Disculpe, no le entendí bien. ¿Me lo podría deletrear?",
    "phone": "Disculpe, ¿me lo podría repetir de dos en dos dígitos?",
    "email": "Disculpe, ¿me lo podría deletrear despacio? Por ejemplo: jota, u, a, ene, "
    "arroba, gmail, punto com.",
}
GREETING = "Buen día, le llamo{company} para confirmar sus datos de contacto."
PURPOSE = "Le llamo para confirmar sus datos de contacto."
CONFIRM = "Tengo {value}. ¿Es correcto?"
SAVED = "Perfecto, quedó registrado."
DEFERRED = "No se preocupe, lo revisaremos después."
FAREWELL = "Gracias por su tiempo, que tenga buen día."
REJECTED = "Entiendo, disculpe la molestia. Que tenga buen día."


def readback(field: str, value: str) -> str:
    """How the agent reads a value back so the caller can confirm it by ear."""
    if field == "phone":
        national = value.removeprefix("+52")
        return " ".join(national[i : i + 2] for i in range(0, len(national), 2))
    if field == "email":
        for symbol, word in (("@", " arroba "), (".", " punto "), ("_", " guion bajo "),
                             ("-", " guion ")):  # fmt: skip
            value = value.replace(symbol, word)
        return " ".join(value.split())
    return value


def keyterms(field: str, contact: dict[str, str]) -> list[str]:
    """Stored values that help the STT hear this field (keyterm prompting)."""
    if field == "name":
        return contact.get("name", "").split()
    if field == "email" and "@" in contact.get("email", ""):
        return [contact["email"].split("@")[1].split(".")[0]]
    return []


def summary(results: dict[str, str]) -> str:
    validated = [FIELD_NAMES[f] for f, r in results.items() if r == "validated"]
    if not validated:
        return f"Revisaremos sus datos por otro medio. {FAREWELL}"
    listed = " y ".join([", ".join(validated[:-1]), validated[-1]] if len(validated) > 1
                        else validated)  # fmt: skip
    plural = "quedaron confirmados" if len(validated) > 1 else "quedó confirmado"
    deferred = " Lo demás lo revisaremos por otro medio." if len(validated) < len(results) else ""
    return f"Listo, {plural} {listed}.{deferred} {FAREWELL}"
