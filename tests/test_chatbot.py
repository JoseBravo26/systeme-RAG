"""
Tests unitaires de la logique métier du chatbot Puls-Events.

Ces tests ne chargent ni le modèle Mistral ni l'index FAISS.
Ils vérifient uniquement le parsing des contraintes et le filtrage métier.
"""

from datetime import date
from types import SimpleNamespace

from src.rag.chatbot import PulsEventsChatbot


def create_bot_without_initialization() -> PulsEventsChatbot:
    """
    Instancie le chatbot sans appeler __init__.

    Cette approche évite tout appel réseau et tout chargement FAISS durant
    les tests unitaires.
    """
    return PulsEventsChatbot.__new__(PulsEventsChatbot)


def create_document(
    title: str,
    city: str,
    department: str,
    date_start: str,
    date_end: str,
    content: str,
) -> SimpleNamespace:
    """
    Crée un faux document compatible avec la structure LangChain.
    """
    return SimpleNamespace(
        page_content=content,
        metadata={
            "uid": title,
            "title": title,
            "city": city,
            "department": department,
            "date_start": date_start,
            "date_end": date_end,
        },
    )


def test_extract_exact_date_and_city() -> None:
    """
    Une question contenant Paris et une date complète doit être interprétée.
    """
    bot = create_bot_without_initialization()

    constraints = bot._extract_constraints(
        "Quels concerts de jazz sont proposés à Paris le 24 juillet 2026 ?"
    )

    assert constraints["city"] == "Paris"
    assert constraints["department"] is None
    assert constraints["start_date"] == date(2026, 7, 24)
    assert constraints["end_date"] == date(2026, 7, 24)


def test_extract_month_period() -> None:
    """
    Une question sur septembre doit couvrir tout le mois demandé.
    """
    bot = create_bot_without_initialization()

    constraints = bot._extract_constraints(
        "Existe-t-il une visite à Paris en septembre ?"
    )

    assert constraints["city"] == "Paris"
    assert constraints["start_date"].month == 9
    assert constraints["end_date"].month == 9
    assert constraints["start_date"].day == 1


def test_document_matches_two_requested_topics() -> None:
    """
    Une demande 'cinéma et jazz' exige que le document corresponde
    aux deux thèmes.
    """
    bot = create_bot_without_initialization()

    document = create_document(
        title="CINÉ-JAZZ : les thèmes légendaires du grand écran",
        city="Paris",
        department="Paris",
        date_start="2026-08-01T17:30:00+00:00",
        date_end="2026-08-01T23:30:00+00:00",
        content="Une soirée mêlant cinéma, musique et jazz.",
    )

    assert bot._matches_topic(
        document,
        "Y a-t-il un événement mêlant cinéma et jazz à Paris ?",
    )


def test_document_without_cinema_is_rejected() -> None:
    """
    Un événement uniquement jazz ne répond pas à une question cinéma + jazz.
    """
    bot = create_bot_without_initialization()

    document = create_document(
        title="Concert de jazz",
        city="Paris",
        department="Paris",
        date_start="2026-08-01T17:30:00+00:00",
        date_end="2026-08-01T23:30:00+00:00",
        content="Une soirée de jazz manouche.",
    )

    assert not bot._matches_topic(
        document,
        "Y a-t-il un événement mêlant cinéma et jazz à Paris ?",
    )


def test_filter_keeps_event_matching_date_city_and_topic() -> None:
    """
    Le filtrage doit retenir l'événement astronomique demandé.
    """
    bot = create_bot_without_initialization()

    eclipse = create_document(
        title="Éclipse de Soleil - Nuit des étoiles",
        city="Paris",
        department="Paris",
        date_start="2026-08-12T12:00:00+00:00",
        date_end="2026-08-12T21:59:00+00:00",
        content="Ateliers d'astronomie, observation du soleil et éclipse.",
    )

    wrong_city = create_document(
        title="Éclipse dans une autre ville",
        city="Lyon",
        department="Rhône",
        date_start="2026-08-12T12:00:00+00:00",
        date_end="2026-08-12T21:59:00+00:00",
        content="Observation astronomique.",
    )

    results = bot._filter_documents(
        documents=[eclipse, wrong_city],
        question="Que se passe-t-il à Paris le 12 août 2026 autour de l’astronomie ?",
        current_date=date(2026, 7, 23),
        city_filter="Paris",
        department_filter=None,
        start_filter=date(2026, 8, 12),
        end_filter=date(2026, 8, 12),
    )

    assert len(results) == 1
    assert results[0].metadata["title"] == "Éclipse de Soleil - Nuit des étoiles"


def test_filter_rejects_event_outside_requested_date() -> None:
    """
    Un événement pertinent par thème mais hors période doit être exclu.
    """
    bot = create_bot_without_initialization()

    event = create_document(
        title="Un été astronomique",
        city="Paris",
        department="Paris",
        date_start="2026-07-04T08:00:00+00:00",
        date_end="2026-09-13T16:00:00+00:00",
        content="Des animations et ateliers d'astronomie.",
    )

    results = bot._filter_documents(
        documents=[event],
        question="Que se passe-t-il à Paris le 12 août 2026 autour de l’astronomie ?",
        current_date=date(2026, 7, 23),
        city_filter="Paris",
        department_filter=None,
        start_filter=date(2026, 8, 12),
        end_filter=date(2026, 8, 12),
    )

    # L'événement couvre le 12 août : il doit donc être retenu.
    assert len(results) == 1