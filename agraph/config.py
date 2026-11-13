"""Shared paths and settings for the researcher knowledge-graph pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
CACHE = DATA / "cache"          # raw OpenAlex responses (re-runs are free)
PROC = DATA / "processed"       # tables, embeddings, graph exports
WEB = ROOT / "web"
for _d in (CACHE, PROC, WEB):
    _d.mkdir(parents=True, exist_ok=True)

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY") or None
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---- Selection ---------------------------------------------------------------
N_RESEARCHERS = int(os.getenv("AGRAPH_N", 1000))
# A researcher qualifies when at least this share of their OpenAlex topic
# counts is Computer Science or robotics.
MIN_FIELD_SHARE = 0.5
CS_FIELD_ID = "17"  # OpenAlex field "Computer Science"

# OpenAlex topics treated as robotics (several live under Engineering, not CS).
ROBOTICS_TOPICS = {
    "T10462": "Reinforcement Learning in Robotics",
    "T10571": "Robotic Mechanisms and Dynamics",
    "T10191": "Robotics and Sensor-Based Localization",
    "T10653": "Robot Manipulation and Learning",
    "T10709": "Social Robot Interaction and HRI",
    "T10868": "Soft Robotics and Applications",
    "T10586": "Robotic Path Planning Algorithms",
    "T11023": "Prosthetics and Rehabilitation Robotics",
    "T10879": "Robotic Locomotion and Control",
    "T11615": "Control and Dynamics of Mobile Robots",
    "T13382": "Robotics and Automated Systems",
    "T12784": "Modular Robots and Swarm Intelligence",
    "T11687": "Teleoperation and Haptic Systems",
    "T14335": "Educational Robotics and Engineering",
    "T13715": "Power Line Inspection Robots",
    "T11486": "Micro and Nano Robotics",
}
# OpenAlex files all deep RL (Atari, Go, ...) under this topic. It counts as robotics only
# for researchers who also have at least one other robotics topic.
AMBIGUOUS_ROBOTICS_TOPICS = {"T10462"}
MIN_WORKS = 5  # profiles with fewer works are fragments or ghosts (e.g. one mega-paper)
ROBOTICS_LABEL_SHARE = 0.25  # robotics share above which a researcher is tagged "robotics"

# ---- Works / embeddings ------------------------------------------------------
WORKS_PER_RESEARCHER = 100   # most-cited works used to describe each researcher
SPECTER_BASE = "allenai/specter2_base"
SPECTER_DOC_ADAPTER = "allenai/specter2"              # proximity adapter (papers)
SPECTER_QUERY_ADAPTER = "allenai/specter2_adhoc_query"  # short text queries

# ---- Graph -------------------------------------------------------------------
KNN_K = 10              # SIMILAR_TO edges per researcher
COMMUNITY_K_RANGE = (14, 24)  # k-means cluster counts tried; the best silhouette wins
SEED = 42


def short_id(openalex_url: str | None) -> str | None:
    """'https://openalex.org/A123' -> 'A123'."""
    if not openalex_url:
        return None
    return openalex_url.rsplit("/", 1)[-1]
