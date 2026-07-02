#!/usr/bin/env python3
"""
Redrob Hackathon – Final Ranking Engine (v10.0)
Team: The defenders | Solo: Tanish M

Pipeline:
  Phase 1   – Hard drops (country, services-only, product-months, non-coding, YOE)
  Phase 1B  – Honeypot detection (score * 0.01 multiplier)
  Phase 2A  – Multiplicative penalties (incl. trust-clipping of inflated durations)
  Phase 2B  – Additive boosts (education tier, certifications, TF-IDF percentile vs JD)
  Phase 2C  – Top-200 re-rank on concrete production-retrieval evidence
  Phase 3   – Dynamic, feature-driven reasoning for top-100

Usage:
  python rank.py --candidates candidates.jsonl --out team_the_defenders.csv
"""

import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# JOB DESCRIPTION TEXT (for semantic-similarity scoring, Phase 2B)
# Condensed from job_description.docx — core requirements only, hackathon
# meta-commentary excluded so the vector represents the actual role.
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer, Founding Team, Redrob AI. Own the intelligence layer:
ranking, retrieval, and matching systems. Deep technical depth in modern ML
systems: embeddings, retrieval, ranking, LLMs, fine-tuning. Scrappy
product-engineering attitude, ships working systems fast.
Production experience with embeddings-based retrieval systems: sentence
transformers, OpenAI embeddings, BGE, E5. Handled embedding drift, index
refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search: Pinecone,
Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python, code quality.
Evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online
correlation, A/B testing.
LLM fine-tuning: LoRA, QLoRA, PEFT. Learning-to-rank models: XGBoost, neural.
HR-tech, recruiting tech, marketplace products. Distributed systems,
large-scale inference optimization. Open-source contributions.
Shipped an end-to-end ranking, search, or recommendation system to real
users at meaningful scale. Hybrid vs dense retrieval, offline vs online
evaluation, when to fine-tune vs prompt.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SERVICES_INDUSTRIES = {
    "IT Services", "Consulting", "Management Consulting",
    "Outsourcing", "BPO", "IT Consulting"
}

KNOWN_SERVICE_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "l&t infotech",
    "mindtree", "hexaware", "persistent"
}

NON_TECH_TITLES = {
    "marketing", "sales", "hr", "human resources", "recruiter",
    "finance", "accountant", "designer", "graphic designer",
    "operations", "product manager", "business development",
    "business analyst", "content writer", "customer support",
    "civil engineer", "mechanical engineer", "project manager",
    "sales executive"
}

ML_KEYWORDS = {
    "embedding", "embeddings", "retrieval", "ranking", "llm",
    "fine-tuning", "fine tuning", "sentence-transformers",
    "sentence transformers", "pinecone", "milvus", "qdrant",
    "weaviate", "faiss", "opensearch", "elasticsearch",
    "rag", "re-ranking", "reranking", "ndcg", "mrr",
    "mean average precision", "a/b test", "bm25", "hybrid search",
    "dense retrieval", "learning-to-rank", "learning to rank",
    "xgboost", "lightgbm", "recommendation",
    "collaborative filtering", "feature engineering"
}

NLP_IR_SKILLS = {
    "nlp", "information retrieval", "embeddings", "semantic search",
    "ranking", "retrieval", "sentence transformers", "rag",
    "vector search", "bm25", "faiss", "pinecone", "qdrant",
    "milvus", "weaviate", "opensearch", "elasticsearch",
    "hugging face transformers", "fine-tuning llms",
    "recommendation systems", "feature engineering"
}

CV_SKILLS = {
    "computer vision", "opencv", "yolo", "image classification",
    "object detection", "speech recognition", "tts", "asr",
    "robotics", "gans", "cnn"
}

PRODUCTION_TERMS = {
    "shipped", "deployed", "production", "a/b test", "launched",
    "users", "our product", "scale", "serving", "inference",
    "latency", "throughput", "pipeline"
}

CODING_TERMS = {
    "built", "shipped", "deployed", "implemented", "python",
    "model", "architecture", "code", "developed", "engineering",
    "system", "designed", "pipeline"
}

# JD preferred locations: Noida/Pune (ideal) vs other metros (welcome)
IDEAL_LOCATIONS   = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram"}

NON_CODING_TITLES = {
    "architect", "tech lead", "engineering manager", "head of",
    "director", "vp", "principal"
}

TIER_1_COMPANIES = {
    "google", "meta", "facebook", "amazon", "apple", "netflix", "microsoft",
    "nvidia", "openai", "deepmind"
}

TIER_2_COMPANIES = {
    "uber", "salesforce", "adobe", "linkedin", "twitter", "stripe", "atlassian",
    "airbnb", "dropbox", "spotify", "oracle", "ibm", "sap", "vmware",
    "databricks", "snowflake"
}

TIER_3_COMPANIES = {
    "swiggy", "zomato", "ola", "flipkart", "cred", "razorpay",
    "sarvam ai", "yellow.ai", "haptik", "observe.ai", "paytm", "zoho"
}

# Concrete production-retrieval evidence terms for the top-200 second pass
RETRIEVAL_EVIDENCE_TERMS = (
    "embedding", "vector search", "vector database", "faiss", "pinecone",
    "qdrant", "milvus", "weaviate", "elasticsearch", "opensearch", "vespa",
    "ndcg", "mrr", "recall@", "a/b test", "bm25", "reranking", "re-ranking",
    "learning to rank", "retrieval",
)

RELEVANT_CERT_TERMS = (
    "machine learning", "deep learning", "ai", "tensorflow", "aws certified",
    "gcp", "google cloud", "azure", "data engineer", "nlp",
)

RELEVANT_FIELDS = ("computer", "artificial intelligence", "machine learning",
                   "data science", "information", "statistics", "mathematics")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def days_since(date_str: str, now: datetime) -> Optional[int]:
    d = parse_date(date_str)
    if d is None:
        return None
    return (now - d.replace(tzinfo=timezone.utc)).days

def parse_date(date_str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

def start_year(job: Dict) -> int:
    s = (job.get("start_date") or "")[:4]
    return int(s) if s.isdigit() else 9999

def clipped_skill_months(skill: Dict, yoe: float) -> int:
    """Trust-clipping: a skill cannot have been practiced longer than the
    candidate's entire career. Caps inflated duration_months at yoe*12 so
    keyword-stuffed / honeypot-style profiles stop out-scoring honest ones."""
    return min(skill.get("duration_months", 0), max(int(yoe * 12), 0))

def is_non_tech_title(title: str) -> bool:
    t = title.lower()
    # Allow data/ml/bi analysts through
    if "analyst" in t and any(x in t for x in ["data", "ml", "bi", "business intelligence"]):
        return False
    return any(nt in t for nt in NON_TECH_TITLES)

def is_service_company(company: str) -> bool:
    comp = company.strip().lower()
    if any(firm in comp for firm in KNOWN_SERVICE_FIRMS):
        return True
    if any(kw in comp for kw in ("consulting", "outsourcing", "bpo", "tech services")):
        return True
    return False

def is_tier_company(company: str, tier_set: set) -> bool:
    comp = company.strip().lower()
    return any(tier_firm in comp for tier_firm in tier_set)

# ---------------------------------------------------------------------------
# COUNTING
# ---------------------------------------------------------------------------
def compute_qualifying_months(c: Dict) -> int:
    """Sum all product-company months in technical (non-non-tech) roles."""
    total = 0
    for job in c.get("career_history", []):
        if job.get("industry", "") in SERVICES_INDUSTRIES:
            continue
        if is_service_company(job.get("company", "")):
            continue
        if is_non_tech_title(job.get("title", "")):
            continue
        total += job.get("duration_months", 0)
    return total

def streak_months_to_reward(months: int, tier: int) -> float:
    if months >= 72:
        base = 2.0
    elif months >= 48:
        base = 1.5
    elif months >= 36:
        base = 1.0
    else:
        return 0.0
    if tier == 3:
        return base * 1.5
    elif tier == 2:
        return base * 1.2
    else:
        return base

def tiered_streak_reward(career: List[Dict]) -> float:
    if not career:
        return 0.0
    sorted_jobs = sorted(career, key=lambda j: j.get("start_date", "9999-99-99"))
    best_reward = 0.0
    current_streak_months = 0
    current_streak_tier = 0
    prev_end = None

    for job in sorted_jobs:
        industry = job.get("industry", "")
        company = job.get("company", "").strip().lower()
        is_service = industry in SERVICES_INDUSTRIES or is_service_company(company)
        start = parse_date(job.get("start_date"))
        if start is None:
            continue

        if prev_end is not None:
            gap_days = (start - prev_end).days
            if gap_days > 180:
                if current_streak_months >= 36:
                    best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
                current_streak_months = 0
                current_streak_tier = 0

        if is_service:
            if current_streak_months >= 36:
                best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
            current_streak_months = 0
            current_streak_tier = 0
        else:
            current_streak_months += job.get("duration_months", 0)
            if is_tier_company(company, TIER_1_COMPANIES):
                current_streak_tier = max(current_streak_tier, 3)
            elif is_tier_company(company, TIER_2_COMPANIES):
                current_streak_tier = max(current_streak_tier, 2)
            elif is_tier_company(company, TIER_3_COMPANIES):
                current_streak_tier = max(current_streak_tier, 1)

        if not job.get("is_current") and job.get("end_date"):
            prev_end = parse_date(job["end_date"])
        else:
            prev_end = None

    if current_streak_months >= 36:
        best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
    return best_reward

# ---------------------------------------------------------------------------
# PHASE 1: HARD DROPS
# ---------------------------------------------------------------------------
def hard_drop(c: Dict, now: datetime) -> bool:
    profile = c.get("profile", {}) or {}
    signals = c.get("redrob_signals", {}) or {}
    career = c.get("career_history", [])

    # 1. Not in India
    if profile.get("country") != "India":
        return True

    # availability/activity: soft penalty in Phase 2A, not a hard drop

    # 4. Pure IT services career (zero product company jobs)
    if career:
        product_jobs = sum(
            1 for job in career
            if job.get("industry") not in SERVICES_INDUSTRIES
            and not is_service_company(job.get("company", ""))
        )
        if product_jobs == 0:
            return True

    # 5. Insufficient product-company tech experience (< 48 months)
    if compute_qualifying_months(c) < 48:
        return True

    # 6. Non-coding current role >= 18 months (hard drop)
    for job in career:
        if job.get("is_current"):
            title = job.get("title", "").lower()
            desc = job.get("description", "").lower()
            duration = job.get("duration_months", 0)
            if any(t in title for t in NON_CODING_TITLES) and not any(term in desc for term in CODING_TERMS):
                if duration >= 18:
                    return True
            break

    # 7. Total experience < 4 years
    if profile.get("years_of_experience", 0) < 4:
        return True

    return False

# ---------------------------------------------------------------------------
# PHASE 1B: HONEYPOT DETECTION
# ---------------------------------------------------------------------------
def is_honeypot(c: Dict) -> bool:
    # flag profiles with 2+ impossible skill signals
    profile = c.get("profile", {}) or {}
    skills = c.get("skills", [])
    yoe = profile.get("years_of_experience", 0)
    red_flags = 0

    # Expert + 0 months
    if any(s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0 for s in skills):
        red_flags += 1

    # Too many experts for YOE
    high_prof = sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert"))
    if high_prof >= 10 and yoe < 5:
        red_flags += 1

    # Impossible skill duration
    if any(s.get("duration_months", 0) > (yoe * 12) + 24 for s in skills):
        red_flags += 1

    # Un-endorsed experts (zero endorsements on an "expert" claim)
    zero_endorsed_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0
    ]
    if len(zero_endorsed_experts) >= 6:
        red_flags += 1

    return red_flags >= 2

# ---------------------------------------------------------------------------
# PHASE 2A: MULTIPLICATIVE PENALTIES
# ---------------------------------------------------------------------------
def penalty_multiplier(c: Dict, now: datetime) -> float:
    mul = 1.0
    profile = c.get("profile", {}) or {}
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {}) or {}
    yoe = profile.get("years_of_experience", 0)
    all_descs = " ".join(job.get("description", "").lower() for job in career)

    nlp_ir_months = sum(
        clipped_skill_months(s, yoe) for s in skills if s.get("name", "").lower() in NLP_IR_SKILLS
    )

    # 0. Duration inflation: skills claiming more months than the whole
    # career + 2yr slack — the JD's "keyword embedding" anti-pattern
    inflated = sum(1 for s in skills if s.get("duration_months", 0) > (yoe * 12) + 24)
    if inflated >= 2:
        mul *= 0.6
    elif inflated == 1:
        mul *= 0.85

    # 1. Shallow LLM-only (×0.1)
    has_hype = any(
        s.get("name", "").lower() in ("langchain", "openai", "llamaindex", "prompt engineering")
        and s.get("duration_months", 0) <= 12
        for s in skills
    )
    has_deep = any(
        s.get("name", "").lower() in NLP_IR_SKILLS and clipped_skill_months(s, yoe) > 24
        for s in skills
    )
    has_pre_llm_work = any(
        start_year(job) < 2023
        and any(kw in job.get("description", "").lower() for kw in ML_KEYWORDS)
        for job in career
    )
    if has_hype and not has_deep and not has_pre_llm_work:
        mul *= 0.1

    # 1b. Framework dominance (×0.3)
    langchain_months = sum(
        clipped_skill_months(s, yoe) for s in skills
        if s.get("name", "").lower() in ("langchain", "llamaindex", "openai")
    )
    if langchain_months > 24 and langchain_months > (nlp_ir_months * 0.5) and not has_deep:
        mul *= 0.3

    # 2. Pure research (×0.1)
    if not any(term in all_descs for term in PRODUCTION_TERMS):
        mul *= 0.1

    # 3. Non-coding current role < 18 months (soft penalty ×0.5)
    for job in career:
        if job.get("is_current"):
            title = job.get("title", "").lower()
            desc = job.get("description", "").lower()
            dur = job.get("duration_months", 0)
            if any(t in title for t in NON_CODING_TITLES) and not any(term in desc for term in CODING_TERMS):
                if dur < 18:
                    mul *= 0.5
            break

    # 4. CV/Speech primary without meaningful NLP/IR depth (×0.3)
    cv_months = sum(
        clipped_skill_months(s, yoe) for s in skills if s.get("name", "").lower() in CV_SKILLS
    )
    if cv_months > 0 and cv_months > nlp_ir_months and nlp_ir_months < 24:
        mul *= 0.3

    # 5. Prompt Engineering dominance (×0.2)
    top_3 = [
        s.get("name", "").lower()
        for s in sorted(skills, key=lambda x: clipped_skill_months(x, yoe), reverse=True)[:3]
    ]
    if "prompt engineering" in top_3 and nlp_ir_months < 48:
        mul *= 0.2

    # 6. Title-chaser / job-hopper (×0.1)
    if len(career) >= 3:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        titles = [j.get("title", "").lower() for j in sorted(career, key=lambda x: x.get("start_date", ""))]
        bumps = 0
        for i in range(1, len(titles)):
            if any(t in titles[i] for t in ("senior", "staff", "principal", "lead")) and \
               not any(t in titles[i - 1] for t in ("senior", "staff", "principal", "lead")):
                bumps += 1
        if avg_tenure < 18 and bumps >= 2:
            mul *= 0.1

    # 7. Junior title (×0.5)
    if "junior" in profile.get("current_title", "").lower():
        mul *= 0.5

    # 8. Notice period pressure
    notice = signals.get("notice_period_days", 60)
    if notice > 120:
        mul *= 0.9
    elif notice > 90:
        mul *= 0.95

    # 9. Service-firm + Junior (double red flag ×0.3)
    curr_comp = profile.get("current_company", "").strip().lower()
    if is_service_company(curr_comp) and "junior" in profile.get("current_title", "").lower():
        mul *= 0.3

    # 10. Zero-retrieval-signal penalty (×0.1)
    retrieval_desc_terms = [
        "embedding", "embeddings", "retrieval", "ranking", "vector search",
        "faiss", "pinecone", "qdrant", "milvus", "weaviate", "elasticsearch",
        "opensearch", "bm25", "learning to rank", "recommendation system"
    ]
    has_retrieval_desc = any(kw in all_descs for kw in retrieval_desc_terms)
    if nlp_ir_months == 0 and not has_retrieval_desc:
        mul *= 0.1

    # 11. Not open to work and no recent applications (×0.2)
    if not signals.get("open_to_work_flag") and signals.get("applications_submitted_30d", 0) == 0:
        mul *= 0.2

    # 12. Inactive > 6 months (×0.15)
    inactive_days = days_since(signals.get("last_active_date"), now)
    if inactive_days is not None and inactive_days > 180:
        mul *= 0.15

    # 13. Declines most offers (×0.85) — "can actually be hired" risk
    oar = signals.get("offer_acceptance_rate", -1)
    if 0 <= oar < 0.3:
        mul *= 0.85

    # 14. Remote-only + won't relocate + not near Pune/Noida offices (×0.85)
    location = profile.get("location", "").lower()
    near_office = any(city in location for city in IDEAL_LOCATIONS | WELCOME_LOCATIONS)
    if (signals.get("preferred_work_mode") == "remote"
            and not signals.get("willing_to_relocate")
            and not near_office):
        mul *= 0.85

    # 15. Senior with zero external validation (×0.85): 5+ yrs, no GitHub —
    # the JD's "entirely closed-source" anti-pattern
    if yoe >= 5 and signals.get("github_activity_score", -1) == -1:
        mul *= 0.85

    return mul

# ---------------------------------------------------------------------------
# PHASE 2B: ADDITIVE BOOSTS
# ---------------------------------------------------------------------------
def build_candidate_text(c: Dict) -> str:
    """Free-text representation of a candidate for TF-IDF similarity against the JD."""
    profile = c.get("profile", {}) or {}
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    # deliberately excludes the skills list: descriptions generalize better,
    # and honeypot-style profiles stuff skill names to game keyword matching
    return " ".join(p for p in parts if p)


def compute_boost_score(c: Dict, qualifying_months: int, jd_sim_percentile: float = 0.0) -> float:
    score = 0.0
    # TF-IDF similarity as a percentile rank across survivors (0..1) so raw
    # cosine magnitudes (typically 0.05-0.3) don't get drowned out
    score += jd_sim_percentile * 4.0
    profile = c.get("profile", {}) or {}
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {}) or {}
    yoe = profile.get("years_of_experience", 0)
    all_descs = " ".join(job.get("description", "").lower() for job in career)

    current_title = profile.get("current_title", "").lower()
    current_desc = next(
        (j.get("description", "").lower() for j in career if j.get("is_current")), ""
    )

    # Senior title + long avg tenure + hands-on coder (+3.0)
    if any(t in current_title for t in ("senior", "lead", "principal", "staff")):
        if len(career) >= 2:
            avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
            if avg_tenure >= 24 and any(term in current_desc for term in CODING_TERMS):
                score += 3.0

    # Currently hands-on coder (+2.0)
    if any(term in current_desc for term in CODING_TERMS):
        score += 2.0

    # location boost: ideal > welcome > willing to relocate
    location = profile.get("location", "").lower()
    if any(city in location for city in IDEAL_LOCATIONS):
        score += 2.0
    elif any(city in location for city in WELCOME_LOCATIONS):
        score += 1.0
    elif signals.get("willing_to_relocate"):
        score += 0.5

    # Notice period <= 30 days (+1.0)
    if signals.get("notice_period_days", 999) <= 30:
        score += 1.0

    # Recruiter engagement (+1.0)
    if signals.get("recruiter_response_rate", 0) > 0.7 and signals.get("interview_completion_rate", 0) > 0.6:
        score += 1.0

    # Skill assessments (max +2.0)
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        ml_assess = sum(
            1 for k in assessments
            if k.lower() in NLP_IR_SKILLS or k.lower() in ML_KEYWORDS
        )
        score += min(ml_assess * 0.5, 2.0)

    # Verified profile (+0.5)
    if signals.get("verified_email") and signals.get("verified_phone"):
        score += 0.5

    # GitHub activity (max +2.0)
    github = signals.get("github_activity_score", -1)
    if github > 0:
        score += min(github * 0.02, 2.0)

    # ML keywords in career descriptions (max +5.0, capped so keyword
    # stuffing can't dominate description-based evidence)
    kw_count = sum(1 for kw in ML_KEYWORDS if kw in all_descs)
    score += min(kw_count, 5.0)

    # Niche valuable skills
    skill_names = {s.get("name", "").lower() for s in skills}
    if any(k in skill_names for k in ("lora", "qlora", "peft")):
        score += 2.0
    if any(k in skill_names for k in ("xgboost", "lightgbm", "learning to rank")):
        score += 1.5

    # Domain experience in HR-Tech / Recruiting (+1.0)
    if any(
        job.get("industry", "").lower() in ("hr-tech", "recruiting", "talent intelligence")
        for job in career
    ):
        score += 1.0

    # Python proficiency
    for s in skills:
        if s.get("name", "").lower() == "python":
            if s.get("proficiency") in ("advanced", "expert"):
                score += 2.0
            elif s.get("proficiency") == "intermediate":
                score += 1.0
            break

    # English proficiency (from root-level languages list)
    for lang in c.get("languages", []):
        if lang.get("language", "").lower() == "english":
            prof = lang.get("proficiency", "").lower()
            if prof == "native":
                score += 1.0
            elif prof == "professional":
                score += 0.5
            break

    # Behavioral signals
    if signals.get("profile_completeness_score", 0) >= 80:
        score += 0.5
    if signals.get("applications_submitted_30d", 0) >= 5:
        score += 0.5
    avg_resp = signals.get("avg_response_time_hours", 999)
    if 0 < avg_resp <= 24:
        score += 0.5
    if signals.get("search_appearance_30d", 0) >= 50:
        score += 0.5

    # Tiered company boost: best ever
    best_tier = 0
    for job in career:
        company = job.get("company", "").strip().lower()
        if is_tier_company(company, TIER_1_COMPANIES):
            best_tier = max(best_tier, 3)
        elif is_tier_company(company, TIER_2_COMPANIES):
            best_tier = max(best_tier, 2)
        elif is_tier_company(company, TIER_3_COMPANIES):
            best_tier = max(best_tier, 1)
    if best_tier == 3:
        score += 1.5
    elif best_tier == 2:
        score += 1.0
    elif best_tier == 1:
        score += 0.5

    # Current company bonus
    curr_company = profile.get("current_company", "").strip().lower()
    if is_tier_company(curr_company, TIER_1_COMPANIES):
        score += 1.0
    elif is_tier_company(curr_company, TIER_2_COMPANIES):
        score += 0.5
    elif is_tier_company(curr_company, TIER_3_COMPANIES):
        score += 0.25

    # Tiered unbroken product streak
    score += tiered_streak_reward(career)

    # NLP/IR depth boost (trust-clipped so inflated durations don't count)
    nlp_ir_total = sum(
        clipped_skill_months(s, yoe) for s in skills if s.get("name", "").lower() in NLP_IR_SKILLS
    )
    if nlp_ir_total >= 72:
        score += 3.0
    elif nlp_ir_total >= 48:
        score += 1.5

    # Saved by recruiters >= 10 (+1.0)
    if signals.get("saved_by_recruiters_30d", 0) >= 10:
        score += 1.0

    # Education: institution tier + relevant field of study
    best_edu = 0.0
    relevant_field = False
    for edu in c.get("education", []):
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            best_edu = max(best_edu, 1.5)
        elif tier == "tier_2":
            best_edu = max(best_edu, 0.75)
        field = (edu.get("field_of_study") or "").lower()
        if any(f in field for f in RELEVANT_FIELDS):
            relevant_field = True
    score += best_edu
    if relevant_field:
        score += 0.5

    # Relevant certifications (max +0.5)
    for cert in c.get("certifications", []):
        cert_name = (cert.get("name", "") + " " + cert.get("issuer", "")).lower()
        if any(t in cert_name for t in RELEVANT_CERT_TERMS):
            score += 0.5
            break

    # Accepts offers when extended (+0.5) — hiring-probability signal
    if signals.get("offer_acceptance_rate", -1) >= 0.7:
        score += 0.5

    # Work-mode fit with hybrid Pune/Noida role (+0.25)
    if signals.get("preferred_work_mode") in ("hybrid", "onsite", "flexible"):
        score += 0.25

    return score

# ---------------------------------------------------------------------------
# PHASE 2C: TOP-200 PRODUCTION-RETRIEVAL EVIDENCE RE-RANK
# ---------------------------------------------------------------------------
def production_retrieval_evidence(c: Dict) -> float:
    """Concrete retrieval-in-production evidence inside long-tenure product-
    company jobs — the strongest predictor for the top of the list (NDCG@10).
    Only counts terms appearing in career descriptions, not skill lists."""
    evidence = 0.0
    for job in c.get("career_history", []):
        if job.get("duration_months", 0) < 12:
            continue
        if job.get("industry", "") in SERVICES_INDUSTRIES:
            continue
        if is_service_company(job.get("company", "")):
            continue
        desc = job.get("description", "").lower()
        if not any(t in desc for t in PRODUCTION_TERMS):
            continue
        hits = sum(1 for t in RETRIEVAL_EVIDENCE_TERMS if t in desc)
        evidence += min(hits * 0.5, 2.0)
    return min(evidence, 3.0)

# ---------------------------------------------------------------------------
# PHASE 3: REASONING GENERATION
# ---------------------------------------------------------------------------
VECTOR_DB_TERMS = ("qdrant", "pinecone", "weaviate", "milvus", "faiss",
                   "opensearch", "elasticsearch", "vespa")
EVAL_METRIC_TERMS = ("ndcg", "mrr", "recall@", "a/b test")
METRIC_PRETTY = {"ndcg": "NDCG", "mrr": "MRR", "recall@": "Recall@K", "a/b test": "A/B testing"}
DB_PRETTY = {"faiss": "FAISS", "opensearch": "OpenSearch", "elasticsearch": "Elasticsearch"}
FINETUNE_TERMS = ("lora", "qlora", "peft", "fine-tun")


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def pretty_term(term: str) -> str:
    """Return a display-friendly version of a tech term using known lookup tables."""
    return METRIC_PRETTY.get(term, DB_PRETTY.get(term, term.title()))


def generate_reasoning(c: Optional[Dict], rank: int, cid: str = "") -> str:
    if c is None:  # backfilled row (only occurs when <100 candidates survive)
        return ("Below-bar profile included to complete the shortlist of 100; "
                "does not meet the core retrieval/product-experience requirements.")

    profile = c.get("profile", {}) or {}
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {}) or {}
    skills = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "") or "engineer"
    current_company = profile.get("current_company", "") or "current employer"
    notice = signals.get("notice_period_days", 0)
    qual = compute_qualifying_months(c)

    all_descs = " ".join(job.get("description", "").lower() for job in career)
    skill_names_text = " ".join(s.get("name", "").lower() for s in skills)
    all_text = all_descs + " " + skill_names_text

    sorted_skills = sorted(skills, key=lambda s: clipped_skill_months(s, yoe), reverse=True)
    nlp_skills = [s for s in sorted_skills if s.get("name", "").lower() in NLP_IR_SKILLS]
    top_skills = [s.get("name", "") for s in (nlp_skills[:2] if nlp_skills else sorted_skills[:2])]
    top_skill = top_skills[0] if top_skills else "general ML"

    vec_hits = [t for t in VECTOR_DB_TERMS if t in all_descs]
    metric_hits = [t for t in EVAL_METRIC_TERMS if t in all_descs]
    ft_hits = [t for t in FINETUNE_TERMS if t in all_text]

    # attribute vector-DB evidence to the job whose description actually
    # mentions it — naming the wrong company is a Stage 4 accuracy penalty
    vec_hit_term = None
    vec_hit_company = ""
    vec_hit_current = False
    for job in sorted(career, key=lambda j: not j.get("is_current")):
        desc = job.get("description", "").lower()
        term = next((t for t in VECTOR_DB_TERMS if t in desc), None)
        if term:
            vec_hit_term = term
            vec_hit_company = job.get("company", "")
            vec_hit_current = bool(job.get("is_current"))
            break

    # concern clause, phrased differently per signal
    location = profile.get("location", "").lower()
    near_office = any(city in location for city in IDEAL_LOCATIONS | WELCOME_LOCATIONS)
    if notice > 60:
        concern = f"Main friction: {notice}-day notice period."
    elif signals.get("github_activity_score", -1) == -1:
        concern = "No public GitHub to verify code externally."
    elif not near_office and not signals.get("willing_to_relocate"):
        concern = "Location fit for hybrid Pune/Noida needs a conversation."
    else:
        concern = ""

    # pick a structurally different template based on the candidate's
    # strongest concrete signal + rank band (tone must match position)
    if rank > 60:
        # hedged tone for the back half of the list
        if vec_hits or metric_hits:
            lead_term = pretty_term((vec_hits + metric_hits)[0])
            body = (f"Adjacent rather than exact fit: mentions {lead_term} "
                    f"in past work but with thinner production-retrieval depth than the top of this list; "
                    f"{yoe} yrs total, {qual} mo in product ML roles.")
        else:
            body = (f"Back-half pick — {current_title} with a {top_skill} foundation; "
                    f"embeddings/retrieval depth sits below the top-tier cutoff, "
                    f"though {qual} mo of product-company experience keeps them shortlist-worthy.")
    elif vec_hits:
        db_name = pretty_term(vec_hit_term or vec_hits[0])
        if vec_hit_current and current_company:
            where = f"at {current_company}"
        elif vec_hit_company:
            where = f"in an earlier role at {vec_hit_company}"
        else:
            where = "in a previous product role"
        body = (f"Production {db_name} work described {where} hits the "
                f"vector-database must-have directly; {yoe} yrs overall with {qual} mo "
                f"inside product ML teams.")
    elif metric_hits:
        metric_name = pretty_term(metric_hits[0])
        body = (f"Cites {metric_name} for ranking quality in shipped work — exactly the "
                f"offline-evaluation rigor this role centers on. Currently {current_title} "
                f"with {yoe} yrs of experience.")
    elif ft_hits:
        body = (f"Brings LLM-adaptation depth (LoRA/PEFT-style fine-tuning) on top of "
                f"{top_skill}, covering both the retrieval and fine-tuning sides of the JD; "
                f"{qual} mo product ML.")
    elif rank <= 10:
        body = (f"{yoe}-yr {current_title} at {current_company}: consistent product-company "
                f"record ({qual} mo) with hands-on {top_skill}"
                + (f" and {top_skills[1]}" if len(top_skills) > 1 else "") + " work.")
    else:
        body = (f"Solid mid-list profile — {yoe} yrs anchored by {top_skill}"
                + (f" plus {top_skills[1]}" if len(top_skills) > 1 else "")
                + f", {qual} mo of it in product engineering environments.")

    return f"{body} {concern}".strip()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(candidates_file: str, output_file: str) -> None:
    now = datetime(2026, 6, 14, tzinfo=timezone.utc)  # fixed ref date per dataset

    # pass 1: stream and hard-drop; keep some dropped IDs as backfill insurance
    candidates: List[Dict] = []
    dropped_ids: List[str] = []
    open_func = gzip.open if candidates_file.endswith(".gz") else open
    with open_func(candidates_file, "rt", encoding="utf-8") as f:
        for line in f:
            # cheap pre-filter: country=="India" requires the substring, so
            # non-India lines can be skipped without paying json.loads
            if "India" not in line:
                continue
            line = line.strip()
            if not line:
                continue
            # malformed lines in a user-supplied sample must not kill the run
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(c, dict) or not c.get("candidate_id"):
                continue
            if hard_drop(c, now):
                if len(dropped_ids) < 200:
                    dropped_ids.append(c.get("candidate_id", ""))
                continue
            candidates.append(c)

    # pass 2: TF-IDF similarity over survivors, converted to percentile ranks
    if candidates:
        texts = [build_candidate_text(c) for c in candidates] + [JD_TEXT]
        vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        sims = cosine_similarity(tfidf[:-1], tfidf[-1]).ravel()
        # stable sort so tied similarities rank identically across numpy versions
        percentiles = sims.argsort(kind="stable").argsort(kind="stable").astype(float) / max(len(sims) - 1, 1)
    else:
        percentiles = np.array([])

    survivors = []
    for c, pct in zip(candidates, percentiles):
        qual_months = compute_qualifying_months(c)
        boost = compute_boost_score(c, qual_months, jd_sim_percentile=float(pct))
        penalty = penalty_multiplier(c, now)
        if is_honeypot(c):
            penalty *= 0.01
        survivors.append((c["candidate_id"], boost * penalty, c, penalty))

    # initial sort: score desc, candidate_id asc (tie-break), CSV precision
    sort_key = lambda x: (-round(x[1], 4), x[0])
    survivors.sort(key=sort_key)

    # pass 3: re-rank the head of the list on concrete production-retrieval
    # evidence (NDCG@10 is 50% of the score — sharpen the top specifically).
    # Evidence is scaled by the candidate's penalty so flagged profiles can't
    # ride the bonus, and only added (never subtracted), which keeps the
    # top-200 block's scores above the untouched tail.
    head = [
        (cid, scr + production_retrieval_evidence(c) * pen, c, pen)
        for cid, scr, c, pen in survivors[:200]
    ]
    head.sort(key=sort_key)
    survivors = head + survivors[200:]

    top100 = [(cid, scr, c) for cid, scr, c, _ in survivors[:100]]

    # backfill insurance: if hard drops ever leave <100 survivors, pad with
    # dropped candidates at a floor score so the validator's 100-row rule holds
    floor = min((s for _, s, _ in top100), default=1.0)
    backfill_score = max(round(floor, 4) - 0.0001, 0.0)
    used = {cid for cid, _, _ in top100}
    for did in sorted(dropped_ids):  # ascending cid = validator tie-break order
        if len(top100) >= 100:
            break
        if did and did not in used:
            top100.append((did, backfill_score, None))
            used.add(did)

    with open(output_file, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, scr, cand) in enumerate(top100, start=1):
            reasoning = generate_reasoning(cand, i, cid)
            writer.writerow([cid, i, f"{scr:.4f}", reasoning])

    if len(top100) < 100:
        print(f"WARNING: only {len(top100)} rows written - the submission "
              f"validator requires exactly 100 (fine for small demo samples).",
              file=sys.stderr)
    print(f"Done. Survivors: {len(survivors)} | Top 100 written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--candidates" or sys.argv[3] != "--out":
        print("Usage: python rank.py --candidates <file.jsonl[.gz]> --out <output.csv>")
        sys.exit(1)
    main(sys.argv[2], sys.argv[4])
