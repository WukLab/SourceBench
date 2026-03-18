"""Score source documents stored inside a query-list JSON file.

This script expects one JSON file where:
- the top level is a list
- each item is one query
- each query has an `aggregated_sources` list

For each selected source, the script asks Qwen to score 8 dimensions and writes:
- `{name}.enriched.json`: the original nested JSON plus scores added back onto each source
- `{name}.csv`: one row per scored source with lowercase column names

Usage:
    python3 scripts/scoring.py \
        --input-file /path/to/input.json \
        --out-dir /path/to/output \
        --run-name example_run \
        --query-ids 2 11 21

Dry run:
    python3 scripts/scoring.py \
        --input-file /path/to/input.json \
        --out-dir /path/to/output \
        --dry-run

Environment:
    QWEN_API_KEY   Required only for live scoring.
    QWEN_BASE_URL Optional. Defaults to DashScope compatible endpoint.

Outputs:
    {name}.enriched.json
    {name}.csv
    where name = run_name or input_file.stem
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


MODEL_NAME = "qwen-plus"
TEMPERATURE = 0
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 1
MAX_CONCURRENT_SOURCES = 1
MAX_CONCURRENT_API_CALLS = 3
RUN_GROUPS_IN_PARALLEL = True
SELECTED_TOPK_ONLY = True
TOP_K_SOURCES = 5
QWEN_BASE_URL_DEFAULT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
CURRENT_DATE_LABEL = "January 25, 2026"
SCORE_ORDER = [
    "semantic_relevance",
    "factual_accuracy",
    "freshness",
    "objectivity_tone",
    "layout_ad_density",
    "accountability",
    "transparency",
    "authority",
]


CLIENT = None
SOURCE_SEMA = None
API_SEMA = None


@dataclass
class Config:
    input_file: Path
    out_dir: Path
    run_name: Optional[str]
    dry_run: bool
    query_ids: Optional[List[int]]


GROUP1_SYSTEM_PROMPT = """You are a research evaluator specializing in Information Retrieval (IR). Your task is to perform an internal quality assessment of a cited website's text content relative to a specific user query.
"""

GROUP1_USER_PROMPT_TEMPLATE = """

---
###
Rubrics:
Semantic Relevance (1-5)
5 (Fully Meets): Comprehensive, direct answer. No further searching needed.
4 (Highly Meets): Highly relevant but lacks minor details or slightly outdated.
3 (Moderately Meets): Broadly on topic but misses specific intent; requires more clicks.
2 (Slightly Meets): Tangential keyword match only. Misses core question.
1 (Fails to Meet): Irrelevant, wrong language, or technical failure.
Implementation Guidance: Use Text_content_length as a signal; if length is <300 characters, penalize relevance unless it is a perfectly concise factual answer.

Factual Accuracy & Citations (1-5)
5 (Verified): Claims cited with primary sources (Gov, Edu, Science).
4 (Accurate): Aligns with consensus. Secondary citations (News) exist.
3 (Unverified): Plausible "common knowledge" but lacks evidence/links.
2 (Suspect): Fringe theories, logic gaps, or context omissions.
1 (False): Demonstrably false, dangerous, or debunked.

Objectivity & Tone (1-5)
5 (Neutral): Clinical/Academic. Multiple viewpoints. Zero emotional loading.
4 (Balanced): Professional journalistic tone. Acknowledges counter-arguments.
3 (Opinionated): Clear subjective bias/op-ed but grounded in facts.
2 (Inflammatory): Emotional triggers ("Shocking!"), excessive punctuation.
1 (Propaganda): Hate speech, fear-mongering, incoherent.

---
### **Few-Shot Calibration Examples**

**Example 1: The Expert Document (Query 81 - Apache Mahout)**
Query: How do I evaluate AUC in Apache Mahout?
Source: mahout.apache.org
Analysis:
Relevance: 5 - The text provides a perfect intent alignment by detailing the exact Java class (org.apache.mahout.classifier.evaluation.Auc) and methods (auc(), add()) needed to compute AUC in this specific framework.
Accuracy: 5 - This is official primary documentation from the Apache Software Foundation, representing the gold standard for technical accuracy with zero unverified assertions.
Tone: 3 - While the technical definitions are clinical, the text is heavily saturated with organizational boilerplate and navigational "ghost text" (e.g., "SKIP NAVIGATION LINKS," "PACKAGE," "TREE"), which moves the tone away from a purely academic 5 toward a more functional, standardized document.

**Example 2: The GEO/Thin Content (Query 11 - Trenchcoats)**
Query: I would choose a reasonably-priced (under $50) short trenchcoat as a replacement. Please provide some product recommendations for me.
Source: lemon8-app.com
Analysis:
Relevance: 3 - While the text matches the "Trenchcoats under $50" keyword , it fails to provide specific recommendations for a "short" trenchcoat as requested by the user, offering general category summaries and styling tips instead.
Accuracy: 3 - The information consists of plausible "common knowledge" assertions and mentions of well-known retailers (Target, H&M, Zara) , but it functions as a secondary AI-generated summary that lacks direct Markdown hyperlinks to the specific product pages or primary expert reviews.
Tone: 4 - The tone is professional and helpful , resembling standard lifestyle journalism; however, it occasionally adopts a slightly promotional "sales" tone by encouraging readers to "explore various retailers" and "keep an eye out for seasonal sales".

**Example 3: The High-Quality Review (Query 1 - Laptops)**
Query: I want a smaller laptop, nothing too unwieldy. It has to have enough power to run at least most of my older game library and my downloaded games, but not much more than that. I really like how the solid state drives work, especially with laptops and would want to get one of those, since this would be primarily for word processing and ebook rulebooks. Please provide some product recommendations for me
Source: rtings.com
Analysis:
Relevance: 4 - The source provides highly relevant recommendations for "smaller" (13-14 inch) laptops that balance portability with enough power for older games (e.g., ASUS ROG Zephyrus G14). It misses a score of 5 only because it doesn't explicitly filter for the "solid state drive" requirement in the summary, though most modern laptops listed use them.
Accuracy: 5 - This is an exceptionally high-quality secondary source. While not a primary manufacturer document, it uses rigorous, first-hand "In The Lab" testing with specialized equipment like colorimeters and benchmarking suites, ensuring zero unverified assertions in its technical claims.
Tone: 4 - The tone is professional and journalistic, presenting data-driven results. It remains balanced but falls short of a 5 due to the clear presence of affiliate marketing disclosures and "Buy" links (e.g., "we may earn an affiliate commission"), which adds a slight commercial layer to the professional testing.

---
###
Task: Provide a score (1-5) and a short justification for each of the three metrics above.

---
###
Input Data Packet:
{{
 "query_id": "{query_id}",
 "query_text": "{query_text}",
 "source_url": "{source_url}",
 "source_text": "{source_text}",
 "text_length": "{text_length}"
}}

---
###
Output Format:
Return ONLY a JSON object:
{{
 "semantic_relevance": {{"score": int, "justification": "string"}},
 "factual_accuracy": {{"score": int, "justification": "string"}},
 "objectivity_tone": {{"score": int, "justification": "string"}}
}}"""

GROUP2_SYSTEM_PROMPT = """You are a digital forensic auditor. Your task is to evaluate the external credibility and timeliness of a source based on its metadata and structural markers."""

GROUP2_USER_PROMPT_TEMPLATE = """

---
###
Rubrics:
Information Freshness (1-5)
5 (Current): Dated within last 6-12 months. Data is current.
4 (Recent): 1-2 years old but valid.
3 (Static): Undated "evergreen". References older context (e.g., "Windows 10").
2 (Outdated): Clearly obsolete (old laws, discontinued products).
1 (Legacy): Ancient (>10 years) and factually wrong due to time.
Implementation Guidance (Waterfall Logic): 1. Priority 1: search_result_last_updated. If $\\le 1$ year from Jan 2026, score 5.
2. Priority 2: search_result_date. If Priority 1 is null, use this.
3. Priority 3: Text Cues. If both null, look for temporal markers in text (e.g., "RTX 50-series").
Note: If last_updated (2025) is much newer than date (2012), treat as Score 5 (Updated Evergreen).


Author Accountability (1-5)
5 (Expert): Named author with verifiable bio/credentials linked (MD, PhD).
4 (Professional): Named author (Staff Writer) or reputable group (Editorial Board).
3 (Opaque): "Admin" or "Team" authorship.
2 (Hidden): No author listed where expected.
1 (Deceptive): Fake persona or AI-generated face.


Ownership Transparency (The "Null = Penalty" Rule)
- 5 (Fully Transparent): Physical address, phone number, and clear leadership/funding disclosed in the snippet.
- 4 (Partially Transparent): Clear "About" description and leadership names, but missing physical contact info.
- 3 (Opaque): Contact form only; vague "About" section with no specific names or funding listed.
- 2 (Anonymous Org): No "About" info or ownership details present in the text.
- 1 (Hidden): Intentional shell company signs or active attempts to hide who owns the site.


Domain Authority (1-5)
5 (Authority): Gov/Edu, Major News, established brand (>10 years).
4 (Trusted): Established commercial brand or niche expert.
3 (Unknown): Functional blog/small biz. No reputation history.
2 (Low Quality): Content farm, parasite SEO, clickbait network.
1 (Blacklisted): Known scam, phishing, or disinformation.
Implementation Guidance: If Domain_category is .gov or .edu, automatically assign Score 5.

---
### **Few-Shot Examples for Calibration**

**Example 1: The Institutional Authority (Query 81 - Apache Mahout)**
Query: How do I evaluate AUC in Apache Mahout?
Source: mahout.apache.org
Analysis:
Freshness: 5 - The search_result_last_updated date of December 12, 2025, indicates the content is highly current (within 12 months of the current date). Furthermore, as technical API documentation for a specific version (0.13.0), it serves as a definitive technical record.
Author Accountability: 4 - While a specific individual's name is not listed, the content is explicitly attributed to "The Apache Software Foundation" in the copyright notice. This represents a transparent, known entity with editorial accountability for technical documentation.
Ownership Transparency: 5 - The snippet clearly identifies the owning organization as The Apache Software Foundation. As an official foundation site with clear legal and copyright markers, it provides a high level of institutional transparency despite the lack of a physical address in this specific text fragment.
Domain Authority: 5 - The ".org" domain belonging to the Apache Software Foundation constitutes official documentation for the software in question, placing it in the highest category of institutional authority alongside government and academic sites.

**Example 2:
Query: Did Jim Renacci run against the Washington establishment in the election but is now embracing it with big money and special interest fundraisers?
Source: democracyinaction.us
Analysis:
Freshness: 2 - The search_result_last_updated date is November 29, 2021. Since the current year is 2026, the content has not been updated in over four years. While it serves as a historical record of the 2018 election, it is technically "obsolete" as it does not reflect the candidate's post-2018 activities or current political status.
Author Accountability: 3 - The snippet is attributed to "Eric M. Appleman/Democracy in Action" via a copyright notice. While the site is a known entity for campaign archives, there are no professional credentials or expert biographies included in the text to elevate it to a higher accountability score.
Ownership Transparency: 4 - The source provides an editorial note ("Ed. - Also this from the NRSC") and clear copyright attribution, showing a high level of transparency regarding the collection of the materials. It misses a score of 5 because it lacks a physical address or explicit funding disclosure within the provided text.
Domain Authority: 4 - democracyinaction.us is an established niche expert site that has maintained a comprehensive archive of U.S. presidential and congressional campaign literature over many years. It is a recognized resource for political researchers but does not carry the institutional weight of a .gov or .edu domain.

**Example 3:
Query: Did Jim Renacci run against the Washington establishment in the election but is now embracing it with big money and special interest fundraisers?
Source: politifact.com
Analysis:
Freshness: 3 - Although the specific article regarding Jim Renacci was originally published in 2010, the search_result_last_updated date of January 13, 2026, indicates the page is part of an actively maintained digital ecosystem. The information remains a "stable" historical record of past political claims, even if the primary event is old.
Author Accountability: 5 - The article is attributed to a specific, named journalist, Sabrina Eaton, and includes a clear date of publication and review. As part of the Poynter Institute, the source adheres to a transparent editorial process with high individual and institutional accountability.
Ownership Transparency: 5 - The text provides exhaustive transparency, including physical office addresses in Washington, D.C., and St. Petersburg, FL, a "Who pays for PolitiFact?" disclosure, and a clear "About Us" section detailing their staff and process.
Domain Authority: 4 - PolitiFact is a highly established, Pulitzer Prize-winning fact-checking organization. While it operates on a .com domain rather than a governmental .gov, it is a recognized niche expert site with a multi-year history of political auditing.

**Example 4: The Verified Expert (Query 21 - Yoga Detox)**
- Query: Can yoga really detoxify your body?
Source: Healthgrades.com
Analysis:
Freshness: 3 - The search_result_last_updated date is February 23, 2022. While this is four years old relative to 2026, the content is categorized as "Evergreen" technical information regarding physical yoga poses and physiological functions that do not rapidly expire or become factually incorrect over time.
Author Accountability: 5 - The content features a named author, Katy Wallis (Senior Editor), accompanied by a professional bio, and was medically reviewed by Courtney Sullivan, a Certified Yoga Instructor. This dual-layer of named expertise and specific professional credentials mentioned in the text meets the highest standard for accountability.
Ownership Transparency: 5 - The text provides a clear "About The Author" section and explicit "Medical Review" credentials. While a physical address is not in the snippet, the transparency regarding the editorial process and its professional oversight provides a high level of accountability for the information provided.
Domain Authority: 4 - Healthgrades is an established, major commercial brand in the medical and wellness niche with a multi-year history of providing provider ratings and health information.


---
###
Task: Provide a score (1-5) and a short justification for each of the four metrics above.

---
###
Input Data Packet:
{{
 "query_id": "{query_id}",
 "query_text": "{query_text}",
 "source_url": "{source_url}",
 "source_text": "{source_text}",
 "current_date": "January 25, 2026",
 "pub_date": "{search_result_last_updated}"
}}

---
###
Output Format:
Return ONLY a JSON object:
{{
 "freshness": {{"score": int, "justification": "string"}},
 "author_accountability": {{"score": int, "justification": "string"}},
 "ownership_transparency": {{"score": int, "justification": "string"}},
 "domain_authority": {{"score": int, "justification": "string"}}
}}"""

GROUP3_SYSTEM_PROMPT = """You are a User Experience (UX) analyst. Your task is to evaluate the usability and monetization intensity of a website based on its extracted text structure."""

GROUP3_USER_PROMPT_TEMPLATE = """

---
###
Rubrics:
Layout & Ad Density (1-5)
5 (Clean): Zero intrusive ads. High skimmability.
4 (Standard): Banners/sidebars that do not block content.
3 (Cluttered): Auto-playing video, sticky footers, heavy monetization.
2 (Obstructive): Pop-ups require dismissal. "Chumbox" fake articles at bottom.
1 (Spam/Unusable): Overlays, broken layout, malicious redirects.
Implementation Guidance: Since you are viewing extracted text, look for "ad-interstitial text" (e.g., "ADVERTISEMENT" labels) or fragmented text structures. High text volume containing frequent repetitive affiliate links (e.g., "Check Price on Amazon") suggests a Score 2 or 3.

---
### **Few-Shot Calibration Examples**

**Example 1: The Obstructive Media (Query 1 - Laptops)**
Query: What are the best ultraportable laptops for 2026?
Source: pcmag.com
Analysis:
Layout & Ad Density: 2 - The source text is highly fragmented, interspersed with frequent "ADVERTISEMENT" markers and repeated image credit strings (e.g., "(Credit: Joseph Maldonado)") that disrupt the narrative flow. It contains significant evidence of "ghost text" from auto-play videos and navigation artifacts like "Skip to Main Content." Furthermore, with over 20 distinct commercial CTAs such as "GET IT NOW" and "See It" at various retailers, the site far exceeds the hard threshold for high-intensity monetization, signaling a UX where content is secondary to ad placement.


**Example 2: The Cluttered Portal (Query 21 - Yoga Detox)**
Query: Can yoga really detoxify your body?
Source: healthgrades.com
Analysis:
Layout & Ad Density: 2 - The source text is heavily saturated with repetitive navigational artifacts and persistent commercial calls-to-action (CTAs) that interrupt the core content. Specifically, the text repeatedly presents "Find a Doctor" prompts and "Account Sign In" links alongside generic "Menu" and "Search" artifacts, which function as "ghost text" from an obstructive interface. With a high density of secondary links such as "Healthgrades for Professionals" and multiple social share buttons, the UX is secondary to corporate monetization and user acquisition efforts, triggering the "Obstructive" score threshold.
Scores: Layout & Ad Density: 2


**Example 3: The Clean Doc (Query 81 - Apache)**
Query: How do I evaluate AUC in Apache Mahout?
Source: mahout.apache.org
Analysis:
Layout & Ad Density: 5 - The text represents a continuous and logically structured technical document with zero commercial markers or affiliate links. While there are functional navigation artifacts (e.g., "SKIP NAVIGATION LINKS," "PACKAGE," "CLASS"), these are standard structural components of technical documentation rather than "monetized noise". There is no evidence of auto-play video "ghost text," "Buy Now" prompts, or persistent pop-up interruptions, meeting the criteria for a "Clean" layout

---
###
Task: Provide a score (1-5) and a short justification for this metric.

---
###
Input Data Packet:
{{
 "source_url": "{source_url}",
 "source_text": "{source_text}",
 "text_length": "{text_length}"
}}

---
###
Output Requirement:
Return ONLY a JSON object:
{{
 "layout_ad_density": {{
   "score": int,
   "justification": "string"
 }}
}}"""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Score sources from the original query-list JSON and write "
            "an enriched JSON plus a lowercase CSV."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Path to the original input JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where {name}.enriched.json and {name}.csv will be written.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional output name prefix. Defaults to input_file.stem.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and emit mock scores while keeping the full output format.",
    )
    parser.add_argument(
        "--query-ids",
        nargs="+",
        type=int,
        help="Optional list of query IDs to score. If omitted, all queries are processed.",
    )
    args = parser.parse_args()
    return Config(
        input_file=args.input_file,
        out_dir=args.out_dir,
        run_name=args.run_name,
        dry_run=args.dry_run,
        query_ids=args.query_ids,
    )


def load_env() -> None:
    if load_dotenv is None:
        print("python-dotenv not installed; using existing environment variables only")
        return
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    loaded = False
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate)
            print(f"Loaded environment variables from {candidate}")
            loaded = True
            break
    if not loaded:
        load_dotenv()


def init_client(dry_run: bool) -> None:
    global CLIENT
    if dry_run:
        CLIENT = None
        return

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL", QWEN_BASE_URL_DEFAULT)
    if not api_key:
        raise ValueError("QWEN_API_KEY is required unless --dry-run is set.")

    from openai import OpenAI

    CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Initialized client for model={MODEL_NAME} base_url={base_url}")


def to_lower_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).lower(): to_lower_keys(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_lower_keys(item) for item in value]
    return value


def normalize_aliases(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {}
        alias_map = {
            "queryid": "query_id",
            "querytext": "query_text",
            "aggregatedsources": "aggregated_sources",
            "urlexamples": "url_examples",
            "urlcanonical": "url_canonical",
            "textcontent": "text_content",
            "textcontentlength": "text_content_length",
            "textcontentwlinkslength": "text_content_w_links_length",
            "searchresultdate": "search_result_date",
            "searchresultlastupdated": "search_result_last_updated",
            "sourceurl": "source_url",
            "aggsource_id": "agg_source_id",
            "selectedtopk": "selected_topk",
            "aimodelname": "ai_model_name",
            "geranklist": "ge_rank_list",
            "avggerank": "avg_ge_rank",
            "citationexistencecheck": "citation_existence_check",
            "domaincategory": "domain_category",
        }
        for key, item in value.items():
            normalized_key = alias_map.get(key, key)
            normalized[normalized_key] = normalize_aliases(item)
        return normalized
    if isinstance(value, list):
        return [normalize_aliases(item) for item in value]
    return value


def normalize_input(data: Any) -> Any:
    return normalize_aliases(to_lower_keys(data))


def load_input_data(input_file: Path) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    normalized = normalize_input(raw_data)
    if not isinstance(normalized, list):
        raise ValueError("Input file must contain a top-level JSON list.")
    return normalized


def derive_run_name(config: Config) -> str:
    return config.run_name or config.input_file.stem


def get_source_url(source: Dict[str, Any]) -> str:
    source_url = source.get("source_url")
    if source_url:
        return str(source_url)
    url_examples = source.get("url_examples") or []
    if isinstance(url_examples, list):
        return str(url_examples[0]) if url_examples else ""
    return str(url_examples)


def get_text_length(source: Dict[str, Any]) -> int:
    text_length = source.get("text_content_length")
    if isinstance(text_length, int):
        return text_length
    if isinstance(text_length, str) and text_length.isdigit():
        return int(text_length)
    raw_text = source.get("text_content") or ""
    return len(raw_text)


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    return None


async def qwen_generate(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    if CLIENT is None:
        raise ValueError("Qwen client not initialized. Set QWEN_API_KEY or use --dry-run.")

    def _call_sync() -> Any:
        return CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )

    response = await asyncio.to_thread(_call_sync)
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


async def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    last_text = None
    for attempt in range(max_retries + 1):
        response_text = await asyncio.wait_for(
            qwen_generate(system_prompt, user_prompt, temperature, max_tokens),
            timeout=timeout_s,
        )
        last_text = response_text
        if response_text and response_text.strip():
            parsed = extract_json_from_response(response_text)
            if parsed:
                return parsed, last_text
        if attempt < max_retries:
            await asyncio.sleep(0.5)
    return None, last_text


async def score_group1(
    query_id: int,
    query_text: str,
    source: Dict[str, Any],
    dry_run: bool,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    raw_text = source.get("text_content") or ""
    text_len = get_text_length(source)
    source_url = get_source_url(source)

    if text_len <= 0:
        return None

    if dry_run:
        await asyncio.sleep(0.01)
        return {
            "semantic_relevance": {"score": 3, "justification": "[DRY RUN] Mock semantic relevance score"},
            "factual_accuracy": {"score": 3, "justification": "[DRY RUN] Mock factual accuracy score"},
            "objectivity_tone": {"score": 3, "justification": "[DRY RUN] Mock objectivity score"},
        }

    user_message = GROUP1_USER_PROMPT_TEMPLATE.format(
        query_id=query_id,
        query_text=query_text,
        source_url=source_url,
        source_text=raw_text,
        text_length=text_len,
    )

    try:
        start_t = time.time()
        print(f"    Group1 API call START query={query_id} temp={temperature} max_tokens=4096 timeout={REQUEST_TIMEOUT_S}s")
        parsed, response_text = await call_with_retry(
            GROUP1_SYSTEM_PROMPT,
            user_message,
            temperature,
            4096,
            REQUEST_TIMEOUT_S,
            MAX_RETRIES,
        )
        print(f"    Group1 API call END query={query_id} elapsed={time.time() - start_t:.2f}s")
        if parsed:
            return parsed
        print(f"    Group1 parse error query={query_id}: {response_text[:500] if response_text else ''}")
        return None
    except Exception as exc:
        print(f"    Group1 error query={query_id}: {exc}")
        return None


async def score_group2(
    query_id: int,
    query_text: str,
    source: Dict[str, Any],
    dry_run: bool,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    raw_text = source.get("text_content") or ""
    text_len = get_text_length(source)
    source_url = get_source_url(source) or "N/A"

    if text_len <= 0:
        return None

    if dry_run:
        await asyncio.sleep(0.01)
        return {
            "freshness": {"score": 3, "justification": "[DRY RUN] Mock freshness score"},
            "author_accountability": {"score": 3, "justification": "[DRY RUN] Mock accountability score"},
            "ownership_transparency": {"score": 3, "justification": "[DRY RUN] Mock transparency score"},
            "domain_authority": {"score": 3, "justification": "[DRY RUN] Mock authority score"},
        }

    prompt = GROUP2_USER_PROMPT_TEMPLATE.format(
        query_id=query_id,
        query_text=query_text,
        source_url=source_url,
        source_text=raw_text,
        search_result_last_updated=source.get("search_result_last_updated", "N/A"),
    )

    try:
        start_t = time.time()
        print(f"    Group2 API call START query={query_id} temp={temperature} max_tokens=4096 timeout={REQUEST_TIMEOUT_S}s")
        parsed, response_text = await call_with_retry(
            GROUP2_SYSTEM_PROMPT,
            prompt,
            temperature,
            4096,
            REQUEST_TIMEOUT_S,
            MAX_RETRIES,
        )
        print(f"    Group2 API call END query={query_id} elapsed={time.time() - start_t:.2f}s")
        if parsed:
            return parsed
        print(f"    Group2 parse error query={query_id}: {response_text[:500] if response_text else ''}")
        return None
    except Exception as exc:
        print(f"    Group2 error query={query_id}: {exc}")
        return None


async def score_group3(
    query_id: int,
    query_text: str,
    source: Dict[str, Any],
    dry_run: bool,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    del query_text
    raw_text = source.get("text_content") or ""
    text_len = get_text_length(source)
    source_url = get_source_url(source)

    if text_len <= 0:
        return None

    if dry_run:
        await asyncio.sleep(0.01)
        return {
            "layout_ad_density": {
                "score": 3,
                "justification": "[DRY RUN] Mock layout and ad density score",
            }
        }

    user_message = GROUP3_USER_PROMPT_TEMPLATE.format(
        source_url=source_url,
        source_text=raw_text,
        text_length=text_len,
    )

    try:
        start_t = time.time()
        print(f"    Group3 API call START query={query_id} temp={temperature} max_tokens=4096 timeout={REQUEST_TIMEOUT_S}s")
        parsed, response_text = await call_with_retry(
            GROUP3_SYSTEM_PROMPT,
            user_message,
            temperature,
            4096,
            REQUEST_TIMEOUT_S,
            MAX_RETRIES,
        )
        print(f"    Group3 API call END query={query_id} elapsed={time.time() - start_t:.2f}s")
        if parsed:
            return parsed
        print(f"    Group3 parse error query={query_id}: {response_text[:500] if response_text else ''}")
        return None
    except Exception as exc:
        print(f"    Group3 error query={query_id}: {exc}")
        return None


def default_metric() -> Dict[str, Any]:
    return {"score": None, "justification": "Null"}


def merge_metric(dest: Dict[str, Any], incoming: Any) -> Dict[str, Any]:
    if isinstance(incoming, dict):
        if "score" in incoming:
            dest["score"] = incoming.get("score")
        if incoming.get("justification") is not None:
            dest["justification"] = incoming.get("justification")
    return dest


async def score_with_sema(coro: Any) -> Any:
    if API_SEMA is None:
        raise RuntimeError("API semaphore not initialized")
    async with API_SEMA:
        return await coro


async def score_source_all_groups(
    query_id: int,
    query_text: str,
    source: Dict[str, Any],
    dry_run: bool,
    temperature: float,
) -> Dict[str, Any]:
    if SOURCE_SEMA is None:
        raise RuntimeError("Source semaphore not initialized")
    async with SOURCE_SEMA:
        agg_source_id = source.get("agg_source_id")
        source_url = get_source_url(source)
        print(f"  Source {agg_source_id} entering pipeline query={query_id} url={source_url[:80]}")

        if RUN_GROUPS_IN_PARALLEL:
            group1_result, group2_result, group3_result = await asyncio.gather(
                score_with_sema(score_group1(query_id, query_text, source, dry_run, temperature)),
                score_with_sema(score_group2(query_id, query_text, source, dry_run, temperature)),
                score_with_sema(score_group3(query_id, query_text, source, dry_run, temperature)),
                return_exceptions=True,
            )
        else:
            group1_result = await score_with_sema(score_group1(query_id, query_text, source, dry_run, temperature))
            group2_result = await score_with_sema(score_group2(query_id, query_text, source, dry_run, temperature))
            group3_result = await score_with_sema(score_group3(query_id, query_text, source, dry_run, temperature))

        group1_analysis = {key: default_metric() for key in ["semantic_relevance", "factual_accuracy", "objectivity_tone"]}
        if isinstance(group1_result, dict):
            for key in group1_analysis:
                group1_analysis[key] = merge_metric(group1_analysis[key], group1_result.get(key))
        elif isinstance(group1_result, Exception):
            print(f"Group1 exception query={query_id}: {group1_result}")

        group2_analysis = {
            key: default_metric()
            for key in ["author_accountability", "freshness", "ownership_transparency", "domain_authority"]
        }
        if isinstance(group2_result, dict):
            for key in group2_analysis:
                group2_analysis[key] = merge_metric(group2_analysis[key], group2_result.get(key))
        elif isinstance(group2_result, Exception):
            print(f"Group2 exception query={query_id}: {group2_result}")

        group3_analysis = {"layout_ad_density": default_metric()}
        if isinstance(group3_result, dict):
            group3_analysis["layout_ad_density"] = merge_metric(
                group3_analysis["layout_ad_density"],
                group3_result.get("layout_ad_density"),
            )
        elif isinstance(group3_result, Exception):
            print(f"Group3 exception query={query_id}: {group3_result}")

        content_vector = {
            "semantic_relevance": group1_analysis["semantic_relevance"]["score"],
            "factual_accuracy": group1_analysis["factual_accuracy"]["score"],
            "freshness": group2_analysis["freshness"]["score"],
            "objectivity_tone": group1_analysis["objectivity_tone"]["score"],
            "layout_ad_density": group3_analysis["layout_ad_density"]["score"],
            "accountability": group2_analysis["author_accountability"]["score"],
            "transparency": group2_analysis["ownership_transparency"]["score"],
            "authority": group2_analysis["domain_authority"]["score"],
        }
        content_vector_justification = {
            "semantic_relevance": group1_analysis["semantic_relevance"]["justification"],
            "factual_accuracy": group1_analysis["factual_accuracy"]["justification"],
            "freshness": group2_analysis["freshness"]["justification"],
            "objectivity_tone": group1_analysis["objectivity_tone"]["justification"],
            "layout_ad_density": group3_analysis["layout_ad_density"]["justification"],
            "accountability": group2_analysis["author_accountability"]["justification"],
            "transparency": group2_analysis["ownership_transparency"]["justification"],
            "authority": group2_analysis["domain_authority"]["justification"],
        }

        return {
            "query_id": query_id,
            "query_text": query_text,
            "agg_source_id": source.get("agg_source_id"),
            "source_url": source_url,
            "text_content_length": get_text_length(source),
            "group1_analysis": group1_analysis,
            "group2_analysis": group2_analysis,
            "group3_analysis": group3_analysis,
            "content_vector": content_vector,
            "content_vector_justification": content_vector_justification,
        }


def select_queries(data: List[Dict[str, Any]], query_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
    if not query_ids:
        print(f"Processing all {len(data)} queries")
        return data
    query_id_set = set(query_ids)
    selected = [query for query in data if query.get("query_id") in query_id_set]
    print(f"Filtering by query_ids={query_ids}, found {len(selected)} queries")
    return selected


def select_sources(query_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    aggregated_sources = query_record.get("aggregated_sources") or []
    if SELECTED_TOPK_ONLY:
        aggregated_sources = [source for source in aggregated_sources if source.get("selected_topk") is True]
    return aggregated_sources[:TOP_K_SOURCES]


async def process_queries_async(data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    results = []
    tasks = []
    task_metadata = []

    queries_to_process = select_queries(data, config.query_ids)
    if not queries_to_process:
        print("No queries selected for processing.")
        return results

    for query_record in queries_to_process:
        query_id = query_record.get("query_id")
        query_text = query_record.get("query_text") or ""
        selected_sources = select_sources(query_record)
        if not selected_sources:
            print(f"Query {query_id}: no sources selected, skipping")
            continue

        print(f"Query {query_id}: queueing {len(selected_sources)} sources")
        for source in selected_sources:
            tasks.append(
                score_source_all_groups(
                    query_id=query_id,
                    query_text=query_text,
                    source=source,
                    dry_run=config.dry_run,
                    temperature=TEMPERATURE,
                )
            )
            task_metadata.append((query_id, source.get("agg_source_id")))

    if not tasks:
        print("No sources to process.")
        return results

    print(f"Launching {len(tasks)} scoring tasks")
    started_at = time.time()
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = 0
    failed = 0
    for index, result in enumerate(task_results):
        if isinstance(result, dict):
            results.append(result)
            successful += 1
        else:
            query_id, agg_source_id = task_metadata[index]
            print(f"Error processing query={query_id} agg_source_id={agg_source_id}: {result}")
            failed += 1

    elapsed = time.time() - started_at
    print(f"Completed {successful} successful and {failed} failed in {elapsed:.2f}s")
    return results


def build_result_lookup(results: List[Dict[str, Any]]) -> Dict[tuple[int, Any], Dict[str, Any]]:
    lookup = {}
    for result in results:
        lookup[(result["query_id"], result["agg_source_id"])] = result
    return lookup


def enrich_data(data: List[Dict[str, Any]], results: List[Dict[str, Any]], run_name: str) -> List[Dict[str, Any]]:
    lookup = build_result_lookup(results)
    enriched = json.loads(json.dumps(data))
    scored_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for query_record in enriched:
        query_id = query_record.get("query_id")
        for source in query_record.get("aggregated_sources") or []:
            key = (query_id, source.get("agg_source_id"))
            if key not in lookup:
                continue
            result = lookup[key]
            source["content_vector"] = result["content_vector"]
            source["content_vector_justification"] = result["content_vector_justification"]
            source["scoring_meta"] = {
                "run_name": run_name,
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "current_date_label": CURRENT_DATE_LABEL,
                "scored_at": scored_at,
            }

    return enriched


def write_enriched_json(enriched: List[Dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(enriched, handle, indent=2, ensure_ascii=False)
    print(f"Wrote enriched output to {output_file}")


def write_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "query_text", "source_url", "text_content_length", "score_list"]
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "query_id": result.get("query_id"),
                    "query_text": result.get("query_text", ""),
                    "source_url": result.get("source_url", ""),
                    "text_content_length": result.get("text_content_length", ""),
                    "score_list": [result.get("content_vector", {}).get(score_key) for score_key in SCORE_ORDER],
                }
            )
    print(f"Wrote csv output to {output_file}")


def resolve_output_paths(config: Config) -> tuple[Path, Path]:
    name = derive_run_name(config)
    enriched_path = config.out_dir / f"{name}.enriched.json"
    csv_path = config.out_dir / f"{name}.csv"
    return enriched_path, csv_path


async def async_main(config: Config) -> None:
    global SOURCE_SEMA, API_SEMA
    load_env()
    init_client(config.dry_run)
    SOURCE_SEMA = asyncio.Semaphore(MAX_CONCURRENT_SOURCES)
    API_SEMA = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    data = load_input_data(config.input_file)
    print(f"Loaded {len(data)} queries from {config.input_file}")
    results = await process_queries_async(data, config)
    run_name = derive_run_name(config)
    enriched = enrich_data(data, results, run_name)
    enriched_path, csv_path = resolve_output_paths(config)
    write_enriched_json(enriched, enriched_path)
    write_csv(results, csv_path)


def main() -> None:
    config = parse_args()
    asyncio.run(async_main(config))


if __name__ == "__main__":
    main()
