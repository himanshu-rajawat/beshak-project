"""
Generic regex + spaCy extraction for insurance policy documents.
Works across insurers (Star Health, HDFC Ergo, Niva Bupa, ICICI Lombard, etc.)
by using priority-ordered synonym lists rather than hardcoded insurer labels.

Public interface: extract_policy_fields(text: str) -> dict
"""
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy lazy singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with: python -m spacy download en_core_web_sm"
        )


# ---------------------------------------------------------------------------
# Core shared helper
# ---------------------------------------------------------------------------

def _label_value(
    text: str,
    labels: list[str],
    value_pattern: str = r"(.+?)(?:\n|$)",
) -> str | None:
    """
    Try each label synonym in priority order.
    Returns the first captured group of value_pattern, stripped, or None.
    """
    for label in labels:
        escaped = re.escape(label)
        m = re.search(rf"(?i){escaped}\s*[:\-_#]?\s*{value_pattern}", text)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Policy number
# ---------------------------------------------------------------------------

def _extract_policy_number(text: str) -> str | None:
    labels = [
        "Policy No", "Policy Number", "Policy #", "Endorsement No",
        "Certificate No", "Policy Ref", "Policy Reference", "Policy Id",
        "Policy ID",
    ]
    val = _label_value(text, labels, r"([A-Z0-9][A-Z0-9/\-]{4,})")
    if val:
        return val.split()[0]  # grab first token only

    # Generic slash-separated code: e.g. P/123456/01/2024/000001
    m = re.search(r"\b([A-Z]{1,4}/\d{4,}/\d{2,}/\d{2,4}/\d+)\b", text)
    if m:
        return m.group(1)

    # Long alphanumeric: e.g. HDFC0012345678
    m = re.search(r"\b([A-Z]{2,}\d{6,}(?:[/\-]\d+)*)\b", text)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Policy type
# ---------------------------------------------------------------------------

def _extract_policy_type(text: str, doc) -> str | None:
    labels = [
        "Plan Name", "Type of Policy", "Product Name", "Policy Type",
        "Policy Name", "Insurance Plan", "Scheme Name", "Plan",
    ]
    val = _label_value(text, labels)
    if val:
        # Trim at tab or 2+ spaces (common in PDFs with tabular layout)
        val = re.split(r"\s{2,}|\t", val)[0].strip()
        return val if len(val) > 2 else None

    # spaCy fallback: ORG/PRODUCT entity near insurance/plan keyword
    try:
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT") and re.search(
                r"(?i)insurance|plan|health|policy", ent.text
            ):
                return ent.text
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_DATE_VALUE_RE = (
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"   # DD/MM/YYYY or DD-MM-YYYY
    r"|\d{1,2}\s+\w{3,9}\s+\d{4}"           # DD Month YYYY
    r"|\w{3,9}\s+\d{1,2},?\s+\d{4})"        # Month DD, YYYY
)


def _extract_dates(text: str, doc) -> dict:
    """Returns dict with issue_date, renewal_date, period (all may be None)."""
    result: dict = {"issue_date": None, "renewal_date": None, "period": None}

    issue_labels = [
        "Issue Date", "Date of Issue", "Date of Inception", "Commencement Date",
        "Policy Start Date", "Policy Issuance Date", "Start Date", "Policy Date",
        "Inception Date",
    ]
    renewal_labels = [
        "Renewal Date", "Policy Renewal Date", "Expiry Date", "Policy End Date",
        "Policy Expiry Date", "Date of Expiry", "Valid To", "Valid Upto",
        "Expiry", "End Date", "Policy Valid Till", "Valid Till", "Due Date",
        "Period To",
    ]

    result["issue_date"] = _label_value(text, issue_labels, _DATE_VALUE_RE)
    result["renewal_date"] = _label_value(text, renewal_labels, _DATE_VALUE_RE)

    # Narrow single-word labels only when specific labels failed
    if not result["issue_date"]:
        result["issue_date"] = _label_value(text, ["From"], _DATE_VALUE_RE)
    if not result["renewal_date"]:
        result["renewal_date"] = _label_value(text, ["To"], _DATE_VALUE_RE)

    # Period assembled from "From: X To: Y" on same or adjacent lines
    m = re.search(
        rf"(?i)\bFrom\s*[:\-]?\s*{_DATE_VALUE_RE}[\s\n]+To\s*[:\-]?\s*{_DATE_VALUE_RE}",
        text,
    )
    if m:
        if not result["issue_date"]:
            result["issue_date"] = m.group(1)
        if not result["renewal_date"]:
            result["renewal_date"] = m.group(2)
        result["period"] = f"{m.group(1)} to {m.group(2)}"
    elif result["issue_date"] and result["renewal_date"]:
        result["period"] = f"{result['issue_date']} to {result['renewal_date']}"

    # spaCy DATE entity fallback
    if not result["issue_date"] or not result["renewal_date"]:
        try:
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            if not result["issue_date"] and dates:
                result["issue_date"] = dates[0]
            if not result["renewal_date"] and len(dates) > 1:
                result["renewal_date"] = dates[-1]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Money amounts
# ---------------------------------------------------------------------------

_AMOUNT_RE = r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)"
_PLAIN_AMOUNT_RE = r"([\d,]{4,}(?:\.\d{1,2})?)"


def _extract_amount(text: str, labels: list[str]) -> str | None:
    """Try currency pattern first; fall back to bare digits."""
    val = _label_value(text, labels, _AMOUNT_RE)
    if val:
        return val
    return _label_value(text, labels, _PLAIN_AMOUNT_RE)


def _extract_premium(text: str) -> str | None:
    labels = [
        "Total Premium", "Net Premium", "Premium Amount", "Premium Payable",
        "Gross Premium", "Total Amount Payable", "Amount Payable", "Premium",
    ]
    return _extract_amount(text, labels)


def _extract_sum_insured(text: str) -> str | None:
    labels = [
        "Sum Insured", "Sum Assured", "Total Sum Insured", "Basic Sum Insured",
        "Coverage Amount", "Cover Amount", "Total Cover", "Total Coverage",
        "Insured Amount",
    ]
    return _extract_amount(text, labels)


def _extract_bonus(text: str) -> str | None:
    labels = [
        "Cumulative Bonus", "No Claim Bonus", "No-Claim Bonus",
        "Accumulated Bonus", "NCB", "Bonus",
    ]
    return _label_value(text, labels, r"(\d[\d,]*(?:\.\d{1,2})?%?|Nil|NIL|None|NA)")


def _extract_limit_of_coverage(text: str) -> str | None:
    labels = [
        "Limit of Coverage", "Maximum Limit", "Floater Limit",
        "Cover Limit", "Annual Limit",
    ]
    return _extract_amount(text, labels)


def _extract_recharge_benefit(text: str) -> str | None:
    labels = [
        "Recharge Benefit", "Restore Benefit", "Reinstatement Benefit",
        "Recharge", "Restore", "Reinstatement",
    ]
    return _label_value(text, labels)


def _extract_payment_frequency(text: str) -> str | None:
    labels = [
        "Payment Frequency", "Premium Frequency", "Mode of Payment",
        "Payment Mode", "Premium Payment Mode",
    ]
    val = _label_value(text, labels)
    if not val:
        return None

    v_up = val.upper()
    freq_map = {
        "ANNUAL": "Annual",
        "YEARLY": "Annual",
        "MONTHLY": "Monthly",
        "QUARTERLY": "Quarterly",
        "HALF-YEARLY": "Half-Yearly",
        "HALFYEARLY": "Half-Yearly",
        "HALF YEARLY": "Half-Yearly",
        "SINGLE": "Single",
    }
    for key, normalized in freq_map.items():
        if key in v_up:
            return normalized
    return val.split()[0]  # return first word as fallback


# ---------------------------------------------------------------------------
# Personal details — helpers
# ---------------------------------------------------------------------------

_NAME_REJECTS_RE = re.compile(
    r"(?i)\b(?:Card|Permanent|Exclusion|Health|Medical|Insurance|Insured|"
    r"Cover|Benefit|Claim|Premium|Nominee|Sector|Urban|Rural|"
    r"Download|Read|Portal|Approved|Schedule|Details|Plan|Product|"
    r"Coverage|Floater|Limit|Recharge|Restore)\b"
)


def _clean_name(raw: str) -> str | None:
    """Strip trailing noise, cap at 5 words, validate as name."""
    raw = raw.strip()
    raw = re.split(r"\s{3,}|\t|\n", raw)[0]
    raw = re.split(
        r"(?i)\s+(?:SAC|Code|Policy|Number|Address|Age|DOB|Date|Gender|M/F|Nominee)",
        raw,
    )[0]
    raw = raw.strip()
    words = raw.split()
    if 2 <= len(words) <= 5 and re.match(r"^[A-Za-z .'-]+$", raw):
        if _NAME_REJECTS_RE.search(raw):
            return None
        return raw
    return None


# ---------------------------------------------------------------------------
# Proposer name
# ---------------------------------------------------------------------------

def _extract_proposer_name(text: str, doc) -> str | None:
    labels = [
        "Proposer Name", "Policyholder Name", "Policy Holder Name",
        "Policy Holder", "Customer Name", "Insured Name",
        "Name of Proposer", "Name of Insured", "CustomerName_",
        "Name of Policy Holder",
    ]

    # -----------------------------
    # 1. Regex candidates
    # -----------------------------
    regex_names = []

    for label in labels:
        val = _label_value(text, [label], r"([A-Za-z][A-Za-z .'-]{3,})")
        if val:
            cleaned = _clean_name(val)
            if cleaned:
                regex_names.append(cleaned)

    regex_names = set(regex_names)

    # -----------------------------
    # 2. spaCy PERSON entities
    # -----------------------------
    spacy_names = {
        _clean_name(ent.text)
        for ent in doc.ents
        if ent.label_ == "PERSON" and _clean_name(ent.text)
    }

    # -----------------------------
    # 3. Intersection (STRICT)
    # -----------------------------
    def normalize(name):
        return name.lower().strip()

    intersection = [
        r for r in regex_names
        if any(normalize(r) == normalize(s) for s in spacy_names)
    ]

    # -----------------------------
    # 4. Return best match
    # -----------------------------
    return intersection[0] if intersection else None

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_COMPANY_EMAIL_RE = re.compile(
    r"support|info|care|help|claim|service|feedback|contact|"
    r"noreply|no-reply|admin|grievance|starhealth|niva|hdfc|icici|"
    r"bajaj|reliance|lic|sbi|max|kotak|tata|united|oriental|"
    r"national|newindia|policybazaar|coverfox|acko|digit|aditya",
    re.IGNORECASE,
)


def _extract_email(text: str) -> str | None:
    candidates = _EMAIL_RE.findall(text)
    for email in candidates:
        if not _COMPANY_EMAIL_RE.search(email):
            return email.lower()
    return candidates[0].lower() if candidates else None


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

def _extract_phone(text: str) -> str | None:
    # Priority 1: +91 prefix
    m = re.search(r"(?:\+91[\s\-]?)([6-9]\d{9})\b", text)
    if m:
        return m.group(1)
    # Priority 2: 10-digit mobile starting 6–9
    m = re.search(r"\b([6-9]\d{9})\b", text)
    if m:
        return m.group(1)
    # Priority 3: landline 0XXXXXXXXXX
    m = re.search(r"\b(0\d{9,10})\b", text)
    if m:
        return m.group(1)
    return None

# ---------------------------------------------------------------------------
# Insured members
# ---------------------------------------------------------------------------

_RELATIONSHIP_RE = re.compile(
    r"\b(Self|Spouse|Son|Daughter|Father|Mother|Brother|Sister|"
    r"Uncle|Aunt|Grandfather|Grandmother|Father-?in-?Law|Mother-?in-?Law|"
    r"Dependent|Child)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Nominee
# ---------------------------------------------------------------------------

_NOMINEE_SECTION_RE = re.compile(
    r"(?i)(?:Nominee\s+Details?|Nominee\s+Information|Beneficiary\s+Details?)"
)


def _extract_nominee(text: str) -> dict | None:
    m = _NOMINEE_SECTION_RE.search(text)
    if not m:
        return None

    section = text[m.start(): m.start() + 500]
    nominee: dict = {}

    name_m = re.search(r"(?i)Name\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{3,})", section)
    if name_m:
        cleaned = _clean_name(name_m.group(1))
        if cleaned:
            nominee["name"] = cleaned

    # Tabular fallback: row starting with digit, then name, then relationship
    if "name" not in nominee:
        tab_m = re.search(
            r"(?m)^\s*\d+\s+([A-Za-z][A-Za-z .'-]{3,})\s+"
            r"(?:Self|Spouse|Son|Daughter|Father|Mother|Brother|Sister|"
            r"Uncle|Aunt|Grandfather|Grandmother|Father-?in-?Law|Mother-?in-?Law|"
            r"Dependent|Child)\b",
            section,
            re.IGNORECASE,
        )
        if tab_m:
            cleaned = _clean_name(tab_m.group(1))
            if cleaned:
                nominee["name"] = cleaned

    rel_m = _RELATIONSHIP_RE.search(section)
    if rel_m:
        nominee["relationship"] = rel_m.group(1).capitalize()

    pct_m = re.search(r"(\d{1,3})\s*%", section)
    if pct_m:
        nominee["share_percentage"] = int(pct_m.group(1))

    return nominee if nominee else None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_policy_fields(text: str) -> dict:
    """
    Primary extraction using regex + spaCy NLP.
    Returns a dict with all policy fields; missing values are None or [].
    Never raises — all failures are silently caught and logged.
    """
    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            logger.debug("Extractor %s failed: %s", fn.__name__, exc)
            return None

    nlp = _get_nlp()
    doc = nlp(text)

    dates = _safe(_extract_dates, text, doc) or {}


    return {
        "policy_number": _safe(_extract_policy_number, text),
        "policy_type": _safe(_extract_policy_type, text, doc),
        "issue_date": dates.get("issue_date"),
        "renewal_date": dates.get("renewal_date"),
        "period": dates.get("period"),
        "premium": _safe(_extract_premium, text),
        "sum_insured": _safe(_extract_sum_insured, text),
        "limit_of_coverage": _safe(_extract_limit_of_coverage, text),
        "recharge_benefit": _safe(_extract_recharge_benefit, text),
        "payment_frequency": _safe(_extract_payment_frequency, text),
        "proposer_name": _safe(_extract_proposer_name, text, doc),
        "email": _safe(_extract_email, text),
        "phone": _safe(_extract_phone, text),
        "nominee": _safe(_extract_nominee, text),
    }
