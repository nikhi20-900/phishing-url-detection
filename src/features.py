"""
Feature extraction module for Phishing URL Detection.
Extracts the exact 18 lexical and structural features aligned with the
PhiUSIIL dataset and the trained Random Forest classifier.
"""

import re
from typing import NamedTuple, Optional
import ipaddress
from urllib.parse import urlsplit
import pandas as pd

FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

FEATURE_DESCRIPTIONS = {
    "URLLength": "Total character count of the URL string (normalized protocol length)",
    "DomainLength": "Total length of the domain/hostname component",
    "IsDomainIP": "Whether the domain host is a direct IP address (1 = Yes, 0 = No)",
    "TLDLength": "Length of the Top-Level Domain suffix (e.g., com=3, org=3, uk=2)",
    "NoOfSubDomain": "Number of subdomain levels in the domain",
    "HasObfuscation": "Presence of hex/percent encoding in the URL (1 = Yes, 0 = No)",
    "NoOfObfuscatedChar": "Total count of characters within hex encodings (%xx)",
    "ObfuscationRatio": "Ratio of obfuscated characters to URL length",
    "NoOfLettersInURL": "Total count of alphabetic letters in the URL body",
    "LetterRatioInURL": "Ratio of alphabetic letters to total URL length",
    "NoOfDegitsInURL": "Total count of numeric digits [0-9] in the URL",
    "DegitRatioInURL": "Ratio of numeric digits to total URL length",
    "NoOfEqualsInURL": "Count of '=' characters (typical in query parameters)",
    "NoOfQMarkInURL": "Count of '?' characters (query parameter delimiter)",
    "NoOfAmpersandInURL": "Count of '&' characters (parameter separators)",
    "NoOfOtherSpecialCharsInURL": "Count of special characters excluding '=', '?', '&', '%'",
    "SpacialCharRatioInURL": "Ratio of special characters to total URL length",
    "IsHTTPS": "Whether HTTPS protocol is active (1 = Secure HTTPS, 0 = Insecure HTTP)"
}

FEATURE_CATEGORIES = {
    "URLLength": "Structural Length",
    "DomainLength": "Domain Structure",
    "IsDomainIP": "Domain Security",
    "TLDLength": "Domain Structure",
    "NoOfSubDomain": "Domain Structure",
    "HasObfuscation": "Obfuscation",
    "NoOfObfuscatedChar": "Obfuscation",
    "ObfuscationRatio": "Obfuscation",
    "NoOfLettersInURL": "Character Composition",
    "LetterRatioInURL": "Character Composition",
    "NoOfDegitsInURL": "Character Composition",
    "DegitRatioInURL": "Character Composition",
    "NoOfEqualsInURL": "Query & Parameters",
    "NoOfQMarkInURL": "Query & Parameters",
    "NoOfAmpersandInURL": "Query & Parameters",
    "NoOfOtherSpecialCharsInURL": "Special Characters",
    "SpacialCharRatioInURL": "Special Characters",
    "IsHTTPS": "Transport Security"
}

class ParsedURL(NamedTuple):
    raw_url: str
    normalized_url: str
    scheme: str
    is_https: int
    hostname: str
    raw_hostname: str
    port: Optional[int]
    is_ip: int
    ip_version: Optional[int]
    has_credentials: int
    username: Optional[str]
    password: Optional[str]
    path: str
    query: str
    fragment: str

def parse_url(url: str) -> ParsedURL:
    """
    Safely validate and parse a URL into structured, normalized components.

    Handles HTTP/HTTPS schemes, IPv4/IPv6, ports, credentials, IDN/punycode,
    and malformed inputs.

    Raises:
        ValueError: If input is empty, not a string, contains an unsupported
                    protocol, invalid port, malformed hostname, or malformed IP.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    raw = url.strip()
    if not raw:
        raise ValueError("URL cannot be empty")

    # Early rejection of unsupported schemes (e.g. javascript:alert(1), ftp://, data:, mailto:)
    if "://" in raw:
        scheme_candidate = raw.split("://", 1)[0].lower()
        if scheme_candidate not in {"http", "https"}:
            raise ValueError(f"Unsupported scheme '{scheme_candidate}'. Only HTTP and HTTPS URLs are supported.")
        normalized = raw
    elif raw.startswith("//"):
        normalized = "http:" + raw
    else:
        # Check for colon-separated scheme like javascript:alert(1), mailto:, etc.
        # Allow domain:port (e.g. example.com:8080 or localhost:8080)
        colon_pos = raw.find(":")
        slash_pos = raw.find("/")
        if colon_pos != -1 and (slash_pos == -1 or colon_pos < slash_pos):
            prefix = raw[:colon_pos].lower()
            suffix = raw[colon_pos + 1:]
            first_suffix_segment = suffix.split("/")[0]
            if not ("." in prefix or first_suffix_segment.isdigit()):
                if prefix not in {"http", "https"}:
                    raise ValueError(f"Unsupported scheme '{prefix}'. Only HTTP and HTTPS URLs are supported.")
                normalized = "http://" + suffix if prefix == "http" else "https://" + suffix
            else:
                normalized = "http://" + raw
        else:
            normalized = "http://" + raw

    try:
        parsed = urlsplit(normalized)
    except Exception as e:
        raise ValueError(f"Malformed URL: {e}")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported scheme '{scheme}'. Only HTTP and HTTPS URLs are supported.")

    # Validate port syntax and range (0-65535)
    try:
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"Invalid URL port: {e}")

    if not parsed.hostname:
        raise ValueError("URL must contain a valid hostname")

    raw_host = parsed.hostname.lower().rstrip(".")
    if not raw_host:
        raise ValueError("URL must contain a non-empty hostname")

    # Internationalized Domain Names (IDN) normalization to Punycode.
    # Note: IDNA normalization converts Unicode domain names to ASCII Punycode
    # for standard DNS resolution and character handling; it does NOT by itself
    # prevent homograph attacks or determine whether a domain is malicious.
    try:
        ascii_host = raw_host.encode("idna").decode("ascii")
    except UnicodeError as e:
        raise ValueError(f"Invalid domain encoding / IDN '{raw_host}': {e}")

    # Check for attempted / malformed IPv4 address (e.g. 256.256.256.256, 192.168.1.999)
    # Explicitly test before treating as standard domain
    is_numeric_ip_pattern = bool(re.match(r"^\d+(\.\d+)+$", ascii_host))

    is_ip = 0
    ip_version = None
    try:
        ip_obj = ipaddress.ip_address(ascii_host)
        is_ip = 1
        ip_version = ip_obj.version
    except ValueError:
        if is_numeric_ip_pattern:
            raise ValueError(f"Malformed IPv4 address: '{ascii_host}' is not a valid IP address")
        is_ip = 0

    # Domain syntax validation for non-IP hosts
    if not is_ip:
        if len(ascii_host) > 253:
            raise ValueError(f"Domain name exceeds maximum RFC length of 253 characters ({len(ascii_host)} chars)")

        labels = ascii_host.split(".")
        label_regex = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
        for label in labels:
            if not label:
                raise ValueError("Domain cannot contain empty labels (e.g., consecutive dots)")
            if not label_regex.match(label):
                raise ValueError(f"Invalid domain label '{label}': labels must be 1-63 alphanumeric characters and cannot start or end with a hyphen")

        # Top-level domain validation: TLD cannot be all-numeric
        if len(labels) >= 2 and labels[-1].isdigit():
            raise ValueError(f"Invalid top-level domain '{labels[-1]}': TLDs cannot be all-numeric")

    is_https = 1 if scheme == "https" else 0
    # Credentials detection: verify whether userinfo was specified in authority component
    has_creds = 1 if "@" in parsed.netloc else 0

    return ParsedURL(
        raw_url=url,
        normalized_url=normalized,
        scheme=scheme,
        is_https=is_https,
        hostname=ascii_host,
        raw_hostname=raw_host,
        port=port,
        is_ip=is_ip,
        ip_version=ip_version,
        has_credentials=has_creds,
        username=parsed.username,
        password=parsed.password,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment
    )

def is_ip_address(host: str) -> int:
    """Check if host is an IPv4 or IPv6 address."""
    clean_host = host.strip("[]")
    if ":" in clean_host and clean_host.count(":") == 1:
        clean_host = clean_host.split(":")[0]
    try:
        ipaddress.ip_address(clean_host)
        return 1
    except ValueError:
        return 0

def extract_url_features(url: str) -> dict:
    """
    Extract the 18 lexical and structural URL features aligned with PhiUSIIL.

    Args:
        url (str): Raw URL string entered by the user.

    Returns:
        dict: Dictionary containing the 18 extracted features.
    """
    parsed_info = parse_url(url)
    u_full = parsed_info.normalized_url
    is_https = parsed_info.is_https
    hostname = parsed_info.hostname
    is_domain_ip = parsed_info.is_ip

    # 1. URLLength: actual full input URL length (normalized)
    url_len = max(1, len(u_full))

    # 2. DomainLength: hostname length
    domain_len = len(hostname)

    # 4. TLDLength & 5. NoOfSubDomain
    # Aligned with PhiUSIIL convention: dot-split applied consistently to both domains and IPs
    parts = hostname.split(".")
    if len(parts) < 2:
        tld_len = 0
        no_of_subdomain = 0
    else:
        tld = parts[-1]
        tld_len = len(tld)
        no_of_subdomain = max(0, len(parts) - 2)

    # Obfuscation: %xx (hexadecimal escape pattern)
    obf_matches = re.findall(r"%[0-9a-fA-F]{2}", u_full)
    has_obfuscation = 1 if len(obf_matches) > 0 else 0
    no_of_obfuscated_char = len(obf_matches) * 3
    obfuscation_ratio = round(no_of_obfuscated_char / url_len, 3)

    # Cleaned body for character analysis (without trailing slash stripping)
    s = re.sub(r"^https?://", "", u_full)
    s_no_www = re.sub(r"^www\.", "", s)

    # 9. NoOfLettersInURL & 10. LetterRatioInURL (rounded to 3 decimal places)
    letters = len(re.findall(r"[a-zA-Z]", s_no_www))
    letter_ratio = round(letters / url_len, 3)

    # 11. NoOfDegitsInURL & 12. DegitRatioInURL (rounded to 3 decimal places)
    digits = len(re.findall(r"[0-9]", u_full))
    digit_ratio = round(digits / url_len, 3)

    # 13. NoOfEqualsInURL, 14. NoOfQMarkInURL, 15. NoOfAmpersandInURL
    equals = u_full.count("=")
    qmark = u_full.count("?")
    ampersand = u_full.count("&")

    # 16. NoOfOtherSpecialCharsInURL & 17. SpacialCharRatioInURL
    # SpacialCharRatioInURL includes all special chars (other + equals + qmark + ampersand) / url_len
    other_special = len(re.findall(r"[^a-zA-Z0-9=?&%]", s_no_www))
    total_special = other_special + equals + qmark + ampersand
    special_ratio = round(total_special / url_len, 3)

    # 18. IsHTTPS: true active TLS scheme (https://).
    # Note: In historical PhiUSIIL CSV, substring check ('https' in url) was used,
    # causing 493 HTTP URLs containing 'https' in query parameters to be labeled 1.
    # For transport security integrity, we evaluate the actual verified protocol.
    return {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_len,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": equals,
        "NoOfQMarkInURL": qmark,
        "NoOfAmpersandInURL": ampersand,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": is_https
    }

def extract_features_dataframe(url: str) -> pd.DataFrame:
    """Extract features as single-row DataFrame aligned with trained model features."""
    features = extract_url_features(url)
    return pd.DataFrame([features], columns=FEATURE_NAMES)
