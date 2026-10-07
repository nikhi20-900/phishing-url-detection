"""
Feature extraction module for Phishing URL Detection.
Extracts the exact 18 lexical and structural features aligned with the
PhiUSIIL dataset and the trained Random Forest classifier.
"""

import re
import ipaddress
from urllib.parse import urlparse
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

def is_ip_address(host: str) -> int:
    """Check if host is an IPv4 or IPv6 address."""
    clean_host = host.split(":")[0].strip("[]")
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
    u = url.strip()
    if not u:
        raise ValueError("URL cannot be empty")

    # Add protocol if absent for URL parsing
    if not (u.startswith("http://") or u.startswith("https://")):
        u_full = "http://" + u
    else:
        u_full = u

    is_https = 1 if u_full.lower().startswith("https://") else 0
    parsed = urlparse(u_full)
    hostname = parsed.netloc.split(":")[0].lower()

    # 1. URLLength: normalized protocol length in PhiUSIIL
    url_len = max(1, len(u_full) - (1 if is_https else 0))

    # 2. DomainLength: hostname length
    domain_len = len(hostname)

    # 3. IsDomainIP: direct IP check
    is_domain_ip = is_ip_address(hostname)

    # 4. TLDLength & 5. NoOfSubDomain
    parts = hostname.split(".")
    if is_domain_ip or len(parts) < 2:
        tld_len = 0
        no_of_subdomain = 0
    else:
        tld = parts[-1]
        tld_len = len(tld)
        # In PhiUSIIL: www.example.com has 1 subdomain (parts=3 -> 1)
        no_of_subdomain = max(0, len(parts) - 2)

    # Obfuscation: %xx
    obf_matches = re.findall(r"%[0-9a-fA-F]{2}", u_full)
    has_obfuscation = 1 if len(obf_matches) > 0 else 0
    no_of_obfuscated_char = len(obf_matches) * 3
    obfuscation_ratio = round(no_of_obfuscated_char / url_len, 4)

    # Cleaned body for character analysis
    s = re.sub(r"^https?://", "", u_full)
    s_no_www = re.sub(r"^www\.", "", s).rstrip("/")

    # 9. NoOfLettersInURL & 10. LetterRatioInURL
    letters = max(0, len(re.findall(r"[a-zA-Z]", s_no_www)) - 1)
    letter_ratio = round(letters / url_len, 4)

    # 11. NoOfDegitsInURL & 12. DegitRatioInURL
    digits = len(re.findall(r"[0-9]", u_full))
    digit_ratio = round(digits / url_len, 4)

    # 13. NoOfEqualsInURL, 14. NoOfQMarkInURL, 15. NoOfAmpersandInURL
    equals = u_full.count("=")
    qmark = u_full.count("?")
    ampersand = u_full.count("&")

    # 16. NoOfOtherSpecialCharsInURL & 17. SpacialCharRatioInURL
    other_special = len(re.findall(r"[^a-zA-Z0-9=?&%]", s_no_www))
    special_ratio = round(other_special / url_len, 4)

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
