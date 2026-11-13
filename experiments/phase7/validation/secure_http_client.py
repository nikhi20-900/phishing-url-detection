"""
Secure HTTP/TLS Retrieval Client for Phase 7 Milestone 1.2.1 (M1.2.1).

Enforces strict network security, SSRF/DNS-rebinding prevention, and observation integrity:
1. Pre-resolves and validates all DNS address records. Rejects mixed public/private records.
2. Directly connects the socket to the validated IP, preventing TOCTOU/DNS-rebinding.
3. Performs TLS handshake and verification on the exact socket used for data retrieval.
4. Detects and rejects cross-domain redirects BEFORE resolving or connecting to destination.
5. Limits redirect hops to max 3 and caps response size at 512 KB.
6. Guarantees zero synthetic fallback: all observations are authentic.
"""

import hashlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit

from src.evaluate_domain_split import extract_registered_domain

REPO_ROOT = Path(__file__).resolve().parents[3]

# Disallowed private and special-purpose IP networks (RFC 1918, 6598, 3927, loopback, etc.)
DISALLOWED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),
]

USER_AGENT = "PhishingURLResearchClient/1.2.1 (+https://github.com/nikhi20-900/phishing-url-detection; legitimate-corpus-validation)"
MAX_RESPONSE_BYTES = 524288  # 512 KB
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_REDIRECTS = 3


class SecurityViolationError(Exception):
    """Raised when an IP address or redirect destination violates security policy."""
    pass


class CrossDomainRedirectBlockedError(Exception):
    """Raised when a cross-domain redirect is detected and blocked before connection."""
    pass


def is_disallowed_ip(ip_str: str) -> bool:
    """Check whether an IP string belongs to any private, loopback, or non-routable range."""
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        return (
            any(ip in net for net in DISALLOWED_NETWORKS)
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return True  # Malformed IP is disallowed by default


def resolve_and_validate_hostname(hostname: str) -> Tuple[List[str], float]:
    """
    Safely resolve hostname via DNS and validate that ALL returned IP addresses are public.
    Rejects any hostname that resolves to mixed public/private records.
    Returns: (list_of_validated_ips, dns_lookup_time_ms)
    """
    if not hostname:
        raise ValueError("Empty hostname")

    clean_host = hostname.strip("[]")
    # Check if host is already a literal IP address
    try:
        ip = ipaddress.ip_address(clean_host)
        ip_str = str(ip)
        if is_disallowed_ip(ip_str):
            raise SecurityViolationError(f"Direct private/reserved IP literal: {ip_str}")
        return [ip_str], 0.0
    except ValueError:
        pass

    # Resolve via socket.getaddrinfo
    t0 = time.perf_counter()
    try:
        addrinfo = socket.getaddrinfo(clean_host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception as e:
        dns_ms = round((time.perf_counter() - t0) * 1000, 2)
        raise socket.gaierror(f"DNS resolution failed for {clean_host}: {e}")

    dns_ms = round((time.perf_counter() - t0) * 1000, 2)
    if not addrinfo:
        raise socket.gaierror(f"No address records returned for {clean_host}")

    resolved_ips: List[str] = []
    seen: Set[str] = set()
    for entry in addrinfo:
        ip_str = entry[4][0]
        if ip_str not in seen:
            seen.add(ip_str)
            resolved_ips.append(ip_str)

    # Validate that every single resolved IP address is public
    for ip_str in resolved_ips:
        if is_disallowed_ip(ip_str):
            raise SecurityViolationError(
                f"DNS rebinding / SSRF hazard: Host {clean_host} resolves to private/reserved IP {ip_str}"
            )

    return resolved_ips, dns_ms


class DirectIPHTTPConnection(http.client.HTTPConnection):
    """HTTP Connection connecting directly to pre-validated IP address."""
    def __init__(self, host: str, validated_ip: str, port: int = 80, timeout: float = 5.0):
        super().__init__(host, port=port, timeout=timeout)
        self.validated_ip = validated_ip

    def connect(self):
        # Connect directly to the validated IP to prevent DNS rebinding
        self.sock = socket.create_connection((self.validated_ip, self.port), self.timeout)


class DirectIPHTTPSConnection(http.client.HTTPSConnection):
    """
    HTTPS Connection connecting directly to pre-validated IP address,
    with TLS handshake and certificate verification on the exact retrieval socket.
    """
    def __init__(
        self,
        host: str,
        validated_ip: str,
        port: int = 443,
        timeout: float = 5.0,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        super().__init__(host, port=port, timeout=timeout, context=ssl_context)
        self.validated_ip = validated_ip
        self.tls_version: Optional[str] = None
        self.tls_cipher: Optional[str] = None
        self.tls_cert: Optional[Dict[str, Any]] = None
        self.tls_verified: bool = False

    def connect(self):
        # 1. Connect socket directly to validated IP (no DNS resolution here)
        raw_sock = socket.create_connection((self.validated_ip, self.port), self.timeout)

        # 2. Wrap socket with TLS specifying the SNI host for certificate validation
        ctx = self._context or ssl.create_default_context()
        self.sock = ctx.wrap_socket(raw_sock, server_hostname=self.host)

        # 3. Extract TLS attributes from the exact connected socket
        self.tls_version = self.sock.version()
        cipher_tuple = self.sock.cipher()
        self.tls_cipher = cipher_tuple[0] if cipher_tuple else None
        self.tls_cert = self.sock.getpeercert()
        self.tls_verified = True


def extract_html_title(html_bytes: bytes) -> Optional[str]:
    """Safely extract <title> text from initial HTML body bytes."""
    try:
        text = html_bytes.decode("utf-8", errors="ignore")
        match = re.search(r"<title[^>]*>([^<]+)</title>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def execute_secure_retrieval(
    candidate_id: str,
    target_url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = MAX_REDIRECTS,
    max_body_bytes: int = MAX_RESPONSE_BYTES,
) -> Dict[str, Any]:
    """
    Execute controlled HTTP/TLS retrieval with full safety guarantees and evidence gathering:
    - Connects directly to the validated IP
    - Extracts TLS verification on the exact socket used for data transfer
    - Blocks cross-domain redirects before connection
    """
    retrieval_start_time = datetime.now(timezone.utc).isoformat()
    current_url = target_url.strip()
    initial_reg_domain = extract_registered_domain(target_url)

    redirect_chain = [current_url]
    hop_count = 0
    final_status_code: Optional[int] = None
    final_body = b""
    content_type: Optional[str] = None
    observation_error: Optional[str] = None
    is_cross_domain = False
    connected_ip: Optional[str] = None
    dns_time_total_ms = 0.0
    tls_info: Dict[str, Any] = {
        "tls_verified": False,
        "tls_protocol": None,
        "tls_cipher": None,
        "tls_cert_subject_cn": None,
        "tls_cert_san_count": 0,
        "tls_error": None,
    }

    ssl_ctx = ssl.create_default_context()

    while hop_count <= max_redirects:
        parsed = urlsplit(current_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        if scheme not in ("http", "https"):
            observation_error = f"UNSUPPORTED_SCHEME: Scheme '{scheme}' is not http/https"
            break

        # 1. DNS Resolution & Multi-record Private IP validation
        try:
            validated_ips, dns_ms = resolve_and_validate_hostname(hostname)
            dns_time_total_ms += dns_ms
            connected_ip = validated_ips[0]
        except SecurityViolationError as e:
            observation_error = f"SECURITY_VIOLATION: {e}"
            return {
                "candidate_id": candidate_id,
                "target_url": target_url,
                "retrieval_timestamp_utc": retrieval_start_time,
                "resolved_ip": connected_ip,
                "connected_ip": None,
                "dns_time_ms": dns_time_total_ms,
                "http_status_code": None,
                "tls_verified": False,
                "tls_protocol": None,
                "tls_cipher": None,
                "tls_cert_subject_cn": None,
                "tls_cert_san_count": 0,
                "redirect_count": hop_count,
                "redirect_chain": redirect_chain,
                "final_url": current_url,
                "is_cross_domain_redirect": False,
                "content_type": None,
                "content_length_bytes": 0,
                "response_sha256": None,
                "html_title": None,
                "is_accessible": False,
                "is_private_ip": True,
                "observation_error": observation_error,
                "is_synthetic": False,
            }
        except socket.gaierror as e:
            observation_error = f"DNS_FAILURE: {e}"
            break
        except Exception as e:
            observation_error = f"DNS_ERROR: {e}"
            break

        # 2. Open Direct Connection to the validated IP
        conn = None
        try:
            if scheme == "https":
                conn = DirectIPHTTPSConnection(
                    host=hostname,
                    validated_ip=connected_ip,
                    port=port,
                    timeout=timeout,
                    ssl_context=ssl_ctx,
                )
            else:
                conn = DirectIPHTTPConnection(
                    host=hostname,
                    validated_ip=connected_ip,
                    port=port,
                    timeout=timeout,
                )

            # Establish socket connection and TLS handshake directly
            conn.connect()

            # If HTTPS, extract TLS parameters from the retrieval connection
            if scheme == "https" and hasattr(conn, "tls_verified"):
                tls_info["tls_verified"] = conn.tls_verified
                tls_info["tls_protocol"] = conn.tls_version
                tls_info["tls_cipher"] = conn.tls_cipher

                cert = getattr(conn, "tls_cert", None)
                if cert and "subject" in cert:
                    for sub in cert["subject"]:
                        for k, v in sub:
                            if k == "commonName":
                                tls_info["tls_cert_subject_cn"] = v
                if cert and "subjectAltName" in cert:
                    san_list = [v for k, v in cert["subjectAltName"] if k == "DNS"]
                    tls_info["tls_cert_san_count"] = len(san_list)

            # Send HTTP GET request
            req_headers = {
                "User-Agent": USER_AGENT,
                "Host": hostname,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "close",
            }
            conn.request("GET", path, headers=req_headers)
            resp = conn.getresponse()
            final_status_code = resp.status
            content_type = resp.getheader("Content-Type")

            # Check for HTTP redirect response
            if final_status_code in (301, 302, 303, 307, 308):
                location = resp.getheader("Location")
                if location:
                    next_url = urljoin(current_url, location)
                    next_reg_domain = extract_registered_domain(next_url)

                    # CRITICAL: Detect cross-domain redirect and REJECT BEFORE following
                    if next_reg_domain != initial_reg_domain:
                        is_cross_domain = True
                        observation_error = (
                            f"CROSS_DOMAIN_REDIRECT_BLOCKED: Redirect from {initial_reg_domain} "
                            f"to {next_reg_domain} ({next_url}) blocked before connection"
                        )
                        # Do not connect to next_url! Terminate here.
                        break

                    # Same-domain redirect: advance hop
                    hop_count += 1
                    redirect_chain.append(next_url)
                    current_url = next_url
                    conn.close()

                    if hop_count > max_redirects:
                        observation_error = f"REDIRECT_LIMIT_EXCEEDED: Exceeded {max_redirects} redirects"
                        break
                    continue

            # Read response body up to max_body_bytes
            body_chunks = []
            total_bytes = 0
            while total_bytes < max_body_bytes:
                chunk = resp.read(min(8192, max_body_bytes - total_bytes))
                if not chunk:
                    break
                body_chunks.append(chunk)
                total_bytes += len(chunk)

            final_body = b"".join(body_chunks)
            conn.close()
            break

        except ssl.SSLCertVerificationError as e:
            tls_info["tls_verified"] = False
            tls_info["tls_error"] = f"SSLCertVerificationError: {e}"
            observation_error = f"TLS_VERIFICATION_FAILED: {e}"
            break
        except ssl.SSLError as e:
            tls_info["tls_verified"] = False
            tls_info["tls_error"] = f"SSLError: {e}"
            observation_error = f"TLS_ERROR: {e}"
            break
        except socket.timeout:
            observation_error = f"TIMEOUT: Socket timed out after {timeout}s"
            break
        except Exception as e:
            observation_error = f"CONNECTION_ERROR: {type(e).__name__}: {e}"
            break
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # Calculate real SHA-256 of response body
    response_sha = hashlib.sha256(final_body).hexdigest() if final_body else None
    html_title = extract_html_title(final_body)

    # Accessibility criteria:
    # 1. Status in [200..399]
    # 2. No cross-domain redirect
    # 3. For HTTPS: TLS verification succeeded on the retrieval socket
    is_https = urlsplit(target_url).scheme.lower() == "https"
    is_accessible = (
        final_status_code is not None
        and (200 <= final_status_code < 400)
        and not is_cross_domain
        and (not is_https or bool(tls_info["tls_verified"]))
    )

    return {
        "candidate_id": candidate_id,
        "target_url": target_url,
        "retrieval_timestamp_utc": retrieval_start_time,
        "resolved_ip": connected_ip,
        "connected_ip": connected_ip,
        "dns_time_ms": dns_time_total_ms,
        "http_status_code": final_status_code,
        "tls_verified": bool(tls_info.get("tls_verified", False)),
        "tls_protocol": tls_info.get("tls_protocol"),
        "tls_cipher": tls_info.get("tls_cipher"),
        "tls_cert_subject_cn": tls_info.get("tls_cert_subject_cn"),
        "tls_cert_san_count": tls_info.get("tls_cert_san_count", 0),
        "redirect_count": hop_count,
        "redirect_chain": redirect_chain,
        "final_url": current_url,
        "is_cross_domain_redirect": is_cross_domain,
        "content_type": content_type,
        "content_length_bytes": len(final_body),
        "response_sha256": response_sha,
        "html_title": html_title,
        "is_accessible": is_accessible,
        "is_private_ip": False,
        "observation_error": observation_error,
        "is_synthetic": False,
    }

