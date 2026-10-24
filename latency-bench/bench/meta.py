"""Machine and network metadata, plus TCP-connect / TLS-handshake probes."""

from __future__ import annotations

import os
import platform
import socket
import ssl
import statistics
import subprocess
import sys
from typing import Any

from bench.clock import now_ns


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def network_interface_type() -> str:
    """Best effort: 'wifi', 'wired', or 'unknown'."""
    try:
        if sys.platform == "win32":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "Get-NetAdapter | Where-Object Status -eq 'Up' | "
                        "Sort-Object -Property ifIndex | Select-Object -ExpandProperty PhysicalMediaType"])
            types = [t.strip().lower() for t in out.splitlines() if t.strip()]
            if any("802.11" in t or "wireless" in t for t in types):
                return "wifi" if not any("802.3" in t for t in types) else "wired+wifi"
            if any("802.3" in t for t in types):
                return "wired"
        elif sys.platform == "darwin":
            route = _run(["route", "-n", "get", "default"])
            iface = next((ln.split(":")[1].strip() for ln in route.splitlines() if "interface:" in ln), "")
            ports = _run(["networksetup", "-listallhardwareports"])
            blocks = ports.split("\n\n")
            for b in blocks:
                if f"Device: {iface}" in b:
                    return "wifi" if "Wi-Fi" in b else "wired"
        else:
            route = _run(["ip", "route", "show", "default"]).split()
            iface = route[route.index("dev") + 1] if "dev" in route else ""
            if iface:
                return "wifi" if os.path.exists(f"/sys/class/net/{iface}/wireless") else "wired"
    except Exception:
        pass
    return "unknown"


def power_and_vpn() -> dict[str, Any]:
    """Power source / plan and active VPN-like adapters (both affect latency). Best effort."""
    info: dict[str, Any] = {}
    try:
        if sys.platform == "win32":
            ps = ("$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; "
                  "if ($b) { 'battery_status=' + $b.BatteryStatus } else { 'battery_status=none' }; "
                  "Get-NetAdapter | Where-Object Status -eq 'Up' | ForEach-Object { 'adapter=' + $_.InterfaceDescription }")
            out = _run(["powershell", "-NoProfile", "-Command", ps])
            adapters = [ln.split("=", 1)[1] for ln in out.splitlines() if ln.startswith("adapter=")]
            bs = next((ln.split("=", 1)[1] for ln in out.splitlines() if ln.startswith("battery_status=")), "none")
            info["power_source"] = {"2": "ac", "1": "battery"}.get(bs.strip(), "ac" if bs == "none" else f"status {bs}")
            scheme = _run(["powercfg", "/getactivescheme"])
            info["power_plan"] = scheme.split("(")[-1].strip().rstrip(")").strip() if "(" in scheme else "unknown"
        elif sys.platform == "darwin":
            out = _run(["pmset", "-g", "batt"])
            info["power_source"] = "ac" if "AC Power" in out else ("battery" if "Battery Power" in out else "unknown")
            adapters = [ln for ln in _run(["ifconfig", "-l"]).split() if ln.startswith(("utun", "ppp", "ipsec"))]
        else:
            adapters = os.listdir("/sys/class/net") if os.path.isdir("/sys/class/net") else []
        vpn_words = ("vpn", "wireguard", "tap-", "tun", "tunnel", "proton", "nord", "tailscale", "zerotier", "openvpn")
        info["vpn_adapters"] = [a for a in adapters if any(w in a.lower() for w in vpn_words)]
    except Exception:
        pass
    return info


def machine_metadata(location: str = "") -> dict[str, Any]:
    return {
        **power_and_vpn(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "network_interface": network_interface_type(),
        "location": location or "unspecified",
    }


def probe_host(host: str, port: int = 443, samples: int = 5, timeout: float = 5.0) -> dict[str, Any]:
    """Median TCP connect and TLS handshake time to host:port."""
    tcp_ms: list[float] = []
    tls_ms: list[float] = []
    errors: list[str] = []
    ctx = ssl.create_default_context()
    for _ in range(samples):
        try:
            addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]  # DNS outside timing
            t0 = now_ns()
            sock = socket.create_connection(addr[:2], timeout=timeout)
            t1 = now_ns()
            try:
                tls = ctx.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
                tls.settimeout(timeout)
                t2 = now_ns()
                tls.do_handshake()
                t3 = now_ns()
                tls.close()
            finally:
                sock.close()
            tcp_ms.append((t1 - t0) / 1e6)
            tls_ms.append((t3 - t2) / 1e6)
        except Exception as e:  # recorded, never fatal
            errors.append(f"{type(e).__name__}: {e}")
    med = (lambda v: round(statistics.median(v), 2) if v else None)
    return {
        "host": host,
        "tcp_connect_ms_median": med(tcp_ms),
        "tls_handshake_ms_median": med(tls_ms),
        "tcp_connect_ms": [round(v, 2) for v in tcp_ms],
        "tls_handshake_ms": [round(v, 2) for v in tls_ms],
        "errors": errors,
    }
