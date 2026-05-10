"""
EarlyFlow — PCAP to Labeled Parquet Pipeline
=============================================
Pass 1: Read every packet for DNS content (no port filter)
        Stores CNAME answer name (not qname) for correct suffix matching
Pass 2: Extract netFound features + label flows
        Persistent flow_label_cache across all files
"""

import dpkt
import os
import socket
import glob
import struct
import time
from collections import defaultdict
import pandas as pd

from domain_map import domain_to_labels, DOMAIN_MAP

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
PCAP_DIR      = "/home/sai/Downloads/early_ml_final"
OUTPUT_DIR    = "/home/sai/Downloads/early_ml_final"
BATCH_SIZE    = 1_000_000
SKIP_EXISTING = True


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────
def _open_pcap(fh):
    try:
        return dpkt.pcap.Reader(fh)
    except ValueError:
        fh.seek(0)
        return dpkt.pcapng.Reader(fh)


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    a, b = (src_ip, src_port), (dst_ip, dst_port)
    if a > b:
        a, b = b, a
    return f"{a[0]}_{a[1]}_{b[0]}_{b[1]}_{proto}"


def _try_parse_dns(data: bytes):
    if len(data) < 12:
        return None
    try:
        dns = dpkt.dns.DNS(data)
        if len(dns.qd) == 0:
            return None
        name = dns.qd[0].name
        if not name or len(name) < 3 or "." not in name:
            return None
        return dns
    except Exception:
        return None


def _try_parse_dns_tcp(data: bytes):
    if len(data) < 14:
        return None
    try:
        dns_len = struct.unpack("!H", data[:2])[0]
        if dns_len < 12 or dns_len > len(data) - 2:
            return None
        return _try_parse_dns(data[2:2 + dns_len])
    except Exception:
        return None


def _fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _progress(fi, total_files, total_pkts, labeled, flow_cache_size,
               file_start, pass_start, label_counts):
    elapsed_file  = time.time() - file_start
    elapsed_total = time.time() - pass_start
    pct_files     = (fi + 1) / total_files * 100
    pct_labeled   = labeled / total_pkts * 100 if total_pkts else 0

    # ETA based on files remaining
    if fi > 0:
        rate = (fi + 1) / elapsed_total   # files/sec
        remaining = (total_files - fi - 1) / rate
        eta_str = _fmt_time(remaining)
    else:
        eta_str = "calculating..."

    print(f"\n  ┌─ Progress ──────────────────────────────────────")
    print(f"  │  Files:    {fi+1}/{total_files} ({pct_files:.1f}%)")
    print(f"  │  Packets:  {total_pkts:,} this file | {labeled:,} labeled ({pct_labeled:.1f}%)")
    print(f"  │  Flows:    {flow_cache_size:,} labeled flows cached")
    print(f"  │  Elapsed:  {_fmt_time(elapsed_total)} total | {_fmt_time(elapsed_file)} this file")
    print(f"  │  ETA:      {eta_str}")
    if label_counts:
        print(f"  │  Labels:   ", end="")
        parts = [f"{k}={v:,}" for k, v in sorted(label_counts.items())]
        print(" | ".join(parts))
    print(f"  └────────────────────────────────────────────────", flush=True)


# ─────────────────────────────────────────────────────────────────
# PASS 1 — exhaustive DNS extraction (no port filter)
# KEY FIX: stores answer 'name' not 'qname' so suffix matching works
# e.g. stores "rr3.sn-npoe7n.googlevideo.com" not "youtube.com"
# ─────────────────────────────────────────────────────────────────
def run_pass1(pcap_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ip_domain_map  = {}   # ip → (domain, expire_ts)
    cname_map      = {}   # alias → canonical
    domain_freq    = defaultdict(int)

    total_files    = len(pcap_paths)
    total_packets  = 0
    total_dns      = 0
    pass_start     = time.time()

    print(f"\n{'='*60}")
    print(f"PASS 1 — DNS extraction from {total_files} PCAP file(s)")
    print(f"Strategy: read EVERY packet, no port filter")
    print(f"Fix: store answer name (not qname) for correct label matching")
    print('='*60)

    for fi, pcap_path in enumerate(pcap_paths):
        fname      = os.path.basename(pcap_path)
        fsize_gb   = os.path.getsize(pcap_path) / 1024**3
        file_start = time.time()
        file_pkts  = 0
        file_dns   = 0

        print(f"\n[{fi+1}/{total_files}] {fname} ({fsize_gb:.1f} GB)")

        with open(pcap_path, "rb") as fh:
            try:
                pcap = _open_pcap(fh)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            for ts, buf in pcap:
                total_packets += 1
                file_pkts     += 1

                # Progress every 2M packets
                if file_pkts % 2_000_000 == 0:
                    elapsed = time.time() - file_start
                    rate    = file_pkts / elapsed / 1e6
                    print(f"  {file_pkts/1e6:.0f}M pkts | "
                          f"{file_dns:,} DNS | "
                          f"{len(ip_domain_map):,} IPs mapped | "
                          f"{rate:.1f}M pkt/s | "
                          f"elapsed {_fmt_time(elapsed)}", flush=True)

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    src_ip = socket.inet_ntoa(ip.src)
                except Exception:
                    continue

                dns = None
                if isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    if len(udp.data) >= 12:
                        dns = _try_parse_dns(bytes(udp.data))
                elif isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    if len(tcp.data) >= 12:
                        dns = _try_parse_dns(bytes(tcp.data))
                        if dns is None:
                            dns = _try_parse_dns_tcp(bytes(tcp.data))

                if dns is None:
                    continue

                file_dns   += 1
                total_dns  += 1

                try:
                    qname = dns.qd[0].name.lower().rstrip(".")
                except Exception:
                    continue

                domain_freq[qname] += 1

                if dns.qr == 1:
                    for ans in dns.an:
                        try:
                            ans_name = ans.name.lower().rstrip(".")

                            if ans.type == dpkt.dns.DNS_CNAME:
                                target = ans.cname.lower().rstrip(".")
                                cname_map[ans_name] = target

                            elif ans.type == dpkt.dns.DNS_A:
                                resolved_ip = socket.inet_ntoa(ans.rdata)
                                expire      = ts + ans.ttl

                                # ── KEY FIX ──────────────────────────────────
                                # Store ans_name (e.g. "rr3.googlevideo.com")
                                # NOT qname (e.g. "youtube.com" which is excluded)
                                # This makes suffix matching work correctly
                                # ─────────────────────────────────────────────
                                label_domain = ans_name

                                # If ans_name has no label, try qname as fallback
                                if domain_to_labels(ans_name) is None:
                                    if domain_to_labels(qname) is not None:
                                        label_domain = qname

                                if (resolved_ip not in ip_domain_map or
                                        ip_domain_map[resolved_ip][1] < expire):
                                    ip_domain_map[resolved_ip] = (label_domain, expire)

                        except Exception:
                            continue

        file_elapsed = time.time() - file_start
        print(f"  Done: {file_pkts/1e6:.1f}M packets | "
              f"{file_dns:,} DNS packets | "
              f"{_fmt_time(file_elapsed)}")

    # Pre-compute ip → (provider, category)
    ip_label_cache = {}
    for ip_addr, (domain, _expire) in ip_domain_map.items():
        result = domain_to_labels(domain)
        if result:
            ip_label_cache[ip_addr] = result

    # Summary
    total_elapsed = time.time() - pass_start
    print(f"\n{'='*60}")
    print(f"PASS 1 COMPLETE in {_fmt_time(total_elapsed)}")
    print(f"  Packets scanned:   {total_packets:,}")
    print(f"  DNS packets found: {total_dns:,}")
    print(f"  Unique IPs mapped: {len(ip_domain_map):,}")
    print(f"  IPs with label:    {len(ip_label_cache):,}")

    cat_counts = defaultdict(int)
    for ip, (domain, _) in ip_domain_map.items():
        result = domain_to_labels(domain)
        if result:
            cat_counts[result[1]] += 1

    print(f"\n  IPs per class:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:15s}: {cnt:,}")

    # Save
    pd.DataFrame([
        {"ip": ip, "domain": d, "expire_ts": e}
        for ip, (d, e) in ip_domain_map.items()
    ]).to_parquet(os.path.join(output_dir, "global_dns_map.parquet"), index=False)

    pd.DataFrame(
        list(domain_freq.items()), columns=["dns.qry.name", "query_count"]
    ).sort_values("query_count", ascending=False).to_csv(
        os.path.join(output_dir, "global_dns_inventory.csv"), index=False
    )

    print(f"\n  Saved: global_dns_map.parquet")
    print(f"  Saved: global_dns_inventory.csv")

    return ip_label_cache


# ─────────────────────────────────────────────────────────────────
# PASS 2 — feature extraction + labeling
# ─────────────────────────────────────────────────────────────────
def run_pass2(pcap_paths, ip_label_cache, output_dir,
              batch_size=BATCH_SIZE, skip_existing=SKIP_EXISTING):

    os.makedirs(output_dir, exist_ok=True)

    # Load full DNS map for TTL fallback
    dns_map_path = os.path.join(output_dir, "global_dns_map.parquet")
    if os.path.exists(dns_map_path):
        dns_df     = pd.read_parquet(dns_map_path)
        global_dns = {
            row["ip"]: (row["domain"], row["expire_ts"])
            for _, row in dns_df.iterrows()
        }
    else:
        global_dns = {}

    # Persistent across ALL files
    flow_label_cache = {}
    flow_pkt_count   = defaultdict(int)
    flow_last_ts     = {}

    total_files   = len(pcap_paths)
    grand_total   = 0
    grand_labeled = 0
    pass_start    = time.time()

    print(f"\n{'='*60}")
    print(f"PASS 2 — Feature extraction from {total_files} file(s)")
    print(f"  IP label cache: {len(ip_label_cache):,} labeled IPs")
    print(f"  DNS TTL map:    {len(global_dns):,} IPs")
    print('='*60)

    for fi, pcap_path in enumerate(pcap_paths):
        stem       = os.path.splitext(os.path.basename(pcap_path))[0]
        fsize_gb   = os.path.getsize(pcap_path) / 1024**3
        file_start = time.time()

        print(f"\n[{fi+1}/{total_files}] {os.path.basename(pcap_path)} ({fsize_gb:.1f} GB)")

        if skip_existing:
            existing = glob.glob(os.path.join(output_dir, f"{stem}_batch_*.parquet"))
            if existing:
                print(f"  SKIP — already has {len(existing)} batch file(s)")
                continue

        rows          = []
        batch_num     = 0
        total         = 0
        labeled       = 0
        label_counts  = defaultdict(int)

        def _flush(final=False):
            nonlocal batch_num, rows
            if not rows:
                return
            tag = "final" if final else str(batch_num)
            out = os.path.join(output_dir, f"{stem}_batch_{tag}.parquet")
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"  → saved {os.path.basename(out)} ({len(rows):,} rows)")
            rows = []
            if not final:
                batch_num += 1

        with open(pcap_path, "rb") as fh:
            try:
                pcap = _open_pcap(fh)
            except Exception as e:
                print(f"  ERROR: {e} — skipping")
                continue

            for ts, buf in pcap:
                total += 1

                # Progress every 500k packets
                if total % 500_000 == 0:
                    pct    = labeled / total * 100 if total else 0
                    elapsed = time.time() - file_start
                    rate   = total / elapsed / 1e6
                    # ETA for this file based on file size vs bytes read
                    print(f"  {total/1e6:.1f}M pkts | "
                          f"{labeled:,} labeled ({pct:.1f}%) | "
                          f"{len(flow_label_cache):,} flows | "
                          f"{rate:.1f}M pkt/s | "
                          f"{_fmt_time(elapsed)} elapsed", flush=True)
                    if label_counts:
                        parts = [f"{k[:5]}={v:,}" for k, v in
                                 sorted(label_counts.items(), key=lambda x: -x[1])]
                        print(f"  Labels: {' | '.join(parts)}", flush=True)

                try:
                    eth    = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip     = eth.data
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)
                except Exception:
                    continue

                proto    = int(ip.p)
                src_port = dst_port = 0
                tcp_flags = tcp_win_size = tcp_seq = tcp_ack = tcp_urg = None
                udp_len     = None
                payload_hex = ""
                proto_type  = "OTHER"

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp          = ip.data
                    src_port     = tcp.sport
                    dst_port     = tcp.dport
                    tcp_flags    = int(tcp.flags)
                    tcp_win_size = int(tcp.win)
                    tcp_seq      = int(tcp.seq)
                    tcp_ack      = int(tcp.ack)
                    tcp_urg      = int(tcp.urp)
                    proto_type   = "TCP"
                    if len(tcp.data) > 0:
                        payload_hex = tcp.data[:12].hex()

                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp        = ip.data
                    src_port   = udp.sport
                    dst_port   = udp.dport
                    udp_len    = int(udp.ulen)
                    proto_type = "UDP"
                    if len(udp.data) > 0:
                        payload_hex = udp.data[:12].hex()

                fk              = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                iat             = ts - flow_last_ts[fk] if fk in flow_last_ts else 0.0
                flow_last_ts[fk]    = ts
                pkt_rank            = flow_pkt_count[fk]
                flow_pkt_count[fk] += 1
                direction = 1 if dst_port < src_port else 0

                provider_label = None
                category_label = None

                # Source 1: flow already labeled
                if fk in flow_label_cache:
                    provider_label, category_label = flow_label_cache[fk]
                else:
                    # Source 2: IP label cache (O(1), pre-computed)
                    for cip in (dst_ip, src_ip):
                        if cip in ip_label_cache:
                            provider_label, category_label = ip_label_cache[cip]
                            break

                    # Source 3: TTL-aware DNS fallback
                    if category_label is None:
                        for cip in (dst_ip, src_ip):
                            entry = global_dns.get(cip)
                            if entry:
                                domain, expire = entry
                                if ts <= expire:
                                    result = domain_to_labels(domain)
                                    if result:
                                        provider_label, category_label = result
                                        break

                    if category_label is not None:
                        flow_label_cache[fk] = (provider_label, category_label)

                if category_label is not None:
                    labeled += 1
                    label_counts[category_label] += 1

                rows.append({
                    "flow_id":        fk,
                    "pkt_rank":       pkt_rank,
                    "timestamp":      ts,
                    "ip_hdr_len":     int(ip.hl) * 4,
                    "ip_tos":         int(ip.tos),
                    "ip_total_len":   int(ip.len),
                    "ip_flags":       0,
                    "ip_ttl":         int(ip.ttl),
                    "proto":          proto,
                    "tcp_flags":      tcp_flags,
                    "tcp_win_size":   tcp_win_size,
                    "tcp_seq":        tcp_seq,
                    "tcp_ack":        tcp_ack,
                    "tcp_urg":        tcp_urg,
                    "udp_len":        udp_len,
                    "payload_hex":    payload_hex,
                    "frame_len":      len(buf),
                    "direction":      direction,
                    "iat":            iat,
                    "src_ip":         src_ip,
                    "dst_ip":         dst_ip,
                    "src_port":       src_port,
                    "dst_port":       dst_port,
                    "proto_type":     proto_type,
                    "provider_label": provider_label,
                    "category_label": category_label,
                    "traffic_type":   category_label,
                })

                if len(rows) >= batch_size:
                    _flush()

        _flush(final=True)
        grand_total   += total
        grand_labeled += labeled
        file_elapsed   = time.time() - file_start
        total_elapsed  = time.time() - pass_start
        pct            = labeled / total * 100 if total else 0

        print(f"\n  File complete:")
        print(f"    Packets:  {total:,} | Labeled: {labeled:,} ({pct:.1f}%)")
        print(f"    Flows:    {len(flow_label_cache):,} cached globally")
        print(f"    Time:     {_fmt_time(file_elapsed)} this file | "
              f"{_fmt_time(total_elapsed)} total")
        print(f"    Labels:   ", end="")
        parts = [f"{k}={v:,}" for k, v in
                 sorted(label_counts.items(), key=lambda x: -x[1])]
        print(" | ".join(parts) if parts else "none")

    # Final summary
    total_elapsed = time.time() - pass_start
    pct_all = grand_labeled / grand_total * 100 if grand_total else 0
    print(f"\n{'='*60}")
    print(f"PASS 2 COMPLETE in {_fmt_time(total_elapsed)}")
    print(f"  Total packets:        {grand_total:,}")
    print(f"  Total labeled:        {grand_labeled:,} ({pct_all:.1f}%)")
    print(f"  Unique labeled flows: {len(flow_label_cache):,}")

    all_out = sorted(glob.glob(os.path.join(output_dir, "*_batch_*.parquet")))
    if all_out:
        s = pd.read_parquet(all_out[0])
        print(f"\n  Class distribution (first batch):")
        print(s["category_label"].value_counts().to_string())
        print(f"\n  Unique flows per class (first batch):")
        print(
            s[s["category_label"].notna()]
            .groupby("category_label")["flow_id"]
            .nunique()
            .sort_values(ascending=False)
            .to_string()
        )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    pcap_files = sorted(glob.glob(os.path.join(PCAP_DIR, "*.pcap")))
    print(f"Found {len(pcap_files)} PCAP file(s) in {PCAP_DIR}")

    if not pcap_files:
        print("No PCAP files found. Check PCAP_DIR.")
        exit(1)

    dns_map_path = os.path.join(OUTPUT_DIR, "global_dns_map.parquet")
    if os.path.exists(dns_map_path):
        print(f"\nFound existing DNS map — loading and skipping pass 1")
        dns_df = pd.read_parquet(dns_map_path)
        ip_label_cache = {}
        for _, row in dns_df.iterrows():
            result = domain_to_labels(str(row["domain"]))
            if result:
                ip_label_cache[row["ip"]] = result
        print(f"Loaded {len(ip_label_cache):,} labeled IPs")
        print(f"If this number is low (< 500), delete the DNS map and rerun.")
    else:
        ip_label_cache = run_pass1(pcap_files, OUTPUT_DIR)

    run_pass2(pcap_files, ip_label_cache, OUTPUT_DIR)
