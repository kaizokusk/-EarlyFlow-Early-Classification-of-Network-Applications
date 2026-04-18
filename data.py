import os
import glob
import numpy as np
import pandas as pd
import ipaddress
import json
import gc
# -----------------------------
# LOAD DNS MAP
# -----------------------------
with open("../dns_map.json", "r") as f:
    dns_category_map = json.load(f)

# -----------------------------
# HELPER
# -----------------------------
def is_private(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except:
        return False

# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(path, dns_category_map):

    df = pd.read_csv(
        path,
        low_memory=False,
        na_values=["", "nan", "N/A"],
        on_bad_lines="skip"
    )

    def clean_numeric(series):
        return (
            series.astype(str)
            .str.split(",")
            .str[0]
            .astype(float)
        )

    df["ip.proto"] = clean_numeric(df["ip.proto"])
    df["tcp.srcport"] = clean_numeric(df["tcp.srcport"])
    df["tcp.dstport"] = clean_numeric(df["tcp.dstport"])
    df["udp.srcport"] = clean_numeric(df["udp.srcport"])
    df["udp.dstport"] = clean_numeric(df["udp.dstport"])


    # ---- ports ----
    df["src_port"] = df["tcp.srcport"].fillna(df["udp.srcport"])
    df["dst_port"] = df["tcp.dstport"].fillna(df["udp.dstport"])

    # ---- drop invalid ----
    df = df.dropna(subset=["ip.src", "ip.dst", "src_port", "dst_port"]).copy()

    df["src_port"] = df["src_port"].astype(int)
    df["dst_port"] = df["dst_port"].astype(int)

    # ---- flow key (bi-directional) ----
    a = df["ip.src"] + ":" + df["src_port"].astype(str)
    b = df["ip.dst"] + ":" + df["dst_port"].astype(str)
    df["flow_key"] = np.where(a < b, a + "_" + b, b + "_" + a)
    df["flow_key"] = path + "_" + df["flow_key"]
    # ---- private IP ----
    df["src_private"] = df["ip.src"].apply(is_private)
    df["dst_private"] = df["ip.dst"].apply(is_private)

    # ---- client IP ----
    df["client_ip"] = np.where(df["src_private"], df["ip.src"], df["ip.dst"])

    # ---- DNS extraction ----
    dns_df = df.loc[df["dns.qry.name"].notna(), ["ip.src", "frame.time_epoch", "dns.qry.name"]].copy()
    dns_df["dns_label"] = dns_df["dns.qry.name"].map(dns_category_map).fillna("other")

    dns_map = dns_df[["ip.src", "dns_label", "frame.time_epoch"]].copy()


    df = df.sort_values("frame.time_epoch")
    dns_map = dns_map.sort_values("frame.time_epoch")
    

    # ---- merge ----
    df = pd.merge_asof(
        df,
        dns_map,
        left_on="frame.time_epoch",
        right_on="frame.time_epoch",
        left_by="client_ip",
        right_by="ip.src",
        direction="backward",
        tolerance=30
    )

    df["label"] = df["dns_label"].fillna("other")

    # ---- clean merge cols ----
    if "ip.src_x" in df.columns:
        df["ip.src"] = df["ip.src_x"]
        df = df.drop(columns=["ip.src_x", "ip.src_y"], errors="ignore")

    # ---- make flow_key unique per file ----
    



        # -----------------------------
    # PACKET-LEVEL FEATURE ENGINEERING (BEST SET)
    # -----------------------------

    # ensure correct order
    df = df.sort_values(["flow_key", "frame.time_epoch"])

    # 1. Inter-arrival time
    df["iat"] = df.groupby("flow_key")["frame.time_epoch"].diff().fillna(0)

    # 2. Packet size difference
    df["len_diff"] = df.groupby("flow_key")["frame.len"].diff().fillna(0)

    # 3. Direction (client → server = 1)
    df["direction_flag"] = (df["ip.src"] == df["client_ip"]).astype(int)

    # 4. Relative time (within flow)
    df["relative_time"] = (
        df["frame.time_epoch"] -
        df.groupby("flow_key")["frame.time_epoch"].transform("min")
    )

    # 5. Packet index (position in flow)
    df["packet_idx"] = df.groupby("flow_key").cumcount()

    # 6. Cumulative bytes
    df["cum_bytes"] = df.groupby("flow_key")["frame.len"].cumsum()

    # 7. Signed packet size (direction-aware)
    df["signed_len"] = df["frame.len"] * (2 * df["direction_flag"] - 1)

    # 8. Burst indicator (adaptive threshold per flow)


    df["is_burst"] = (
        df["iat"] < df.groupby("flow_key")["iat"].transform("median")
    ).astype(int)

    # -----------------------------
    # FINAL FEATURE LIST
    # -----------------------------
#     packet_features = [
#         "frame.len",
#         "iat",
#         "len_diff",
#         "direction_flag",
#         "relative_time",
#         "packet_idx",
#         "cum_bytes",
#         "signed_len",
#         "is_burst",
#         "ip.proto",
#         "src_port",
#         "dst_port"
# ]





    # -------- FLOW AGG --------
    flow_features = df.groupby("flow_key").agg({
        "frame.len": ["mean", "std", "min", "max", "sum"],
        "frame.time_epoch": ["min", "max", "count"],
        "iat": ["mean", "std", "max"],
        "len_diff": ["mean", "std"],
        "direction_flag": ["sum", "mean"],  # IMPORTANT
        "src_port": "first",
        "dst_port": "first",
        "ip.proto": "first",
        "label": lambda x: x.mode()[0] if not x.mode().empty else "other"
    })

    # flatten ONCE
    flow_features.columns = ["_".join(col) for col in flow_features.columns]
    flow_features = flow_features.reset_index()


    flow_features = flow_features.rename(columns={
    "frame.len_mean": "len_mean",
    "frame.len_std": "len_std",
    "frame.len_min": "len_min",
    "frame.len_max": "len_max",
    "frame.len_sum": "len_sum",
    "frame.time_epoch_count": "packet_count"
    })


    # -------- DERIVED --------
    flow_features["duration"] = (
        flow_features["frame.time_epoch_max"] -
        flow_features["frame.time_epoch_min"]
    )



    flow_features["len_range"] = (
        flow_features["len_max"] -
        flow_features["len_min"]
    )

    flow_features["len_cv"] = (
        flow_features["len_std"] /
        (flow_features["len_mean"] + 1e-6)
    )

    flow_features["iat_cv"] = (
    flow_features["iat_std"] /
    (flow_features["iat_mean"].replace(0, 1e-6))
    )
    flow_features["byte_rate"] = (
    flow_features["len_sum"] / (flow_features["duration"] + 1e-6)
    )
    flow_features["total_length"] = flow_features["len_sum"]

    # -------- DIRECTION --------
    flow_features["forward_packets"] = flow_features["direction_flag_sum"]

    flow_features["pps"] = (
        flow_features["packet_count"] /
        (flow_features["duration"] + 1e-6)
    )

    flow_features["receiving_packets"] = (
        flow_features["packet_count"] - flow_features["direction_flag_sum"]
    )

    flow_features["forward_ratio"] = (
        flow_features["forward_packets"] /
        (flow_features["packet_count"] + 1e-6)
    )

    flow_features["direction_balance"] = abs(
        flow_features["direction_flag_mean"] - 0.5
    )



    # -------- PORT FLAGS --------
    flow_features["is_https"] = (flow_features["dst_port_first"] == 443).astype(int)
    flow_features["is_dns"] = (flow_features["dst_port_first"] == 53).astype(int)
    flow_features["is_http"] = (flow_features["dst_port_first"] == 80).astype(int)

    df = df[df["label"] != "other"]
    flow_features = flow_features[flow_features["label_<lambda>"] != "other"]
    return df, flow_features


# -----------------------------
# MAIN (STREAMING WRITE)
# -----------------------------
packet_out = "packet_dataset.csv"
flow_out = "flow_dataset.csv"

# remove old files
if os.path.exists(packet_out):
    os.remove(packet_out)
if os.path.exists(flow_out):
    os.remove(flow_out)

first_packet = True
first_flow = True

csv_files = glob.glob("*.csv")

for file in csv_files:
    print(f"Processing: {file}")

    df_packet, df_flow = process_file(file, dns_category_map)

    # add file column
    df_packet["file"] = file
    df_flow["file"] = file

    # ---- STREAM WRITE ----
    df_packet.to_csv(packet_out, mode="a", index=False, header=first_packet)
    df_flow.to_csv(flow_out, mode="a", index=False, header=first_flow)

    first_packet = False
    first_flow = False

    # ---- FREE MEMORY ----
    del df_packet, df_flow
    gc.collect()

print("✅ Done!")
