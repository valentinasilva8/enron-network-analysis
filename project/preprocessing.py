from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from email.utils import getaddresses
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from data_loading import iter_sent_email_files, parse_email_file


SYSTEM_ADDRESSES = {
    "no.address@enron.com",
    "announcements@enron.com",
    "all.worldwide@enron.com",
}


@dataclass
class PreprocessingArtifacts:
    email_level_df: pd.DataFrame
    edges_df: pd.DataFrame
    validation: dict
    sanity: dict


def normalize_email_address(raw_email: str) -> str:
    """Normalize an email address without aggressive alias collapsing."""
    normalized = (raw_email or "").strip().lower().strip("<>").strip()
    return normalized


def extract_addresses(*header_values: Optional[str]) -> list[str]:
    """Extract normalized addresses from To/Cc/Bcc style headers."""
    pairs = getaddresses([h for h in header_values if h])
    normalized = [normalize_email_address(addr) for _, addr in pairs if addr]
    return [addr for addr in normalized if "@" in addr]


def _safe_parse_record(mailbox_user: str, folder_name: str, email_file: Path):
    try:
        return parse_email_file(mailbox_user, folder_name, email_file), None
    except UnicodeDecodeError:
        return None, "encoding_errors"
    except Exception:
        return None, "malformed_headers"


def build_email_level_dataframe(
    maildir_root: Path,
    max_emails: Optional[int] = None,
    allowed_mailbox_users: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse sent emails into an email-level dataframe with validation counters."""
    rows: list[dict] = []
    counts = Counter()
    unique_before_filter: set[str] = set()

    file_iter = iter_sent_email_files(maildir_root)
    for mailbox_user, folder_name, email_file in tqdm(file_iter, desc="Parsing sent emails"):
        if allowed_mailbox_users and mailbox_user not in allowed_mailbox_users:
            continue
        if max_emails and counts["total_scanned"] >= max_emails:
            break

        counts["total_scanned"] += 1
        parsed, parse_error = _safe_parse_record(mailbox_user, folder_name, email_file)
        if parse_error:
            counts["dropped_total"] += 1
            counts[parse_error] += 1
            continue

        sender_candidates = extract_addresses(parsed.sender_raw)
        sender = sender_candidates[0] if sender_candidates else ""
        recipients = extract_addresses(parsed.to_raw, parsed.cc_raw, parsed.bcc_raw)
        recipients = [r for r in recipients if r not in SYSTEM_ADDRESSES]
        recipient_count = len(recipients)

        if not sender or "@" not in sender:
            counts["dropped_total"] += 1
            counts["malformed_headers"] += 1
            continue

        if recipient_count == 0:
            counts["dropped_total"] += 1
            counts["no_recipients"] += 1
            continue

        if recipient_count > 10:
            counts["dropped_total"] += 1
            counts["broadcast_drop"] += 1
            continue

        counts["parsed_success"] += 1
        counts["recipient_total"] += recipient_count
        unique_before_filter.add(sender)
        unique_before_filter.update(recipients)

        rows.append(
            {
                "file_path": parsed.file_path,
                "mailbox_user": parsed.mailbox_user,
                "folder_name": parsed.folder_name,
                "message_id": (parsed.message_id or "").strip(),
                "date_raw": parsed.date_raw,
                "date": pd.to_datetime(parsed.date_raw, errors="coerce", utc=True),
                "sender": sender,
                "recipients": recipients,
                "recipient_count": recipient_count,
                "subject_raw": parsed.subject_raw,
            }
        )

    email_df = pd.DataFrame(rows)
    counts["unique_users_before_filter"] = len(unique_before_filter)
    return email_df, dict(counts)


def _build_edges(email_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = Counter()

    if email_df.empty:
        return pd.DataFrame(columns=["sender", "receiver", "weight", "first_date", "last_date"]), {}

    dedup_df = email_df.copy()
    dedup_df["message_id"] = dedup_df["message_id"].fillna("").astype(str).str.strip()
    with_id = dedup_df["message_id"] != ""
    dedup_subset = dedup_df[with_id]
    no_id_subset = dedup_df[~with_id]
    deduped_with_id = dedup_subset.drop_duplicates(subset=["message_id"], keep="first")
    dedup_df = pd.concat([deduped_with_id, no_id_subset], ignore_index=True)

    counts["unique_message_ids"] = deduped_with_id["message_id"].nunique()
    counts["duplicates_removed"] = len(email_df) - len(dedup_df)

    expanded_rows: list[dict] = []
    unique_after_filter: set[str] = set()
    for _, row in dedup_df.iterrows():
        sender = row["sender"]
        recipients = [r for r in row["recipients"] if r.endswith("@enron.com")]
        sender_is_enron = sender.endswith("@enron.com")
        if not sender_is_enron:
            continue
        for receiver in recipients:
            expanded_rows.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "date": row["date"],
                }
            )
            unique_after_filter.add(sender)
            unique_after_filter.add(receiver)

    counts["total_edges_before_aggregation"] = len(expanded_rows)
    counts["unique_users_after_filter"] = len(unique_after_filter)

    expanded_df = pd.DataFrame(expanded_rows)
    if expanded_df.empty:
        return pd.DataFrame(columns=["sender", "receiver", "weight", "first_date", "last_date"]), dict(counts)

    aggregated = (
        expanded_df.groupby(["sender", "receiver"], as_index=False)
        .agg(weight=("sender", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    counts["total_edges_after_aggregation"] = len(aggregated)

    retained = aggregated[aggregated["weight"] >= 2].copy()
    counts["edges_retained_weight_ge_2"] = len(retained)
    counts["edges_dropped_weight_lt_2"] = len(aggregated) - len(retained)
    return retained, dict(counts)


def _sanity_checks(edges_df: pd.DataFrame, email_df: pd.DataFrame) -> dict:
    if edges_df.empty:
        return {
            "top_10_senders": [],
            "edge_samples": [],
            "total_nodes": 0,
            "core_mailbox_users_found": 0,
            "largest_component_pct": 0.0,
            "isolated_nodes": 0,
        }

    import networkx as nx

    out_weights = edges_df.groupby("sender")["weight"].sum().sort_values(ascending=False)
    top_senders = out_weights.head(10).to_dict()

    samples = edges_df.sample(min(10, len(edges_df)), random_state=42)[["sender", "receiver", "weight"]]
    edge_samples = samples.to_dict(orient="records")

    g = nx.from_pandas_edgelist(edges_df, "sender", "receiver", edge_attr="weight", create_using=nx.Graph())
    total_nodes = g.number_of_nodes()
    components = [len(c) for c in nx.connected_components(g)] if total_nodes else []
    largest = max(components) if components else 0
    largest_pct = (largest / total_nodes * 100.0) if total_nodes else 0.0
    isolated_nodes = len(list(nx.isolates(g)))

    core_users_found = email_df["mailbox_user"].nunique() if "mailbox_user" in email_df.columns else 0

    return {
        "top_10_senders": top_senders,
        "edge_samples": edge_samples,
        "total_nodes": total_nodes,
        "core_mailbox_users_found": int(core_users_found),
        "largest_component_pct": largest_pct,
        "isolated_nodes": isolated_nodes,
    }


def run_preprocessing(
    maildir_root: Path,
    max_emails: Optional[int] = None,
    allowed_mailbox_users: Optional[set[str]] = None,
) -> PreprocessingArtifacts:
    """Run full preprocessing and return dataframes plus validation metadata."""
    email_df, counts_parse = build_email_level_dataframe(maildir_root, max_emails, allowed_mailbox_users)
    edges_df, counts_edges = _build_edges(email_df)

    avg_recipients = (
        counts_parse.get("recipient_total", 0) / counts_parse.get("parsed_success", 1)
        if counts_parse.get("parsed_success", 0) > 0
        else 0.0
    )

    validation = {
        "total_email_files_scanned": counts_parse.get("total_scanned", 0),
        "successfully_parsed": counts_parse.get("parsed_success", 0),
        "successfully_parsed_pct": (
            counts_parse.get("parsed_success", 0) / counts_parse.get("total_scanned", 1) * 100.0
            if counts_parse.get("total_scanned", 0) > 0
            else 0.0
        ),
        "failed_to_parse_dropped": counts_parse.get("dropped_total", 0),
        "failed_to_parse_dropped_pct": (
            counts_parse.get("dropped_total", 0) / counts_parse.get("total_scanned", 1) * 100.0
            if counts_parse.get("total_scanned", 0) > 0
            else 0.0
        ),
        "no_recipients": counts_parse.get("no_recipients", 0),
        "malformed_headers": counts_parse.get("malformed_headers", 0),
        "encoding_errors": counts_parse.get("encoding_errors", 0),
        "unique_message_ids": counts_edges.get("unique_message_ids", 0),
        "duplicates_removed": counts_edges.get("duplicates_removed", 0),
        "average_recipients_per_email": avg_recipients,
        "emails_dropped_over_10_recipients": counts_parse.get("broadcast_drop", 0),
        "emails_dropped_over_10_recipients_pct": (
            counts_parse.get("broadcast_drop", 0) / counts_parse.get("total_scanned", 1) * 100.0
            if counts_parse.get("total_scanned", 0) > 0
            else 0.0
        ),
        "total_edges_before_aggregation": counts_edges.get("total_edges_before_aggregation", 0),
        "total_edges_after_aggregation": counts_edges.get("total_edges_after_aggregation", 0),
        "edges_dropped_weight_lt_2": counts_edges.get("edges_dropped_weight_lt_2", 0),
        "edges_dropped_weight_lt_2_pct": (
            counts_edges.get("edges_dropped_weight_lt_2", 0) / counts_edges.get("total_edges_after_aggregation", 1) * 100.0
            if counts_edges.get("total_edges_after_aggregation", 0) > 0
            else 0.0
        ),
        "edges_retained_weight_ge_2": counts_edges.get("edges_retained_weight_ge_2", 0),
        "unique_users_before_filter": counts_parse.get("unique_users_before_filter", 0),
        "unique_users_after_enron_filter": counts_edges.get("unique_users_after_filter", 0),
        "date_range_start": email_df["date"].min(),
        "date_range_end": email_df["date"].max(),
    }

    sanity = _sanity_checks(edges_df, email_df)
    return PreprocessingArtifacts(email_level_df=email_df, edges_df=edges_df, validation=validation, sanity=sanity)


def validation_report_text(validation: dict, sanity: dict) -> str:
    """Render the required validation + sanity report."""
    start = validation.get("date_range_start")
    end = validation.get("date_range_end")
    start_s = start.isoformat() if pd.notna(start) else "N/A"
    end_s = end.isoformat() if pd.notna(end) else "N/A"

    lines = [
        "=== PREPROCESSING VALIDATION REPORT ===",
        f"Total email files scanned:        {validation.get('total_email_files_scanned', 0)}",
        f"Successfully parsed:              {validation.get('successfully_parsed', 0)}  ({validation.get('successfully_parsed_pct', 0):.2f}%)",
        f"Failed to parse (dropped):        {validation.get('failed_to_parse_dropped', 0)}  ({validation.get('failed_to_parse_dropped_pct', 0):.2f}%)",
        f"  - No recipients:                {validation.get('no_recipients', 0)}",
        f"  - Malformed headers:            {validation.get('malformed_headers', 0)}",
        f"  - Encoding errors:              {validation.get('encoding_errors', 0)}",
        f"Unique Message-IDs:               {validation.get('unique_message_ids', 0)}",
        f"Duplicates removed:               {validation.get('duplicates_removed', 0)}",
        f"Average recipients per email:     {validation.get('average_recipients_per_email', 0):.2f}",
        f"Emails dropped (>10 recipients):  {validation.get('emails_dropped_over_10_recipients', 0)}  ({validation.get('emails_dropped_over_10_recipients_pct', 0):.2f}%)",
        f"Total edges (before aggregation): {validation.get('total_edges_before_aggregation', 0)}",
        f"Total edges (after aggregation):  {validation.get('total_edges_after_aggregation', 0)}",
        f"Edges dropped (weight < 2):       {validation.get('edges_dropped_weight_lt_2', 0)}  ({validation.get('edges_dropped_weight_lt_2_pct', 0):.2f}%)",
        f"Edges retained (weight >= 2):     {validation.get('edges_retained_weight_ge_2', 0)}",
        f"Unique users (before filtering):  {validation.get('unique_users_before_filter', 0)}",
        f"Unique users (after @enron filter): {validation.get('unique_users_after_enron_filter', 0)}",
        f"Date range:                       {start_s} to {end_s}",
        "",
        "=== SANITY CHECK: TOP 10 SENDERS ===",
    ]
    for sender, degree in sanity.get("top_10_senders", {}).items():
        lines.append(f"- {sender}: {degree}")

    lines += ["", "=== SANITY CHECK: EDGE SAMPLE ==="]
    for row in sanity.get("edge_samples", []):
        lines.append(f"- {row['sender']} -> {row['receiver']} (weight={row['weight']})")

    lines += [
        "",
        "=== SANITY CHECK: GRAPH PLAUSIBILITY ===",
        f"- Total nodes: {sanity.get('total_nodes', 0)}",
        f"- Core mailbox users found: {sanity.get('core_mailbox_users_found', 0)} / 150",
        f"- Largest connected component: {sanity.get('largest_component_pct', 0.0):.2f}% of nodes",
        f"- Isolated nodes: {sanity.get('isolated_nodes', 0)}",
        "",
    ]
    return "\n".join(lines)
