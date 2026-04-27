from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Optional


SENT_FOLDER_NAMES = {"sent", "sent_items", "_sent", "_sent_mail"}


@dataclass
class ParsedEmail:
    file_path: str
    mailbox_user: str
    folder_name: str
    message_id: Optional[str]
    date_raw: Optional[str]
    sender_raw: Optional[str]
    to_raw: Optional[str]
    cc_raw: Optional[str]
    bcc_raw: Optional[str]
    subject_raw: Optional[str]


def iter_mailbox_dirs(maildir_root: Path) -> Iterable[Path]:
    """Yield first-level mailbox directories in the Enron maildir."""
    for child in sorted(maildir_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield child


def iter_sent_email_files(maildir_root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield (mailbox_user, folder_name, email_file_path) for sent-folder emails."""
    for mailbox_dir in iter_mailbox_dirs(maildir_root):
        mailbox_user = mailbox_dir.name
        for subdir in mailbox_dir.rglob("*"):
            if not subdir.is_dir():
                continue
            folder_name = subdir.name.lower()
            if folder_name not in SENT_FOLDER_NAMES:
                continue
            for email_file in subdir.iterdir():
                if email_file.is_file():
                    yield mailbox_user, subdir.name, email_file


def parse_email_file(mailbox_user: str, folder_name: str, email_file: Path) -> ParsedEmail:
    """Parse headers from one RFC822-style email file."""
    with email_file.open("rb") as f:
        msg = BytesParser(policy=policy.default).parse(f, headersonly=False)

    return ParsedEmail(
        file_path=str(email_file),
        mailbox_user=mailbox_user,
        folder_name=folder_name,
        message_id=msg.get("Message-ID"),
        date_raw=msg.get("Date"),
        sender_raw=msg.get("From"),
        to_raw=msg.get("To"),
        cc_raw=msg.get("Cc"),
        bcc_raw=msg.get("Bcc"),
        subject_raw=msg.get("Subject"),
    )
