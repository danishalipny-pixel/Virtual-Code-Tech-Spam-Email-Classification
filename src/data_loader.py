"""Data loading and extraction module for the Apache SpamAssassin Public Corpus.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Project: Spam or Not Spam – Email Classification System
"""

from email import policy
from email.header import decode_header
from email.parser import BytesParser
import logging
import os
from pathlib import Path
import re
import tarfile
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base URLs and archive definitions
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = [
    {
        "filename": "20030228_easy_ham.tar.bz2",
        "label": "ham",
        "target_dir": "easy_ham",
    },
    {
        "filename": "20030228_spam.tar.bz2",
        "label": "spam",
        "target_dir": "spam",
    },
    {
        "filename": "20030228_spam_2.tar.bz2",
        "label": "spam",
        "target_dir": "spam_2",
    },
]


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with user-agent headers and progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Archive already exists: {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")
        return dest_path

    logger.info(f"Downloading {url} -> {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamClassifier/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    logger.info(f"Successfully downloaded {dest_path.name}")
    return dest_path


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a tar.bz2 archive safely."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} to {extract_dir}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=extract_dir)
    logger.info(f"Extraction complete for {archive_path.name}")
    return extract_dir


def decode_header_value(header_val: str) -> str:
    """Safely decode RFC 2047 encoded email headers."""
    if not header_val:
        return ""
    try:
        decoded_parts = decode_header(header_val)
        result = []
        for text, encoding in decoded_parts:
            if isinstance(text, bytes):
                encoding = encoding or "utf-8"
                try:
                    result.append(text.decode(encoding, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    result.append(text.decode("latin-1", errors="replace"))
            else:
                result.append(str(text))
        return " ".join(result).strip()
    except Exception:
        return str(header_val).strip()


def extract_text_from_payload(part) -> str:
    """Extract plain text from an email message part, stripping HTML if needed."""
    try:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""

        charset = part.get_content_charset() or "utf-8"
        try:
            raw_text = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            raw_text = payload.decode("latin-1", errors="replace")

        content_type = part.get_content_type()
        if content_type == "text/html":
            soup = BeautifulSoup(raw_text, "html.parser")
            return soup.get_text(separator=" ").strip()
        return raw_text.strip()
    except Exception:
        return ""


def parse_raw_email(file_path: Path) -> dict:
    """Parse a single raw RFC 822 email file into structured subject and body text."""
    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        msg = BytesParser(policy=policy.compat32).parsebytes(raw_bytes)

        # Subject extraction
        raw_subject = msg.get("Subject", "")
        subject = decode_header_value(raw_subject)

        # Body extraction
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    part_text = extract_text_from_payload(part)
                    if part_text:
                        body_parts.append(part_text)
        else:
            body_parts.append(extract_text_from_payload(msg))

        body = "\n".join(body_parts).strip()

        # Combine subject and body for classification
        if subject and body:
            full_text = f"Subject: {subject}\n\n{body}"
        elif subject:
            full_text = f"Subject: {subject}"
        else:
            full_text = body

        return {
            "subject": subject,
            "body": body,
            "full_text": full_text,
            "filename": file_path.name,
        }
    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return {"subject": "", "body": "", "full_text": "", "filename": file_path.name}


def load_and_process_dataset(
    data_dir: Path = Path("data"),
    force_download: bool = False,
) -> pd.DataFrame:
    """Download, extract, parse, clean, and save the SpamAssassin dataset.

    Returns:
        pd.DataFrame: Cleaned dataset containing ['text', 'label', 'label_name', 'length']
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    corpus_dir = raw_dir / "corpus"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    processed_file = processed_dir / "emails_cleaned.csv"
    if processed_file.exists() and not force_download:
        logger.info(f"Loading already processed dataset from {processed_file}")
        df = pd.read_csv(processed_file)
        logger.info(f"Loaded {len(df)} records. Class distribution:\n{df['label_name'].value_counts().to_dict()}")
        return df

    # Step 1: Download and extract archives
    records = []
    for arch in ARCHIVES:
        archive_name = arch["filename"]
        archive_path = raw_dir / archive_name
        url = SPAMASSASSIN_BASE_URL + archive_name
        download_file(url, archive_path)

        extract_target = corpus_dir / arch["target_dir"]
        if not extract_target.exists() or len(list(extract_target.glob("*"))) == 0:
            extract_archive(archive_path, corpus_dir)

        # Some archives extract to an inner subfolder (e.g., easy_ham/ or spam/ or spam_2/)
        candidate_dirs = [
            corpus_dir / arch["target_dir"],
            corpus_dir / arch["filename"].replace(".tar.bz2", "").split("_", 1)[-1],
        ]
        target_dir = None
        for d in candidate_dirs:
            if d.exists() and d.is_dir():
                target_dir = d
                break

        if not target_dir:
            # Look inside corpus_dir for matching names
            for item in corpus_dir.iterdir():
                if item.is_dir() and (arch["target_dir"] in item.name or arch["label"] in item.name):
                    target_dir = item
                    break

        if not target_dir or not target_dir.exists():
            logger.warning(f"Could not locate extracted directory for {archive_name}")
            continue

        email_files = [p for p in target_dir.glob("*") if p.is_file() and not p.name.startswith("cmds")]
        logger.info(f"Parsing {len(email_files)} files from {target_dir.name} ({arch['label']})...")

        for file_path in email_files:
            parsed = parse_raw_email(file_path)
            if parsed["full_text"] and len(parsed["full_text"].strip()) > 10:
                records.append({
                    "text": parsed["full_text"],
                    "subject": parsed["subject"],
                    "label": 1 if arch["label"] == "spam" else 0,
                    "label_name": arch["label"],
                    "source_archive": archive_name,
                })

    df = pd.DataFrame(records)
    initial_count = len(df)
    logger.info(f"Raw parsed records count: {initial_count}")

    # Step 2: Cleaning & deduplication
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 15]

    # Deduplicate on email text
    duplicates_count = df.duplicated(subset=["text"]).sum()
    logger.info(f"Found {duplicates_count} duplicate records. Dropping duplicates...")
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df["char_length"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    # Step 3: Save cleaned dataset
    df.to_csv(processed_file, index=False, encoding="utf-8")
    logger.info(f"Cleaned dataset saved to {processed_file} ({len(df)} records)")
    logger.info(f"Class breakdown: {df['label_name'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    df = load_and_process_dataset(force_download=False)
    print("\nDataset Summary:")
    print(df.info())
    print("\nClass Distribution:")
    print(df["label_name"].value_counts())
