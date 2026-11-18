# Dataset Documentation: Apache SpamAssassin Public Corpus

## 1. Overview and Source Attribution
The dataset used in this project is sourced from the official **Apache SpamAssassin Public Corpus**:
- **Source URL:** https://spamassassin.apache.org/old/publiccorpus/
- **License / Usage:** Publicly available benchmark dataset provided by the Apache Software Foundation for spam filtering and anti-abuse research.
- **Corpus Version Archives Used:**
  - `20030228_easy_ham.tar.bz2` (2,500 legitimate emails)
  - `20030228_spam.tar.bz2` (500 spam emails)
  - `20030228_spam_2.tar.bz2` (1,397 spam emails)

## 2. Dataset Structure and Processing
Raw emails are standard RFC 822 / MIME format containing headers (`Subject`, `From`, `To`, `Date`, `Content-Type`) and body payloads.

### Extraction Pipeline:
1. **RFC 822 Parsing**: Emails are parsed using Python's `email` library with `BytesParser`.
2. **Subject Extraction**: Headers are decoded handling RFC 2047 multi-byte encodings (UTF-8, Latin-1, ASCII).
3. **Multipart & HTML Handling**: Traverses MIME parts, extracts `text/plain` content, and uses BeautifulSoup to parse `text/html` into clean textual content.
4. **Subject + Body Combination**: Format: `Subject: <subject>\n\n<body>`.
5. **Deduplication**: Exact text duplicates (108 detected) are eliminated.
6. **Filtering**: Non-informative/corrupted emails under 15 characters are dropped.

## 3. Dataset Characteristics
- **Total Records:** 4,289
- **Legitimate (Ham) Emails:** 2,472 (57.6%)
- **Spam Emails:** 1,817 (42.4%)
- **Class Balance:** Approximately 58:42 ratio, suitable for supervised classification without severe minority class distortion.

### Schema of `data/processed/emails_cleaned.csv`:
| Column | Type | Description |
| :--- | :--- | :--- |
| `text` | string | Full parsed text (Subject + Body) |
| `subject` | string | Decoded email subject line |
| `label` | integer | Binary label: `0` = Ham (Legitimate), `1` = Spam |
| `label_name` | string | Categorical label: `"ham"` or `"spam"` |
| `source_archive` | string | Origin archive name in raw corpus |
| `char_length` | integer | Total character count of parsed email |
| `word_count` | integer | Total whitespace-separated token count |

## 4. Privacy and Anonymization
All email headers such as personal recipient addresses, server relay paths, and internal authentication tokens were parsed out during extraction. The text feature only contains the message Subject and Content Body.
