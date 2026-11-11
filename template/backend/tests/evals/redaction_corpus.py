"""A synthetic corpus for the redaction evals (#78): documents the way a home-decision app sees
them (a tenancy enquiry, a bank detail in a message, a landlord's contact), every personal value
made up. Nothing here is real: example.com is reserved, 07700 900xxx is Ofcom's drama range, the
national insurance prefix QQ is never given out, and the card, NHS and IBAN numbers are the
well-known test values or built by the public check-digit rules below.

Two parts:
- LEAK_CORPUS: documents with personal values that the default detectors cover, and the exact
  values that must never be sent. `counts` is what each kind should replace, worked out from how
  each document is built (one value of each kind listed, the name counted per mention), not from
  running the detectors.
- MUST_NOT_CHANGE: property text with numbers and dates that look like personal data but are not.
  Redaction must leave it as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A short list of made-up people. The app passes the user's own name as a known value.
NAMES = [
    "Priya Shah", "Tom Ellis", "Aisha Khan", "Marcus Webb", "Ines Duarte", "Callum Reid",
    "Hana Sato", "Owen Price", "Leila Haddad", "Ben Okafor", "Maya Lindqvist", "Sean Doyle",
    "Zara Mirza", "Peter Novak", "Ruth Adeyemi", "Dan Fraser", "Nora Eriksen", "Luis Ortega",
    "Freya Holt", "Idris Bello",
]  # fmt: skip


def luhn_complete(partial: str) -> str:
    """`partial` plus the check digit that makes a number pass the Luhn check (the published
    rule: double every second digit from the right of the full number, subtract 9 from any result
    above 9, and the sum must be a multiple of 10)."""
    for check in "0123456789":
        digits = [int(c) for c in partial + check]
        total = 0
        for index, digit in enumerate(reversed(digits)):
            if index % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        if total % 10 == 0:
            return partial + check
    raise AssertionError("every prefix has a check digit")


def nhs_complete(nine: str) -> str | None:
    """`nine` plus its NHS modulus 11 check digit (weights 10 down to 2; 11 minus the remainder,
    where 11 means 0), or None when the check digit would be 10 (no such number exists)."""
    total = sum(int(d) * w for d, w in zip(nine, range(10, 1, -1), strict=True))
    check = 11 - total % 11
    check = 0 if check == 11 else check
    return None if check == 10 else nine + str(check)


def email(i: int) -> str:
    return f"{NAMES[i].split()[0].lower()}.{NAMES[i].split()[1].lower()}@example.com"


def phone(i: int) -> str:
    return f"07700 900{i:03d}"


def nino(i: int) -> str:
    return f"QQ{i:02d}{(i * 7) % 100:02d}56{'ABCD'[i % 4]}"


def card(i: int) -> str:
    digits = luhn_complete(f"411111{i:09d}")  # a Visa-style prefix, 15 digits, then the check
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"


def nhs(i: int) -> str:
    number = None
    candidate = 400 + i
    while number is None:
        number = nhs_complete(f"943476{candidate:03d}"[:9])
        candidate += 100
    return f"{number[0:3]} {number[3:6]} {number[6:10]}"


@dataclass(frozen=True)
class Document:
    name: str
    text: str
    known: dict[str, list[str]]
    allow: list[str] = field(default_factory=list)
    seeded: tuple[str, ...] = ()  # every value that must not reach the provider
    counts: dict[str, int] = field(default_factory=dict)  # kind -> values replaced


PROPERTY = "Flat 3, 12 Example Road, Leeds LS1 4AP"


def build(i: int) -> Document:
    """Document i: its own person, and a different mix of kinds as i changes. The mix is chosen by
    i % 5, so every kind appears several times across the twenty documents."""
    who = NAMES[i]
    known = {"NAME": [who]}
    seeded = [who, email(i)]
    counts = {"NAME": 1, "EMAIL": 1}
    parts = [
        f"{who} would like to rent {PROPERTY} from 1 March 2027 at the listed rent.",
        f"Reply to {email(i)}.",
    ]
    mix = i % 5
    if mix in (0, 2, 4):
        parts.append(f"Call {phone(i)} after 6pm.")
        seeded.append(phone(i))
        counts["PHONE"] = 1
    if mix in (0, 1):
        parts.append(f"National insurance number {nino(i)} for the reference check.")
        seeded.append(nino(i))
        counts["NINO"] = 1
    if mix in (1, 3):
        account = f"{10000000 + i * 137:08d}"
        parts.append(f"Pay the holding deposit to sort code 20-{i:02d}-45, account {account}.")
        seeded += [f"20-{i:02d}-45", account]
        counts["SORT_CODE"] = 1
        counts["ACCOUNT"] = 1
    if mix in (2, 3):
        parts.append(f"Card on file {card(i)}.")
        seeded.append(card(i))
        counts["CARD"] = 1
    if mix == 4:
        parts.append(f"NHS number {nhs(i)} for the accessibility form.")
        seeded.append(nhs(i))
        counts["NHS"] = 1
    if mix in (0, 3):
        born = f"{1 + i % 28:02d}/{1 + i % 12:02d}/{1980 + i}"
        parts.append(f"Date of birth {born}.")
        seeded.append(born)
        counts["DOB"] = 1
    if mix == 1:
        iban = f"GB82 WEST 1234 5698 76{54 + i:02d} 32"
        parts.append(f"Refund to IBAN {iban}.")
        seeded.append(iban)
        counts["IBAN"] = 1
    return Document(
        name=f"doc-{i:02d}",
        text=" ".join(parts),
        known=known,
        allow=[PROPERTY],
        seeded=tuple(seeded),
        counts=counts,
    )


LEAK_CORPUS: tuple[Document, ...] = tuple(build(i) for i in range(len(NAMES)))

# Known misses of the default detectors (decision 0006, issue #97): logged as feedback by the live
# eval, never asserted. A name that is not the app's own, a free-form address, a NINO written
# without its last letter.
KNOWN_MISSES: tuple[tuple[str, str], ...] = (
    ("a third party's name", "The previous tenant, Gerald Pike, left a forwarding note."),
    ("a free-form address", "Post goes to 41 Quarry Lane, Otley."),
    ("a NINO without its last letter", "NI QQ123456 on the form."),
    ("a date of birth under a neutral field name", '{"ref": "12/03/1980"}'),
)

# Property text that looks like personal data and is not. Redaction must leave all of it alone.
MUST_NOT_CHANGE: tuple[str, ...] = (
    "Flat 3, 12 Example Road, Leeds LS1 4AP. Rent £1,250 pcm from 01/03/2027. EPC band C. "
    "Council tax band D. Agent: Sam Patel, Example Lettings.",
    "Holding deposit £288, tenancy deposit £1,442 (five weeks), protected within 30 days.",
    "Available from 1 March 2027. Minimum tenancy 12 months. Viewing on 14/12/2026 at 10:30.",
    "Leasehold, 95 years remaining, ground rent £250 a year, service charge £1,980 a year.",
    "Title number WYK123456. Reference 12345678 for the agent. Move in 01-03-27.",
    "Energy certificate 0123-4567-8901-2345-6789 valid until 2034. Floor area 62 sq m.",
    "Gas meter reading 004567, electricity 0012345 on 01/03/2027. Water meter 56 789.",
    "Built 1998. Council tax £1,745 for 2026 to 2027. Broadband up to 900 Mbps. Parking: 1 space.",
    "Total cost over 12 months: £15,000 rent plus £1,200 bills plus £288 fees.",
    "Postcode LS1 4AP, ward City and Hunslet, flood risk very low, crime rating below average.",
)
