"""Isolated tests for the new-patient-from-document demographics paths.

Covers:

- The HL7 PID-5/7/8/11/13 parser exposed via `Hl7Message.patient_identity`.
- The workbook full-name splitter the endpoint uses to turn the
  workbook's single `patient_name` cell into `(given_name, family_name)`.

The vision-extracted paths (fax_packet, referral_letter) aren't tested
here — that surface depends on Anthropic returning the new
`patient_identity` field in tool-call output, which is exercised by
manual smoke tests against Hetzner. The schema change itself is
covered by the existing schema tests.
"""

from __future__ import annotations

from datetime import date

from app.extraction.hl7 import parse_hl7_message
from app.extraction.schemas import PatientIdentity
from app.main import _identity_from_optional, _split_full_name


# ─── HL7 PID parser ───────────────────────────────────────────────────


def _adt_with(pid_segment: str) -> bytes:
    """Build a minimal ADT-A08 message ending with the PID segment we want
    to test. MSH + EVN are required for the message to parse, but their
    contents don't matter for the PID-driven assertions below."""
    msh = (
        "MSH|^~\\&|REGISTRATION|MEMORIAL|MIRTH|MEMORIAL|"
        "20260506143215||ADT^A08^ADT_A01|MSG-001|P|2.5"
    )
    evn = "EVN|A08|20260506143215||registration update"
    msg = "\r".join([msh, evn, pid_segment])
    return msg.encode("ascii")


class TestHl7PatientIdentity:
    def test_full_pid_extracts_every_field(self) -> None:
        # Standard v2.5 PID with name, DOB, sex, address, phone.
        pid = (
            "PID|1||EMR-12345^^^MEMORIAL^MR||DOE^JANE^M^^^^L||"
            "19850315|F|||"
            "123 Main St^^Springfield^IL^62701||"
            "(555) 555-0123|||EN|S"
        )
        msg = parse_hl7_message(_adt_with(pid), source_document_id="DocumentReference/test-1")
        assert msg.patient_mrn == "EMR-12345"
        ident = msg.patient_identity
        assert ident is not None
        assert ident.given_name == "JANE"
        assert ident.family_name == "DOE"
        assert ident.date_of_birth == date(1985, 3, 15)
        assert ident.sex == "female"
        assert ident.address == "123 Main St, Springfield IL 62701"
        assert ident.phone == "(555) 555-0123"

    def test_pid_with_only_mrn_returns_none_identity(self) -> None:
        # Senders that emit only PID-3 (MRN) should NOT manufacture an
        # empty PatientIdentity object — return None so the UI shows
        # blank fields rather than a half-populated form.
        pid = "PID|1||EMR-99999^^^^MR"
        msg = parse_hl7_message(_adt_with(pid), source_document_id="DocumentReference/test-2")
        assert msg.patient_mrn == "EMR-99999"
        assert msg.patient_identity is None

    def test_pid_sex_codes_map_to_lowercase_vocabulary(self) -> None:
        # PID-8 is a single-letter code; downstream Pydantic expects
        # the lowercase Sex literal.
        pid_male = "PID|1||EMR-1^^^^MR||SMITH^JOHN||19700101|M"
        pid_other = "PID|1||EMR-2^^^^MR||SMITH^TAYLOR||19700101|O"
        pid_unknown = "PID|1||EMR-3^^^^MR||SMITH^ALEX||19700101|U"
        for pid_segment, expected in (
            (pid_male, "male"),
            (pid_other, "other"),
            (pid_unknown, "unknown"),
        ):
            msg = parse_hl7_message(_adt_with(pid_segment), source_document_id="DocumentReference/x")
            assert msg.patient_identity is not None
            assert msg.patient_identity.sex == expected

    def test_pid_partial_dob_emits_none(self) -> None:
        # PID-7 of length < 8 isn't a valid date; helper returns None
        # rather than raising.
        pid = "PID|1||EMR-1^^^^MR||DOE^JANE||1985"
        msg = parse_hl7_message(_adt_with(pid), source_document_id="DocumentReference/x")
        assert msg.patient_identity is not None
        assert msg.patient_identity.date_of_birth is None

    def test_pid_address_with_missing_components_skips_blanks(self) -> None:
        # A PID-11 with no city/state/zip should fall back to the street
        # alone — not produce a dangling ", , ".
        pid = "PID|1||EMR-1^^^^MR||DOE^JANE||19850315|F|||"\
              "456 Oak Ave"
        msg = parse_hl7_message(_adt_with(pid), source_document_id="DocumentReference/x")
        assert msg.patient_identity is not None
        assert msg.patient_identity.address == "456 Oak Ave"


# ─── Workbook name splitter ───────────────────────────────────────────


class TestSplitFullName:
    def test_comma_separated_last_first(self) -> None:
        assert _split_full_name("Doe, Jane") == ("Jane", "Doe")

    def test_comma_separated_with_middle(self) -> None:
        assert _split_full_name("Smith, John Robert") == ("John Robert", "Smith")

    def test_space_separated_first_last(self) -> None:
        assert _split_full_name("Jane Doe") == ("Jane", "Doe")

    def test_space_separated_with_middle(self) -> None:
        # "First Middle Last" — the LAST token is family, the rest is given.
        assert _split_full_name("John Robert Smith") == ("John Robert", "Smith")

    def test_single_token_is_treated_as_family(self) -> None:
        # Ambiguous case — keep the half a clinician can search by
        # ("Smith") in the family slot. The user fills in given on
        # the create-patient form.
        assert _split_full_name("Smith") == (None, "Smith")

    def test_blank_input_returns_pair_of_nones(self) -> None:
        assert _split_full_name(None) == (None, None)
        assert _split_full_name("") == (None, None)
        assert _split_full_name("   ") == (None, None)


# ─── Identity projection helper ───────────────────────────────────────


class TestIdentityFromOptional:
    def test_none_input_yields_all_nones(self) -> None:
        out = _identity_from_optional(None)
        assert out == {
            "given_name": None, "family_name": None, "date_of_birth": None,
            "sex": None, "address": None, "phone": None,
        }

    def test_populated_identity_projects_to_dict(self) -> None:
        ident = PatientIdentity(
            given_name="Jane",
            family_name="Doe",
            date_of_birth=date(1985, 3, 15),
            sex="female",
        )
        out = _identity_from_optional(ident)
        assert out["given_name"] == "Jane"
        assert out["family_name"] == "Doe"
        assert out["date_of_birth"] == "1985-03-15"
        assert out["sex"] == "female"
        # Unset fields project to None, not missing keys.
        assert out["address"] is None
        assert out["phone"] is None
