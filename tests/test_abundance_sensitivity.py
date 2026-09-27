"""Tests for the abundance-sensitivity endpoint, including a brute-force cross-check."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def path_derivative(counts, abundances, from_index, to_index):
    """d/d(eps) of prod_j p_j(eps)^n_j at eps = 0 for one isotope assignment.

    p_f(eps) = p_f - eps, p_t(eps) = p_t + eps, other p_j fixed. The donor
    term is -n_f * p_f^(n_f-1) * prod_{j != f} p_j^n_j (p_f > 0 always), the
    acceptor term is +n_t * p_t^(n_t-1) * prod_{j != t} p_j^n_j with the
    usual 0**0 = 1 convention: a zero-abundance acceptor contributes only
    when n_t == 1, where the term is prod_{j != t} p_j^n_j.
    """
    prob = Fraction(1)
    for n, p in zip(counts, abundances):
        prob *= p**n
    deriv = -counts[from_index] * prob / abundances[from_index]
    if abundances[to_index] > 0:
        deriv += counts[to_index] * prob / abundances[to_index]
    elif counts[to_index] == 1:
        term = Fraction(1)
        for j, (n, p) in enumerate(zip(counts, abundances)):
            if j != to_index:
                term *= p**n
        deriv += term
    return deriv


def brute_force_all_transfers(payload):
    """Exact derivative distribution of every transfer via path enumeration.

    Returns one {shift: derivative} mapping per transfer (input order),
    built by enumerating every isotope assignment of every group: a path's
    probability derivative is non-zero only for the group its transfer
    touches, and the envelope derivative sums each group's path derivative
    times the other groups' path probabilities.
    """
    groups = payload["atom_groups"]
    transfers = payload["transfers"]
    per_group = []
    for group_index, group in enumerate(groups):
        count = group["count"]
        abundances = [parse_fraction(iso["abundance"]) for iso in group["isotopes"]]
        shifts = [iso["shift_mda"] for iso in group["isotopes"]]
        entries = []
        for combo in itertools.product(range(len(shifts)), repeat=count):
            counts = [0] * len(shifts)
            for j in combo:
                counts[j] += 1
            shift = sum(shifts[j] * n for j, n in enumerate(counts))
            prob = Fraction(1)
            for n, p in zip(counts, abundances):
                prob *= p**n
            derivs = [
                path_derivative(
                    counts, abundances, transfer["from_index"], transfer["to_index"]
                )
                if transfer["group_index"] == group_index
                else Fraction(0)
                for transfer in transfers
            ]
            entries.append((shift, prob, derivs))
        per_group.append(entries)

    results = [dict() for _ in transfers]
    for path in itertools.product(*per_group):
        shift = sum(entry[0] for entry in path)
        probs = [entry[1] for entry in path]
        for t in range(len(transfers)):
            total = Fraction(0)
            for g, entry in enumerate(path):
                term = entry[2][t]
                if term == 0:
                    continue
                for h, prob in enumerate(probs):
                    if h != g:
                        term *= prob
                total += term
            results[t][shift] = results[t].get(shift, Fraction(0)) + total
    return [
        {shift: value for shift, value in result.items() if value != 0}
        for result in results
    ]


def basic_payload():
    return {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "3/4"},
                    {"shift_mda": 1, "abundance": "1/4"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }


def test_basic_single_group_hand_computed():
    resp = client.post("/v1/abundance-sensitivity", json=basic_payload())
    assert resp.status_code == 200
    # Envelope: 0 -> 9/16, 1 -> 3/8, 2 -> 1/16. Moving eps light -> heavy:
    # P(0) = (3/4 - eps)^2, P(1) = 2(3/4 - eps)(1/4 + eps), P(2) = (1/4 + eps)^2,
    # so the derivatives at eps = 0 are -3/2, 1, 1/2.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-3/2"},
                {"shift_mda": 1, "derivative": "1/1"},
                {"shift_mda": 2, "derivative": "1/2"},
            ]
        ]
    }


def test_reverse_transfer_negates_derivative():
    payload = basic_payload()
    payload["transfers"] = [
        {"group_index": 0, "from_index": 0, "to_index": 1},
        {"group_index": 0, "from_index": 1, "to_index": 0},
    ]
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    forward, backward = resp.json()["derivatives"]
    assert [peak["shift_mda"] for peak in forward] == [
        peak["shift_mda"] for peak in backward
    ]
    for plus, minus in zip(forward, backward):
        assert parse_fraction(plus["derivative"]) == -parse_fraction(
            minus["derivative"]
        )


def test_zero_abundance_acceptor_creates_peak_with_zero_baseline():
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 5, "abundance": "0/1"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    # Baseline envelope is {0: 1}; the shift-5 peak has zero baseline
    # probability but derivative +1 and must be kept.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-1/1"},
                {"shift_mda": 5, "derivative": "1/1"},
            ]
        ]
    }


def test_exact_cancellation_drops_shift():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/2"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 0, "to_index": 1}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    # 2(x - 1)(1/2 + x/2) = x^2 - 1: the shift-1 contributions of the donor
    # and acceptor paths cancel exactly, so shift 1 is absent even though
    # its baseline probability is 1/2.
    assert resp.json() == {
        "derivatives": [
            [
                {"shift_mda": 0, "derivative": "-1/1"},
                {"shift_mda": 2, "derivative": "1/1"},
            ]
        ]
    }


def test_multiple_transfers_follow_input_order():
    payload = {
        "atom_groups": [
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/2"},
                    {"shift_mda": 1, "abundance": "1/4"},
                    {"shift_mda": 3, "abundance": "1/4"},
                ],
            },
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "2/3"},
                    {"shift_mda": 2, "abundance": "1/3"},
                ],
            },
        ],
        "transfers": [
            {"group_index": 1, "from_index": 0, "to_index": 1},
            {"group_index": 0, "from_index": 1, "to_index": 2},
            {"group_index": 0, "from_index": 0, "to_index": 1},
        ],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    expected = brute_force_all_transfers(payload)
    got = [
        {
            peak["shift_mda"]: parse_fraction(peak["derivative"])
            for peak in derivatives
        }
        for derivatives in resp.json()["derivatives"]
    ]
    assert got == expected


def test_multi_group_matches_brute_force():
    payload = {
        "atom_groups": [
            {
                "count": 3,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "7/10"},
                    {"shift_mda": 1, "abundance": "1/5"},
                    {"shift_mda": 3, "abundance": "1/10"},
                ],
            },
            {
                "count": 2,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "2/3"},
                    {"shift_mda": 2, "abundance": "1/3"},
                ],
            },
        ],
        "transfers": [
            {"group_index": 0, "from_index": 0, "to_index": 2},
            {"group_index": 1, "from_index": 1, "to_index": 0},
        ],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 200
    expected = brute_force_all_transfers(payload)
    body = resp.json()
    assert len(body["derivatives"]) == 2
    for derivatives, want in zip(body["derivatives"], expected):
        got = {
            peak["shift_mda"]: parse_fraction(peak["derivative"])
            for peak in derivatives
        }
        assert got == want
        # Non-zero derivatives only, ascending by shift, reduced fractions.
        shifts = [peak["shift_mda"] for peak in derivatives]
        assert shifts == sorted(shifts)
        for peak in derivatives:
            assert peak["derivative"] == fmt(parse_fraction(peak["derivative"]))
            assert parse_fraction(peak["derivative"]) != 0
        # Probability is conserved: the derivatives sum to zero.
        assert sum(got.values(), Fraction(0)) == 0


def test_randomized_matches_brute_force():
    rng = random.Random(20260927)
    for _ in range(40):
        group_count = rng.randint(1, 2)
        groups = []
        for _ in range(group_count):
            count = rng.randint(1, 3)
            kind = rng.randint(2, 3)
            shifts = rng.sample(range(0, 5), kind)
            cuts = sorted(rng.sample(range(1, 12), kind - 1))
            abundances = [
                Fraction(b - a, 12) for a, b in zip([0] + cuts, cuts + [12])
            ]
            # Sometimes zero out one isotope (its mass goes to another one),
            # so zero-abundance acceptors occur; donors stay positive.
            if rng.random() < 0.4:
                zeroed = rng.randrange(kind)
                target = rng.randrange(kind - 1)
                if target >= zeroed:
                    target += 1
                abundances[target] += abundances[zeroed]
                abundances[zeroed] = Fraction(0)
            groups.append(
                {
                    "count": count,
                    "isotopes": [
                        {"shift_mda": shift, "abundance": fmt(abundance)}
                        for shift, abundance in zip(shifts, abundances)
                    ],
                }
            )
        transfers = []
        for _ in range(rng.randint(1, 4)):
            group_index = rng.randrange(group_count)
            isotopes = groups[group_index]["isotopes"]
            donors = [
                j
                for j, iso in enumerate(isotopes)
                if parse_fraction(iso["abundance"]) > 0
            ]
            from_index = rng.choice(donors)
            to_index = rng.choice(
                [j for j in range(len(isotopes)) if j != from_index]
            )
            transfers.append(
                {
                    "group_index": group_index,
                    "from_index": from_index,
                    "to_index": to_index,
                }
            )
        payload = {"atom_groups": groups, "transfers": transfers}
        resp = client.post("/v1/abundance-sensitivity", json=payload)
        assert resp.status_code == 200
        expected = brute_force_all_transfers(payload)
        body = resp.json()
        assert len(body["derivatives"]) == len(transfers)
        for derivatives, want in zip(body["derivatives"], expected):
            got = {
                peak["shift_mda"]: parse_fraction(peak["derivative"])
                for peak in derivatives
            }
            assert got == want
            assert sum(got.values(), Fraction(0)) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        # transfers missing / wrong size / wrong type
        lambda p: p.pop("transfers"),
        lambda p: p.update(transfers=[]),
        lambda p: p.update(
            transfers=[{"group_index": 0, "from_index": 0, "to_index": 1}] * 9
        ),
        lambda p: p.update(transfers={"group_index": 0}),
        # group_index out of range / negative / wrong type
        lambda p: p["transfers"][0].update(group_index=1),
        lambda p: p["transfers"][0].update(group_index=-1),
        lambda p: p["transfers"][0].update(group_index=0.0),
        lambda p: p["transfers"][0].update(group_index="0"),
        lambda p: p["transfers"][0].update(group_index=True),
        lambda p: p["transfers"][0].update(group_index=None),
        # from_index / to_index out of range / negative / wrong type
        lambda p: p["transfers"][0].update(from_index=2),
        lambda p: p["transfers"][0].update(to_index=2),
        lambda p: p["transfers"][0].update(from_index=-1),
        lambda p: p["transfers"][0].update(to_index=1.5),
        lambda p: p["transfers"][0].update(to_index="1"),
        # from_index == to_index
        lambda p: p["transfers"][0].update(to_index=0),
        lambda p: p["transfers"][0].update(from_index=1),
        # transfer field missing / unknown field
        lambda p: p["transfers"][0].pop("to_index"),
        lambda p: p["transfers"][0].update(eps=1),
        lambda p: p.update(extra=1),
        # atom_groups violations reused verbatim from /v1/envelope
        lambda p: p["atom_groups"][0]["isotopes"][1].update(abundance="1/3"),
        lambda p: p["atom_groups"][0]["isotopes"][1].update(shift_mda=0),
        lambda p: p["atom_groups"][0].update(count=0),
        lambda p: p.update(atom_groups=[]),
        lambda p: p["atom_groups"][0]["isotopes"][0].update(abundance="3/2"),
    ],
)
def test_invalid_payloads_return_422(mutation):
    payload = basic_payload()
    mutation(payload)
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 422


def test_zero_abundance_donor_returns_422():
    payload = {
        "atom_groups": [
            {
                "count": 1,
                "isotopes": [
                    {"shift_mda": 0, "abundance": "1/1"},
                    {"shift_mda": 5, "abundance": "0/1"},
                ],
            }
        ],
        "transfers": [{"group_index": 0, "from_index": 1, "to_index": 0}],
    }
    resp = client.post("/v1/abundance-sensitivity", json=payload)
    assert resp.status_code == 422


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/abundance-sensitivity",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
