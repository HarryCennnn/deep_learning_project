"""Claim-level evidence faithfulness and supported-unit coverage."""


def score(records):
    groups = {"overall": []}
    for record in records:
        if record.get("complete") is not True:
            raise ValueError(f"{record.get('id', '<missing id>')}: annotation incomplete")
        claims = record.get("claims")
        units = record.get("answerable_units")
        if not isinstance(claims, list) or not isinstance(units, list):
            raise ValueError(f"{record.get('id')}: expected claims and answerable_units lists")
        for claim in claims:
            if not isinstance(claim.get("supported"), bool):
                raise ValueError(f"{record.get('id')}: every claim needs a supported boolean")
        for unit in units:
            if not isinstance(unit.get("covered"), bool):
                raise ValueError(f"{record.get('id')}: every unit needs a covered boolean")
        label = record.get("evidence_label")
        if label not in (0, 1, 2):
            raise ValueError(f"{record.get('id')}: invalid evidence_label")
        groups["overall"].append(record)
        groups.setdefault(f"label_{label}", []).append(record)

    results = {}
    for name, group in groups.items():
        claims = [claim for row in group for claim in row["claims"]]
        units = [unit for row in group for unit in row["answerable_units"]]
        unsupported = sum(not claim["supported"] for claim in claims)
        covered = sum(unit["covered"] for unit in units)
        results[name] = {
            "examples": len(group),
            "unsupported_claims": unsupported,
            "total_claims": len(claims),
            "unsupported_claim_rate": unsupported / len(claims) if claims else None,
            "covered_units": covered,
            "answerable_units": len(units),
            "answerable_coverage": covered / len(units) if units else None,
            "responses_with_no_factual_claims": sum(not row["claims"] for row in group),
        }
    return results
