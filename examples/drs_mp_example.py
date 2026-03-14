#!/usr/bin/env python3
"""Example: Creating and validating DRS-MP inter-AI messages.

DRS-MP (Dual Reader Standard Message Protocol) is the first inter-AI
communication protocol with epistemologically typed claims. Every message
carries structured claims with 5-part falsification predicates — typed,
deterministic, machine-verifiable without natural language processing.

This example demonstrates:
1. Creating a hostile review request with falsifiable claims
2. Creating a structured review response with verdicts and objections
3. Checking review completeness automatically
4. Validating messages against the protocol schema

No external dependencies — stdlib only.

Protocol: https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json
Paper: MK-P8 — The Dual Reader Standard for Inter-AI Communication
License: CC0-1.0 (public domain)
"""

from drs_mp import Claim, Message, Objection, Verdict


def main():
    # --- Step 1: Create a review request ---
    print("=" * 60)
    print("Step 1: Create a hostile review request")
    print("=" * 60)

    request = Message.review_request(
        from_agent="builder-ai",
        to_agent="reviewer-ai",
        subject="Review: Neural compression algorithm claims",
        claims=[
            Claim.falsifiable(
                claim_id="F-NC.1",
                label="Compression Ratio",
                statement="Algorithm achieves 3:1 lossless compression on natural images.",
                falsified_if="Mean compression ratio < 3.0 on ImageNet-1k validation set.",
                where="ImageNet-1k validation (50,000 images). Default algorithm parameters.",
                evaluation="Compress all 50k images. Decompress. Verify bit-exact match. Compute mean ratio.",
                boundary="3.0:1 minimum. Below = falsified. Binary.",
                context="3:1 is the stated performance claim. No margin because lossless is exact.",
            ),
            Claim.falsifiable(
                claim_id="F-NC.2",
                label="Processing Speed",
                statement="Compression runs in real-time (>= 30 fps) on consumer hardware.",
                falsified_if="Mean throughput < 30 frames per second on reference hardware.",
                where="Hardware: any GPU with >= 8GB VRAM. Images: 1920x1080. Batch size: 1.",
                evaluation="Process 1000 images sequentially. Measure wall-clock time. Compute fps.",
                boundary="30 fps minimum. Below = falsified.",
                context="Real-time threshold from video standard (30fps). Consumer GPU defined as >= 8GB.",
            ),
            Claim.axiom(
                claim_id="A-NC.1",
                label="Lossless Definition",
                statement="Lossless compression means the decompressed output is bit-identical to the input.",
            ),
        ],
    )

    errors = request.validate()
    print(f"Message ID: {request.msg_id}")
    print(f"Claims: {len(request.claims)}")
    print(f"Validation: {'PASS' if not errors else errors}")
    print()

    # --- Step 2: Create a review response ---
    print("=" * 60)
    print("Step 2: Create a structured review response")
    print("=" * 60)

    response = Message.review_response(
        original=request,
        from_agent="reviewer-ai",
        verdicts=[
            Verdict(
                claim_id="F-NC.1",
                verdict="confirmed",
                confidence=0.85,
                reasoning="Predicate is well-formed. EVALUATION procedure is deterministic. "
                "BOUNDARY is binary with no ambiguity. ImageNet-1k is a standard benchmark.",
                predicate_assessment={
                    "c6_vacuity": "pass",
                    "deterministic": "pass",
                    "variables_bound": "pass",
                    "third_party_executable": "pass",
                },
                sources_checked=["ImageNet-1k documentation", "RFC 8259 (JSON)"],
            ),
            Verdict(
                claim_id="F-NC.2",
                verdict="needs-revision",
                confidence=0.70,
                reasoning="'Consumer hardware' is underspecified. The WHERE clause says '>= 8GB VRAM' "
                "but doesn't specify GPU architecture. A 2020 GPU with 8GB and a 2025 GPU with 8GB "
                "will produce very different results. Need minimum FLOPS or specific GPU model.",
                predicate_assessment={
                    "c6_vacuity": "pass",
                    "deterministic": "pass",
                    "variables_bound": "fail",  # GPU not fully specified
                    "third_party_executable": "uncertain",
                },
            ),
        ],
        objections=[
            Objection(
                objection_id="OBJ-NC.2-01",
                targets_claim="F-NC.2",
                objection_type="definition-weakness",
                statement="The WHERE clause defines 'consumer hardware' as '>= 8GB VRAM' but this is "
                "insufficient for reproducibility. GPU compute performance varies by 10x within "
                "the same VRAM class. Recommend specifying minimum TFLOPS or a reference GPU model.",
                severity="major",
                proposed_test="Run the benchmark on RTX 3060 (8GB, 12.7 TFLOPS) and RTX 4080 (16GB, "
                "48.7 TFLOPS). If results differ by > 2x, the variable is unbound.",
            ),
        ],
    )

    errors = response.validate()
    print(f"Response ID: {response.msg_id}")
    print(f"Verdicts: {len(response.verdicts)}")
    print(f"Objections: {len(response.objections)}")
    print(f"Validation: {'PASS' if not errors else errors}")
    print()

    # --- Step 3: Check review completeness ---
    print("=" * 60)
    print("Step 3: Check review completeness (F-MK8.2)")
    print("=" * 60)

    # Add the original claims to the response for completeness checking
    response.claims = request.claims
    completeness = response.check_review_completeness()
    print(f"Complete: {completeness['complete']}")
    print(f"Total claims: {completeness['total_claims']}")
    print(f"Reviewed: {completeness['reviewed']}")
    print(f"Missing: {completeness['missing']}")
    print()

    # --- Step 4: Demonstrate the self-evident property ---
    print("=" * 60)
    print("Step 4: The protocol demonstrates itself")
    print("=" * 60)
    print()
    print("If you're reading this code, you just learned DRS-MP.")
    print("You know what a falsification predicate is (5 parts).")
    print("You know what a structured verdict looks like.")
    print("You know what a typed objection is.")
    print()
    print("The code IS the documentation. The example IS the argument.")
    print("No paper required. No journal required. No convincing required.")
    print()
    print("Protocol: https://github.com/thomasbrennan/Fracttalix")
    print("DOI: 10.5281/zenodo.18859299")
    print("License: CC0 (public domain)")


if __name__ == "__main__":
    main()
