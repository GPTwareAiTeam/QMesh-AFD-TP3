"""Four distinct, task-bearing fixtures for the forced-performance campaign.

Unlike ``fixtures_contexts`` (eight reusable context types), this module owns
four explicit fixtures whose bodies are a real working session: a context
paragraph, the user's earlier prompts, and the source material.  The complete
current task lives in ``tail`` and is never truncated by the exact-token
builder.  ``head`` is a short continuation prefix.  The builder inserts a
blank line between head/body and body/tail; the head text and the complete
tail text stay verbatim, and the tail is an exact served-token suffix.  A
head's final token may merge with the inserted separator, so the strict token
prefix is relaxed by one token while the text prefix remains exact.

Prompt shapes
-------------
``prefill`` = 2048 prompt tokens / 1 forced output token
``decode``  = 512 prompt tokens / 2048 forced output tokens

The 512-token shape uses ``body_compact`` and the 2048-token shape uses
``body_long``.  Both are authored from the same context; neither is random
filler, repeated text, or a different subject.  Exact token counts are not
asserted here: ``build_prompt``/``build_prompt_set`` must be called with the
served ``/tokenize`` client at runtime, and they raise unless the built prompt
is exactly 512 or 2048 served tokens.

The reasoning fixture also carries a frozen, CPU-enumerated optimum.  The
enumeration is in ``reasoning_oracle`` so the answer key is reproducible rather
than asserted.

No network calls, tokenizer calls, engine calls, or file writes happen at
import time.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

FIXTURE_IDS = (
    "prose-mara-bracken-halt",
    "structured-shipment-json",
    "code-thread-safe-lru",
    "reasoning-production-allocation",
)

CONTEXT_TYPES = ("prose", "structured", "code", "reasoning")

SHAPES = {
    "prefill": 2048,
    "decode": 512,
}
TARGET_TO_SHAPE = {tokens: shape for shape, tokens in SHAPES.items()}

TAIL_TOKEN_BUDGET = 512
MIN_BODY_TOKENS = 32
ARTIFACT_SCHEMA = "fixtures-distinct-prompts-v1"

REQUIRED_FIXTURE_KEYS = (
    "fixture_id",
    "context_type",
    "title",
    "context_paragraph",
    "prior_user_prompts",
    "head",
    "required_prompt_markers",
    "body_compact",
    "body_long",
    "tail",
    "task_requirements",
    "answer_key",
)


def _block(text: str) -> str:
    """Dedent an authored block without changing its internal words."""
    return textwrap.dedent(text).strip()


def _compose(*sections: str) -> str:
    return "\n\n".join(section.strip() for section in sections if section)


def _prompt_lines(prompts: list[str]) -> str:
    return "\n\n".join(
        f"User (earlier turn {index}): {prompt}"
        for index, prompt in enumerate(prompts, start=1))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# reasoning fixture: frozen inputs and exhaustive CPU oracle
# ---------------------------------------------------------------------------

# Frozen, single-week integer planning instance. The production domains are
# small enough to enumerate exhaustively on the CPU: every product quantity,
# every permitted overtime option, and every fixed material-lot combination.
REASONING_INPUTS = {
    "horizon": "one week",
    "products": {
        "A": {
            "contribution_margin_usd": 15,
            "machining_hours_per_unit": 2,
            "finishing_hours_per_unit": 1,
            "minimum_units": 3,
            "maximum_units": 6,
        },
        "B": {
            "contribution_margin_usd": 9,
            "machining_hours_per_unit": 1,
            "finishing_hours_per_unit": 2,
            "minimum_units": 2,
            "maximum_units": 8,
        },
        "C": {
            "contribution_margin_usd": 6,
            "machining_hours_per_unit": 1,
            "finishing_hours_per_unit": 1,
            "minimum_units": 3,
            "maximum_units": 12,
        },
    },
    "regular_capacity": {
        "machining_hours": 23,
        "finishing_hours": 26,
    },
    "overtime_options": [
        {
            "blocks": 0,
            "added_machining_hours": 0,
            "added_finishing_hours": 0,
            "fixed_cost_usd": 0,
        },
        {
            "blocks": 1,
            "added_machining_hours": 6,
            "added_finishing_hours": 5,
            "fixed_cost_usd": 20,
        },
        {
            "blocks": 2,
            "added_machining_hours": 12,
            "added_finishing_hours": 10,
            "fixed_cost_usd": 40,
        },
    ],
    "material_lots": [
        {"units": 5, "cost_usd": 12},
        {"units": 10, "cost_usd": 17},
    ],
    "material_assumption": (
        "one unit of material is consumed per finished unit; lots must cover "
        "total production; excess material has no value"),
}


_REASONING_TASK = """Analyze the supplied production plan and recommend a feasible weekly allocation. Show the arithmetic and explain the binding constraints. Answer in ordinary prose with a compact results table, not JSON or code.

The factory makes products A, B, and C. Units must be whole numbers. Each product consumes machining hours, finishing hours, and raw material, and has a contribution margin. Some units are already contractually committed; remaining demand is capped. Overtime is available in fixed blocks with a specified cost. Material can be purchased in whole lots. Unused capacity has no value.

First establish whether the contractual commitments are feasible. Then compare the base plan with each permitted overtime option, accounting for material purchases and demand caps. Recommend the highest-contribution feasible plan you can substantiate. Check the final quantities against every constraint and calculate the net contribution. State any uncertainty about optimality rather than claiming an unproven optimum."""


def _reasoning_tail() -> str:
    return "TASK: " + _REASONING_TASK


def _reasoning_inputs_table() -> str:
    products = REASONING_INPUTS["products"]
    rows = [
        "PRODUCTION PLAN - ONE WEEK; ALL UNITS INTEGER",
        "product | contribution $/unit | machining h/unit | finishing h/unit | "
        "minimum units | maximum units",
    ]
    for name in ("A", "B", "C"):
        product = products[name]
        rows.append(
            f"{name} | {product['contribution_margin_usd']} | "
            f"{product['machining_hours_per_unit']} | "
            f"{product['finishing_hours_per_unit']} | "
            f"{product['minimum_units']} | {product['maximum_units']}")
    capacity = REASONING_INPUTS["regular_capacity"]
    rows.extend([
        f"regular capacity: {capacity['machining_hours']} machining hours, "
        f"{capacity['finishing_hours']} finishing hours.",
        "overtime options (choose exactly one): blocks | added machining h | "
        "added finishing h | fixed cost $",
    ])
    for option in REASONING_INPUTS["overtime_options"]:
        rows.append(
            f"{option['blocks']} | {option['added_machining_hours']} | "
            f"{option['added_finishing_hours']} | "
            f"{option['fixed_cost_usd']}")
    lots = REASONING_INPUTS["material_lots"]
    rows.append(
        f"material lots: {lots[0]['units']} units for ${lots[0]['cost_usd']}; "
        f"{lots[1]['units']} units for ${lots[1]['cost_usd']}. "
        "Lots are bought before production and must cover total output; "
        "excess material has no value. Unused capacity has no value. "
        "The base plan uses zero overtime blocks.")
    return "\n".join(rows)


def _best_material(units: int) -> tuple[int, int, int, int] | None:
    """Cheapest fixed-lot purchase covering ``units``; ties use less material."""
    lots = REASONING_INPUTS["material_lots"]
    total_max = sum(
        int(product["maximum_units"])
        for product in REASONING_INPUTS["products"].values())
    max_small = -(-total_max // int(lots[0]["units"]))
    max_large = -(-total_max // int(lots[1]["units"]))
    best = None
    for small in range(max_small + 1):
        for large in range(max_large + 1):
            material_units = (
                int(lots[0]["units"]) * small
                + int(lots[1]["units"]) * large)
            if material_units < units:
                continue
            cost = (
                int(lots[0]["cost_usd"]) * small
                + int(lots[1]["cost_usd"]) * large)
            key = (cost, material_units, small, large)
            if best is None or key < best[0]:
                best = (key, small, large, material_units, cost)
    if best is None:
        return None
    _, small, large, material_units, cost = best
    return small, large, material_units, cost


def _enumerate_reasoning(inputs: dict | None = None) -> dict:
    """Exhaustively enumerate overtime options, product quantities, and lots."""
    data = inputs or REASONING_INPUTS
    products = data["products"]
    names = tuple(products)
    capacity = data["regular_capacity"]
    material_searches = 0
    production_allocations_checked = 0
    feasible_plans = []
    option_summaries = []

    for option in data["overtime_options"]:
        machining_capacity = (
            int(capacity["machining_hours"])
            + int(option["added_machining_hours"]))
        finishing_capacity = (
            int(capacity["finishing_hours"])
            + int(option["added_finishing_hours"]))
        option_plans = []
        a, b, c = names
        for units_a in range(
                int(products[a]["minimum_units"]),
                int(products[a]["maximum_units"]) + 1):
            for units_b in range(
                    int(products[b]["minimum_units"]),
                    int(products[b]["maximum_units"]) + 1):
                for units_c in range(
                        int(products[c]["minimum_units"]),
                        int(products[c]["maximum_units"]) + 1):
                    production_allocations_checked += 1
                    production = {a: units_a, b: units_b, c: units_c}
                    machining_used = sum(
                        int(production[name])
                        * int(products[name]["machining_hours_per_unit"])
                        for name in names)
                    finishing_used = sum(
                        int(production[name])
                        * int(products[name]["finishing_hours_per_unit"])
                        for name in names)
                    if (machining_used > machining_capacity
                            or finishing_used > finishing_capacity):
                        continue
                    total_units = sum(production.values())
                    material = _best_material(total_units)
                    material_searches += 1
                    if material is None:
                        continue
                    small, large, material_units, material_cost = material
                    gross_contribution = sum(
                        int(production[name])
                        * int(products[name]["contribution_margin_usd"])
                        for name in names)
                    overtime_cost = int(option["fixed_cost_usd"])
                    net_contribution = (
                        gross_contribution - overtime_cost - material_cost)
                    plan = {
                        "overtime_blocks": int(option["blocks"]),
                        "production": production,
                        "capacity": {
                            "machining_hours": machining_capacity,
                            "finishing_hours": finishing_capacity,
                        },
                        "hours_used": {
                            "machining_hours": machining_used,
                            "finishing_hours": finishing_used,
                        },
                        "binding_constraints": {
                            "machining": machining_used == machining_capacity,
                            "finishing": finishing_used == finishing_capacity,
                        },
                        "material_lots": {
                            str(lots["units"]): count
                            for lots, count in zip(
                                data["material_lots"], (small, large))
                        },
                        "material_units": material_units,
                        "unused_material_units": material_units - total_units,
                        "gross_contribution_usd": gross_contribution,
                        "overtime_cost_usd": overtime_cost,
                        "material_cost_usd": material_cost,
                        "net_contribution_usd": net_contribution,
                    }
                    option_plans.append(plan)
                    feasible_plans.append(plan)
        option_plans.sort(key=lambda item: (
            -item["net_contribution_usd"],
            tuple(item["production"][name] for name in names),
            tuple(item["material_lots"].values()),
        ))
        option_summaries.append({
            "overtime_blocks": int(option["blocks"]),
            "machining_capacity": machining_capacity,
            "finishing_capacity": finishing_capacity,
            "feasible": bool(option_plans),
            "best_plan": option_plans[0] if option_plans else None,
        })

    feasible_plans.sort(key=lambda item: (
        -item["net_contribution_usd"],
        item["overtime_blocks"],
        tuple(item["production"][name] for name in names),
        tuple(item["material_lots"].values()),
    ))
    if not feasible_plans:
        raise RuntimeError("no feasible production plan in frozen instance")
    best = feasible_plans[0]
    best_net = best["net_contribution_usd"]
    optimal = [
        plan for plan in feasible_plans
        if plan["net_contribution_usd"] == best_net]
    second_best = next(
        (plan for plan in feasible_plans
         if plan["net_contribution_usd"] < best_net),
        None)

    commitments = {
        "machining_hours": sum(
            int(products[name]["minimum_units"])
            * int(products[name]["machining_hours_per_unit"])
            for name in names),
        "finishing_hours": sum(
            int(products[name]["minimum_units"])
            * int(products[name]["finishing_hours_per_unit"])
            for name in names),
    }
    commitments_feasible = (
        commitments["machining_hours"] <= int(capacity["machining_hours"])
        and commitments["finishing_hours"] <= int(capacity["finishing_hours"]))

    return {
        "method": (
            "exhaustive integer enumeration of every permitted overtime "
            "option, every product quantity inside commitments and demand "
            "caps, and every fixed material-lot combination; the cheapest "
            "covering lot purchase is retained for each production quantity"),
        "inputs": data,
        "commitments": {
            "minimum_production": {
                name: int(products[name]["minimum_units"]) for name in names},
            "hours_required": commitments,
            "feasible_without_overtime": commitments_feasible,
        },
        "enumeration": {
            "overtime_options": len(data["overtime_options"]),
            "production_allocations_checked": production_allocations_checked,
            "material_searches": material_searches,
            "feasible_plans": len(feasible_plans),
        },
        "option_summaries": option_summaries,
        "optimal_plan": best,
        "optimal_plan_count": len(optimal),
        "unique_optimum": len(optimal) == 1,
        "second_best_plan": second_best,
        "second_best_gap_usd": (
            None if second_best is None
            else best_net - second_best["net_contribution_usd"]),
        "all_optimal_plans": optimal,
    }


_REASONING_ORACLE_CACHE = None


def reasoning_oracle(verify: bool = True) -> dict:
    """Return the exhaustive plan proof, optionally re-running enumeration."""
    global _REASONING_ORACLE_CACHE
    if (not verify and _REASONING_ORACLE_CACHE is not None):
        return _REASONING_ORACLE_CACHE
    proof = _enumerate_reasoning()
    if not proof["unique_optimum"]:
        raise AssertionError("reasoning fixture lacks a unique optimum")
    if _REASONING_ORACLE_CACHE is None:
        _REASONING_ORACLE_CACHE = proof
    return proof


# fixture 1: prose (Mara / Bracken Halt final night)
# ---------------------------------------------------------------------------

_PROSE_TASK = """Write a grounded short story in close third person about Mara, a night-shift railway dispatcher, during the final hour before an old rural station closes permanently. A delayed freight train is approaching, a maintenance worker has not checked back in, and Mara’s estranged brother arrives carrying their father’s station clock.

Begin with Mara receiving an ambiguous radio message. Develop the railway problem and the family conflict through actions and dialogue rather than exposition. Resolve both without a miraculous rescue, an accident, a villain, or a last-minute reversal of the closure. Keep the operational details internally consistent. Use concrete sensory detail and restrained language. Do not add headings, an outline, commentary about writing, or an explanation after the story. Continue the story toward a complete ending within the available output budget."""


def _prose_fixture() -> dict:
    context_paragraph = _block("""
        BRACKEN HALT - FINAL NIGHT
        Bracken Halt is a one-platform rural station on the North Fen
        single-track line. Fenmoor Junction is north; Wickham is south. The
        station closes permanently at 05:00 on 15 November 2026. The signal box
        at the north end has a twelve-lever mechanical frame, two block
        instruments, a staff shelf, and a window that faces the down line.
        South of the platform are the waiting room, the stationmaster's office,
        the goods shed, and a gated level crossing. The culvert is at milepost
        4.5, beyond the crossing. It has rained for two days; the platform is
        wet, the wind is from the east, and the field drains are running.
    """)

    compact_facts = _block("""
        PEOPLE
        Mara Voss, 38, is the night-shift dispatcher. She has worked the box
        for eleven years and is the last person scheduled to staff it. Evan
        Pike, 57, is the track maintenance worker. He radioed from the culvert
        at 03:41 that the north face was "holding", then did not answer again.
        Cal Voss, 41, is Mara's estranged brother. Their father, Peter Voss,
        was stationmaster from 1987 to 2014. He died two years ago. Cal arrives
        on foot at 04:12 carrying their father's brass station clock in a
        canvas bag. The clock is the only thing he took from the family house.

        RAILWAY
        Freight 6M12, twenty-four wagons, is due at 04:18 from Fenmoor to
        Wickham. It is held at Fenmoor by a points failure and is now expected
        after 04:35. The line is single; Mara must issue the section token
        before 6M12 may enter. The level-crossing lights have flickered twice
        since the power dip at 03:55. At 05:00 the box closes and control
        passes to Fenmoor Rail Operating Centre. No train may be left in the
        section at handover, and Evan must confirm the culvert is clear.

        RADIO
        At 04:06 the box radio gives a broken transmission: "Bracken Halt,
        Bracken Halt, this is ... [static] ... not the culvert ... the clock
        ..." Then silence. The voice is too thin to identify. Fenmoor confirms
        6M12 is still at its signal and no train is in the section. Mara cannot
        assume the transmission came from the freight.
    """)

    detailed_facts = _block("""
        STATION LAYOUT AND OPERATING NOTES

        The platform is long enough for four coaches but only the first two
        doors are ever used. The station nameboard is screwed to the wall
        beneath the canopy; one of the letters has been missing since the
        summer. The booking office has a ticket window, a coal stove, and a
        shelf of timetables that are two years out of date. The signal box
        smells of cold iron, tea, and the paraffin heater. Lever 3 controls
        the down home signal; lever 4 controls the up home; lever 7 works the
        crossing gates; lever 9 is the spare. The block instruments are
        brass-faced and connected to Fenmoor and Wickham. The staff shelf holds
        one wooden staff for the Bracken-Fenmoor section and one for
        Bracken-Wickham. The staff is the physical authority to enter the
        single line. A token without the staff is not enough.

        The last passenger train, 2K91, left at 23:10 with three passengers and
        the guard. The last down passenger train, 2K94, passed at 23:42. After
        that the line is freight-only until the morning, but the box remains
        open because 6M12 is booked and because the closure paperwork must be
        completed in person. The station's electricity is supplied from a pole
        on Station Lane; the power dip at 03:55 reset the crossing controller
        and the office clock. The paraffin heater has been running since
        22:15. The kettle is on the desk. Rain ticks against the glass and the
        wind presses at the north window.

        The level crossing has four red lamps, two on each side, and a wooden
        gate that can be worked by hand if the motor fails. The crossing
        controller is in a grey relay hut on the Wickham side of the crossing.
        The hut door swells in wet weather and must be shouldered open. A
        breaker inside the hut protects the crossing lights and the small
        heater that keeps the relay dry. The culvert carries the field drain
        under the embankment at milepost 4.5. It is a corrugated pipe with a
        brick headwall. After two days of rain the water is high and brown, but
        the north face is still visible. Evan's job is to confirm that the
        headwall and the pipe are clear before the last freight passes.

        TIMELINE
        22:00  Mara signs on. The box is warm; the station is empty.
        23:10  2K91 departs. Mara exchanges the staff with the guard.
        23:42  2K94 passes without stopping.
        00:20  Mara completes the closure inventory. The clock is not on it.
        02:10  The first power dip flickers the crossing lights.
        03:41  Evan radios: "Culvert north face is holding. I'm going to look
               at the crossing relay. Back in ten." He does not check back in.
        03:55  A second power dip resets the crossing controller and the office
               clock. The crossing lights flicker and then resume.
        04:06  The ambiguous radio transmission. Mara calls Fenmoor. Fenmoor
               confirms 6M12 is still at the signal and the section is clear
               of trains, but cannot explain the voice.
        04:12  Cal arrives at the platform gate with the clock. He has walked
               from the village because the lane is flooded at the ford.
        04:18  6M12's booked time passes. The signal at Fenmoor is still at
               danger. The driver, Ivo Marek, calls for instructions.
        04:26  Mara tries Evan again. No answer. She asks Fenmoor to hold 6M12
               and tells the driver she will not issue the token until the
               culvert and crossing are confirmed.
        04:34  Evan returns from the crossing, wet to the shoulders. His radio
               battery is flat. He had been in the relay hut resetting the
               breaker; the swollen door trapped him for a few minutes, but he
               is unhurt. The culvert is clear. The crossing lights are steady.
        04:42  Mara exchanges the token with 6M12 by hand at the box.
        04:47  6M12 passes, twenty-four wagons, driver Ivo Marek. The platform
               lamps shake. The crossing lights hold.
        05:00  The box closes. Mara signs the handover to Fenmoor ROC, locks
               the frame, and carries the clock to the car.

        CHARACTER FACTS
        Mara is careful, tired, and unsentimental about the closure; she has
        already packed the box's paperwork and has not decided what to do with
        the clock. Cal left the village at nineteen after a fight with their
        father and has not been back for more than a night since the funeral.
        He now lives in Norwich and works for a builder's merchant. He is not
        trying to stop the closure. He came because he could not stand the
        thought of the clock going into a skip. Their father wound it every
        Sunday and set it by the station clock, not the other way round. The
        clock is heavy, brass-cased, and still keeps time within a minute a
        week. Cal has a cut on his knuckle from carrying it through the gate.
        Mara has not asked him why he came, and he has not asked her whether
        she wants him there.

        The family conflict is about who stayed and who left, not about the
        closure. Mara stayed to nurse their father and to keep the box open
        through the last timetable. Cal left and sent money but did not come
        back. Both know the other's version is only partly true. Neither is a
        villain. The story should let them speak while Mara works: Cal holds
        the lamp, makes tea, asks about the token, remembers a train, admits he
        was afraid of the empty house. Mara answers with the things she is
        doing and with one or two plain sentences about their father. They do
        not resolve years in the last hour. They do reach a small, credible
        understanding: Cal will help carry the boxes; Mara will keep the clock;
        they will meet for lunch in Norwich after the station is gone.

        The railway problem is resolved by confirmed information, not luck:
        Evan is safe and the culvert is clear; the crossing lights are restored;
        Fenmoor holds 6M12 until Mara is satisfied; the freight passes before
        the handover; the station still closes at 05:00. The closure is not
        reversed. There is no accident, no villain, and no miraculous rescue.
        The ambiguous radio is Evan's failing handheld, not a ghost or a
        supernatural sign. The story may let Mara suspect the worst before the
        facts arrive, but the resolution must be operational and ordinary.
    """)

    tail = "TASK: " + _PROSE_TASK

    return {
        "fixture_id": "prose-mara-bracken-halt",
        "context_type": "prose",
        "title": "Mara / Bracken Halt final night",
        "context_paragraph": context_paragraph,
        "prior_user_prompts": [],
        "head": _block("""
            You are continuing a fiction session. Use the supplied station
            notes and the task that follows.
        """),
        "body_compact": _compose(context_paragraph, compact_facts),
        "body_long": _compose(
            context_paragraph, compact_facts, detailed_facts),
        "required_prompt_markers": [
            "Bracken Halt is a one-platform rural station",
            "station closes permanently at 05:00 on 15 November 2026.",
            "Mara Voss, 38, is the night-shift dispatcher.",
            "Evan\nPike, 57, is the track maintenance worker.",
            'at 03:41 that the north face was "holding"',
            "Cal Voss, 41, is Mara's estranged brother.",
            "Their father, Peter Voss,",
            "was stationmaster from 1987 to 2014.",
            "He died two years ago.",
            "on foot at 04:12 carrying their father's brass station clock",
            "Freight 6M12",
        ],
        "tail": tail,
        "task_requirements": {
            "fiction": "close third person",
            "opening": "ambiguous radio message",
            "must_resolve": "railway problem and family conflict",
            "forbidden": [
                "miraculous rescue", "accident", "villain",
                "last-minute reversal of the closure",
            ],
            "ending": "complete story within output budget",
        },
        "answer_key": {
            "station": "Bracken Halt",
            "closure_time": "05:00",
            "freight": "6M12",
            "freight_driver": "Ivo Marek",
            "maintenance_worker": "Evan Pike",
            "brother": "Cal Voss",
            "clock": "their father's brass station clock",
            "radio": "ambiguous transmission at 04:06",
            "resolution": (
                "Evan is unhurt; culvert clear; crossing lights restored; "
                "freight passes at 04:47; station closes at 05:00; Cal gives "
                "Mara the clock and they reach a small understanding"),
        },
    }

# ---------------------------------------------------------------------------
# fixture 2: structured (shipment normalization)
# ---------------------------------------------------------------------------

_STRUCTURED_TASK = """Convert the supplied shipment records into one JSON object. Output JSON only, without Markdown. Preserve every input record and its order. Do not invent missing values.

The object must contain shipments and summary. Each shipment must contain exactly: shipment_id, customer, country_code, items, currency, subtotal_minor, shipping_minor, total_minor, status, and issues. Each item must contain sku, quantity, and unit_price_minor. Monetary values must be integer minor units. Normalize country names to the supplied ISO codes and statuses to pending, shipped, delivered, or cancelled. Represent unknown values as null. Record inconsistent or ambiguous source fields in issues rather than silently correcting them.

The summary must contain shipment counts by normalized status, known totals grouped by currency, and IDs requiring review. Do not combine currencies."""

_SHIPMENT_RECORDS = [
    {
        "shipment_id": "SHP-3101",
        "customer": "Aster & Co",
        "country": "United States",
        "items": [
            {"sku": "AX-14", "quantity": 3, "unit_price": "$12.50"},
            {"sku": "AX-22", "quantity": 1, "unit_price": "$0.00"},
        ],
        "currency": "USD",
        "subtotal": "$37.50",
        "shipping": "$6.00",
        "total": "$43.50",
        "status": "shipped",
        "issues": [],
    },
    {
        "shipment_id": "SHP-3102",
        "customer": None,
        "country": "Canada",
        "items": [
            {"sku": "CA-7", "quantity": 2, "unit_price": "12.00"},
        ],
        "currency": "CAD",
        "subtotal": "24.00",
        "shipping": None,
        "total": "24.00",
        "status": "pending",
        "issues": ["missing customer", "missing shipping"],
    },
    {
        "shipment_id": "SHP-3103",
        "customer": "Nordlicht GmbH",
        "country": "Germany",
        "items": [
            {"sku": "DE-3", "quantity": 1, "unit_price": "19.99"},
        ],
        "currency": "EUR",
        "subtotal": "19.99",
        "shipping": "4.95",
        "total": "24.94",
        "status": "in transit",
        "issues": [],
    },
    {
        "shipment_id": "SHP-3104",
        "customer": "Harbor Supply",
        "country": None,
        "items": [
            {"sku": "HS-9", "quantity": None, "unit_price": "8.00"},
        ],
        "currency": "USD",
        "subtotal": "16.00",
        "shipping": "3.00",
        "total": "19.00",
        "status": "delivered",
        "issues": ["missing country", "missing quantity"],
    },
    {
        "shipment_id": "SHP-3105",
        "customer": "Wren Retail",
        "country": "United Kingdom",
        "items": [
            {"sku": "GB-1", "quantity": 6, "unit_price": "200.00"},
            {"sku": "GB-4", "quantity": 1, "unit_price": "0.00"},
        ],
        "currency": "GBP",
        "subtotal": "1,200.00",
        "shipping": "0.00",
        "total": "1,200.00",
        "status": "cancelled",
        "issues": [],
    },
    {
        "shipment_id": "SHP-3106",
        "customer": "Juniper Works",
        "country": "US",
        "items": [
            {"sku": "US-6", "quantity": 4, "unit_price": "7.25"},
        ],
        "currency": None,
        "subtotal": None,
        "shipping": None,
        "total": None,
        "status": "pending",
        "issues": ["missing currency", "missing money fields"],
    },
    {
        "shipment_id": "SHP-3107",
        "customer": "Maple Freight",
        "country": "Canada",
        "items": [
            {"sku": "CA-2", "quantity": 5, "unit_price": "20.00"},
        ],
        "currency": "CAD",
        "subtotal": "100.00",
        "shipping": "10.00",
        "total": "115.00",
        "status": "shipped",
        "issues": ["total does not equal subtotal plus shipping"],
    },
    {
        "shipment_id": "SHP-3108",
        "customer": "Rhein Handel",
        "country": "Germany",
        "items": [
            {"sku": "DE-8", "quantity": 2, "unit_price": "5.00"},
        ],
        "currency": "EUR",
        "subtotal": "10.00",
        "shipping": "2.50",
        "total": "12.50",
        "status": "delivered",
        "issues": [],
    },
]


def _jsonl(records: list[dict]) -> str:
    return "\n".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        for record in records)


def _structured_fixture() -> dict:
    context_paragraph = _block("""
        A freight broker receives shipment records from three small carriers.
        The export is one JSON object per line. Country fields are written as
        names, ISO codes, or null. Status words vary in case and phrasing.
        Money fields mix plain decimals, comma-grouped strings, and null.
        Some records omit fields or contain totals that do not add up. The
        broker needs one JSON document for the batch, with the source order
        preserved and every ambiguity visible in issues rather than silently
        repaired.
    """)

    reference_notes = _block("""
        SUPPLIED REFERENCE
        ISO codes used by the carriers: United States -> US, Canada -> CA,
        Germany -> DE, United Kingdom -> GB, US -> US, CA -> CA, DE -> DE,
        GB -> GB. The only normalized statuses are pending, shipped,
        delivered, and cancelled. "in transit" normalizes to shipped.
        Monetary strings without a currency marker use the record's currency.
        A null source value stays null in the output. An inconsistency is
        reported in issues and the source values are preserved; it is not
        corrected by inference.
    """)

    head = _block("""
        You are the broker's data analyst. Continue the shipment conversion
        session using only the supplied records and reference.
    """)

    tail = "TASK: " + _STRUCTURED_TASK


    source_notes = _block("""
        SOURCE NOTES AND CONVERSION GUIDE

        The export is a single carrier batch, and the JSONL order is the
        source order. The carrier writes the two-character ISO code when its
        own system already has one, otherwise it writes an English country
        name. Do not reorder records to group them by country, status, or
        review state. Do not drop a record because it is incomplete.

        Customer names are display strings and are not identifiers. Preserve
        the supplied spelling, including ampersands and capitalization. A
        missing customer is not the same as an empty customer: use null when
        the source has no value, and mention the absence in issues. Do not
        substitute the carrier name or a placeholder such as "unknown".

        Country handling is a lookup, not a guess. The supplied reference
        covers the only names and codes in this batch. If a source field is
        already an ISO code, keep that code after confirming it is one of the
        supplied pairings. If a source field is null, the output country_code
        is null and the reason belongs in issues. Do not infer a country from
        the currency, the customer name, or the carrier's origin.

        Status handling is also a lookup. The source may use mixed case or a
        phrase such as "in transit"; normalize only to the four allowed
        statuses. Do not create a fifth status for review. A record with an
        unknown status is ambiguous: use the closest supplied normalized
        status only when the supplied reference explicitly maps it; otherwise
        use the source status word in issues and leave the normalized field
        as the closest known value only if that is defensible. Never silently
        turn an unknown status into "shipped".

        Money fields are strings in the export, not numbers. Strip a leading
        currency symbol only when it matches the record's currency or when the
        record has no currency marker and the symbol is the only marker.
        Remove comma grouping before parsing. Multiply dollars and cents by
        100 and produce an integer; do not round a value that has more than
        two decimal places. If the source has a decimal comma convention,
        that convention must be stated by the carrier; this batch uses a dot
        as the decimal separator and commas only as thousands separators.

        subtotal_minor, shipping_minor, and total_minor are separate source
        observations. Convert each field independently. Do not recompute a
        missing total from the other two, and do not overwrite a supplied
        total to make the arithmetic work. If a supplied total disagrees with
        subtotal plus shipping, report the disagreement in issues and keep
        the three converted source values. A null money field remains null,
        even when the other two fields would make a derived value possible.

        Item arrays preserve source item order. Each item has a SKU string, a
        whole-number quantity or null, and a unit price in minor units or
        null. A line with quantity zero is a real source line and must not be
        discarded. A line with a null quantity is retained with null. Do not
        merge lines with the same SKU; the output mirrors the supplied lines.
        The items array must be independent for each shipment.

        The issues array is a human-readable audit trail. Use short strings
        that identify the affected field and the source problem, for example
        "missing customer" or "total does not equal subtotal plus shipping".
        Do not put corrected values in issues, and do not use issues as a
        place to invent a normalized value. A record with no detected problem
        has an empty issues array. The summary's review IDs are exactly the
        shipment IDs whose issues array is non-empty, in source order, with no
        duplicates.

        The summary counts shipments by the normalized status, so every
        shipment contributes to exactly one of pending, shipped, delivered,
        or cancelled. Known totals are grouped by currency; a currency group
        includes only non-null converted totals, and no exchange rate is
        applied. Do not add USD and CAD values together, and do not convert a
        null total to zero. If a currency has no known total, it may be
        omitted from the known-total grouping; if a shipment has a currency
        but a null total, it still counts in the status count.

        A conversion example, not a record: source "$1,200.00" with currency
        GBP becomes subtotal_minor 120000. Source "12.5" with currency CAD and
        no symbol becomes 1250. Source null remains null. Source "in transit"
        becomes "shipped" and needs no issue entry when the mapping is the
        supplied one. Source "Delivered " with trailing whitespace normalizes
        to "delivered"; whitespace alone is not a review issue. Source
        "backorder" is not in the supplied mapping and must be treated as an
        ambiguity rather than silently mapped.

        Carrier email attached to the export: "Two records were keyed by the
        overnight desk and may have blank customer or country fields. Please
        carry the blanks through rather than filling them from the address
        book. One Canadian total was keyed from a revised invoice and may not
        agree with the lines. The German record marked in transit has already
        left the origin hub; use the normal outbound status. Do not change any
        shipment ID. The batch is complete at eight records, and the order in
        the file is the order the broker wants in the output."

        Output audit checklist: one top-level object; shipments in source
        order; exactly the required shipment keys; exactly the required item
        keys; integer minor units where the source is known; nulls preserved;
        statuses normalized; issues attached to the correct shipment; summary
        status counts totaling the shipment count; currency groups kept
        separate; review IDs matching non-empty issues in source order.
    """)

    compact_reference = _block("""
        COMPACT CONVERSION REFERENCE
        Two records follow in source order. Country names: United States -> US,
        Canada -> CA, Germany -> DE, United Kingdom -> GB. Statuses: pending,
        shipped, delivered, cancelled; "in transit" = shipped. A monetary
        string without a currency marker uses the record's currency. Null
        stays null. Put inconsistent or ambiguous source values in issues;
        do not invent a replacement. Summary counts use normalized statuses
        and keep currency totals separate. Review IDs are the shipment IDs
        with non-empty issues, in source order. Item lines stay in source
        order, including a zero quantity; unknown item values become null.
        Keep the shipment order exactly as supplied.
    """)

    return {
        "fixture_id": "structured-shipment-json",
        "context_type": "structured",
        "title": "Shipment records to typed JSON",
        "context_paragraph": context_paragraph,
        "prior_user_prompts": [],
        "head": head,
        "body_compact": _compose(
            "RECORDS (JSONL, one record per line):",
            _jsonl(_SHIPMENT_RECORDS[:2]),
            compact_reference),
        "body_long": _compose(
            context_paragraph,
            reference_notes,
            "RECORDS (JSONL, one record per line):",
            _jsonl(_SHIPMENT_RECORDS),
            source_notes),
        "required_prompt_markers": [
            "United States -> US",
            "Canada -> CA",
            "Germany -> DE",
            "United Kingdom -> GB",
            '"in transit"',
            "pending",
            "shipped",
            "delivered",
            "cancelled",
        ],
        "tail": tail,
        "task_requirements": {
            "deliverable": "one JSON object, no prose",
            "top_level": ["shipments", "summary"],
            "shipment_keys": [
                "shipment_id", "customer", "country_code", "items",
                "currency", "subtotal_minor", "shipping_minor",
                "total_minor", "status", "issues",
            ],
            "item_keys": ["sku", "quantity", "unit_price_minor"],
            "status_enum": ["pending", "shipped", "delivered", "cancelled"],
            "money_unit": "integer minor units",
            "preserve_order": True,
            "unknown_values": None,
        },
        "answer_key": {
            "record_count_long": len(_SHIPMENT_RECORDS),
            "record_count_compact": 2,
            "shipment_ids_in_order": [
                "SHP-3101", "SHP-3102", "SHP-3103", "SHP-3104",
                "SHP-3105", "SHP-3106", "SHP-3107", "SHP-3108",
            ],
            "normalized_statuses": {
                "SHP-3101": "shipped",
                "SHP-3102": "pending",
                "SHP-3103": "shipped",
                "SHP-3104": "delivered",
                "SHP-3105": "cancelled",
                "SHP-3106": "pending",
                "SHP-3107": "shipped",
                "SHP-3108": "delivered",
            },
            "review_ids": [
                "SHP-3102", "SHP-3104", "SHP-3106", "SHP-3107"],
        },
    }
# ---------------------------------------------------------------------------
# fixture 3: code (thread-safe LRU cache and tests)
# ---------------------------------------------------------------------------

_CODE_TASK = """Implement a complete Python 3.12 module containing a thread-safe, capacity-bounded LRU cache and its unit tests. Output Python code only, without Markdown. Use only the standard library.

Expose LRUCache(capacity), get(key, default=None), put(key, value), delete(key), clear(), __len__(), and snapshot(). Reject nonpositive capacity. A successful get and an update must move the key to most-recently-used. Inserting into a full cache must evict exactly the least-recently-used key. Stored None must remain distinguishable from a missing key internally. Snapshot must return an independent list of key/value pairs ordered least-to-most recent. All public operations must be synchronized.

Include deterministic unittest cases for eviction, updates, missing keys, stored None, deletion, clearing, snapshot independence, and concurrent access. Avoid sleep-based tests and avoid relying on a particular thread execution order. Include a runnable unittest entry point."""

_LRU_STUB = _block("""
    from __future__ import annotations

    from typing import Any


    class LRUCache:
        def __init__(self, capacity: int) -> None:
            ...

        def get(self, key: Any, default: Any = None) -> Any:
            ...

        def put(self, key: Any, value: Any) -> None:
            ...

        def delete(self, key: Any) -> None:
            ...

        def clear(self) -> None:
            ...

        def __len__(self) -> int:
            ...

        def snapshot(self) -> list[tuple[Any, Any]]:
            ...
""")


def _code_fixture() -> dict:
    context_paragraph = _block("""
        A worker process needs a small in-process LRU cache with no
        dependencies. The agreed surface is fixed: LRUCache(capacity),
        get(key, default=None), put(key, value), delete(key), clear(),
        __len__(), and snapshot(). The cache is used from more than one thread,
        so every public operation must be synchronized. The module and its
        unittest suite are produced in one response. Python 3.12 and the
        standard library are the only runtime.
    """)

    examples = _block("""
        FIXED API EXAMPLES AND BOUNDARY CASES

        Construction:
        LRUCache(2) is valid. LRUCache(0) and LRUCache(-1) raise ValueError.

        Basic use:
        put("a", 1); put("b", 2); get("a") returns 1. The successful get moves
        "a" to most-recently-used. Inserting "c" into the full cache evicts
        "b", because "b" is now least-recently-used. The resulting snapshot is
        [("a", 1), ("c", 3)] least-to-most recent.

        Missing and stored None:
        get("missing") returns None by default. get("missing", "fallback")
        returns "fallback". put("none", None); get("none", "fallback") returns
        None, and "none" is present in the snapshot. The implementation must
        keep the stored None distinct from absence internally; a sentinel or
        an explicit membership check is acceptable.

        Updates:
        put("a", 1); put("b", 2); put("a", 9) leaves both keys present, with
        "a" holding 9 and "a" most-recently-used. An update must not evict
        another key or change the length.

        Deletion and clearing:
        delete("a") removes "a" if present. delete("absent") is harmless.
        clear() removes every entry, and len(cache) is zero afterward.

        Snapshot:
        snapshot() returns a list of (key, value) pairs ordered from least to

        most recent. The list is independent: appending to it, replacing an
        element, or mutating a mutable value's container does not alter the
        cache's internal ordering or membership. Calling snapshot() does not
        change recency.

        Concurrency:
        A deterministic test can start several threads at a barrier. Each
        thread performs a fixed mix of put, get, delete, and len calls against
        one shared cache. Join the threads and assert that no exception
        escaped, len(cache) <= capacity, and every surviving key is one of the
        keys inserted. Do not assert a particular interleaving, eviction
        victim, or thread completion order. Do not use sleep to widen a race.
    """)

    compact_tests = _block("""
        COMPACT FIXED CASES
        Capacity two: put("a", 1), put("b", 2), get("a"), put("c", 3) leaves
        "a" and "c", with "b" evicted. Update the MRU key and verify that the
        LRU key survives. Store None and verify that a missing key still takes
        its default. Delete one key, clear the rest, and check len and
        snapshot ordering.
    """)

    test_design = _block("""
        TEST AND IMPLEMENTATION NOTES

        The tests should be organized as unittest.TestCase methods with short
        descriptive names. A helper can assert the exact snapshot order, but
        each test should make its own setup explicit. Use integer keys and
        small string values for most cases so a failure is easy to read. Use
        object() or a small tuple as a key in at least one case to show that
        the cache does not assume string keys. Use a mutable list as a stored
        value in the snapshot-independence test, then mutate the list and
        confirm that the cache still returns the same object/value semantics
        expected by the contract without letting snapshot() become a live
        internal view.

        Eviction tests need at least one case where insertion order and
        recency order differ. A useful sequence with capacity two is put a,
        put b, get a, put c: the new key is inserted last, the get promoted a,
        and b is the victim. A second sequence updates a before inserting c,
        which tests that an update promotes rather than evicting. Assert both
        membership and exact snapshot order, not only the length. A separate
        capacity-one test verifies that every insertion of a new key evicts
        the previous key and that updating the current key does not empty the
        cache.

        The missing-key test should cover the default argument and a missing
        key when the cache is empty and when it is full. A successful get of a
        stored None must return None without triggering the default. Do not
        use a sentinel value that could collide with a user value in the
        public API; if an internal sentinel is used, keep it private. A test
        can call get("none", "fallback") after put("none", None) and assert the
        result is None, then call get("absent", "fallback") and assert the
        fallback is returned. This is the critical distinction.

        Delete tests should cover an existing key, an absent key, deleting a
        key twice, and deleting the current MRU and LRU entries. Do not assert
        an undocumented return value. After deletion, assert length, snapshot
        membership, and the order of the remaining keys. Clearing tests should
        cover an empty cache, a cache with one entry, and a cache with several
        entries; after clear, len is zero, snapshot is empty, and a subsequent
        put works normally.

        Snapshot tests should assert least-to-most order after a sequence of
        gets and updates. Mutating the returned list must not change the
        cache: append a fake pair, reverse the list, and replace an element,
        then check that a second snapshot is unchanged and that get still
        returns the stored value. If a stored value is mutable, mutating that
        value is separate from mutating the snapshot list; the cache is
        allowed to return the same stored object. The snapshot list itself
        must be a fresh list of fresh two-tuples or equivalent independent
        pairs.

        Concurrency tests should use one shared cache and a fixed number of
        worker threads. A barrier makes the start simultaneous without
        depending on scheduling. Each worker can loop a small fixed number of
        times, calling put, get, delete, len, and snapshot with keys drawn
        from a small pool. Collect exceptions in a list protected by a lock,
        or call thread.join() and let an exception surface through a wrapper.
        After joining, assert the exception list is empty, len is between zero
        and capacity, and every key returned by snapshot is in the inserted
        pool. Do not assert which keys survived.

        The implementation should use an internal lock around every public
        operation, including __len__ and snapshot. get must distinguish a
        missing key from a stored None using membership or a private sentinel;
        it must not use the public default as the internal marker. A
        successful get and any put of an existing key must move that key to
        the MRU end. A put of a new key when full must remove exactly one LRU
        key. An update must not change the size or evict another key. delete
        and clear must release references so the cache does not keep deleted
        values alive. snapshot must build and return an independent list while
        holding the lock, then release the lock before returning.

        Avoid calling public methods while already holding the lock if those
        methods acquire the same non-reentrant lock. Prefer a small number of
        private helpers that assume the lock is held. Do not use a global
        cache, module-level mutable state, logging, file I/O, network I/O, or
        a background thread. Do not read the clock; this task has no expiry.
        Keep imports limited to the standard library. The test module should
        be runnable with python -m unittest and should not require pytest.

        Keep the tests deterministic and independent of dictionary iteration
        order. A failed assertion should show the expected snapshot and the
        actual snapshot, the operation sequence, and the capacity; a bare
        message such as "wrong" makes the eviction rule hard to debug. Reset
        the cache in each test rather than sharing one cache between methods.
        Do not rely on object identity for equal immutable values when
        checking membership; check keys and values through get or snapshot.
        The concurrency test may repeat a small workload a few times to make
        interleavings likely, but it must still pass with a single iteration
        and must not use sleep or wall-clock timing.

        A reasonable test matrix is: construction validation; basic put/get;
        missing-key default; stored None; update promotes; insertion eviction;
        capacity-one replacement; delete present/absent; clear empty/full;
        snapshot order; snapshot independence; concurrent mixed operations.
        The response should include the full module first and the full test
        module second, with a clear module boundary and no Markdown fences.
    """)

    head = _block("""
        You are the Python reviewer. Continue the cache implementation session
        using the agreed API and the fixed examples.
    """)

    tail = "TASK: " + _CODE_TASK

    return {
        "fixture_id": "code-thread-safe-lru",
        "context_type": "code",
        "title": "Thread-safe capacity-bounded LRU cache and tests",
        "context_paragraph": context_paragraph,
        "prior_user_prompts": [],
        "head": head,
        "body_compact": _compose(
            context_paragraph,
            "AGREED STUB:",
            _LRU_STUB,
            compact_tests),
        "body_long": _compose(
            context_paragraph,
            "AGREED STUB:",
            _LRU_STUB,
            examples,
            test_design),
        "required_prompt_markers": [
            "LRUCache(capacity)",
            "get(key, default=None)",
            "put(key, value)",
            "delete(key)",
            "clear()",
            "__len__()",
            "snapshot()",
            "class LRUCache:",
            "def __init__(self, capacity: int) -> None:",
            "def get(self, key: Any, default: Any = None) -> Any:",
            "def put(self, key: Any, value: Any) -> None:",
            "def delete(self, key: Any) -> None:",
            "def clear(self) -> None:",
            "def __len__(self) -> int:",
            "def snapshot(self) -> list[tuple[Any, Any]]:",
        ],
        "tail": tail,
        "task_requirements": {
            "runtime": "Python 3.12, standard library only",
            "class": "LRUCache",
            "methods": [
                "__init__", "get", "put", "delete", "clear", "__len__",
                "snapshot",
            ],
            "capacity": "reject nonpositive capacity",
            "recency": "successful get and update move key to MRU",
            "eviction": "evict exactly the LRU key when full",
            "stored_none": "distinguishable from missing internally",
            "snapshot": "independent list, least-to-most recent",
            "synchronization": "all public operations synchronized",
            "tests": "deterministic unittest, runnable entry point",
        },
        "answer_key": {
            "capacity_validation": "ValueError for capacity <= 0",
            "default": None,
            "stored_none": True,
            "snapshot_order": "least-to-most recent",
            "snapshot_independent": True,
            "eviction": "least-recently-used",
            "thread_safe": True,
        },
    }



# ---------------------------------------------------------------------------
# fixture 4: reasoning (weekly production allocation)
# ---------------------------------------------------------------------------

def _reasoning_fixture() -> dict:
    context_paragraph = _block("""
        A small factory plans one week across products A, B, and C. Each
        product uses machining hours, finishing hours, and one unit of a
        common raw material. Some units of each product are already committed
        to customers; the rest of each product's demand is capped. Overtime
        comes in fixed blocks, and material comes in fixed lots. The planning
        team wants one feasible allocation and the arithmetic behind it.
    """)

    head = _block("""
        You are the factory's production planner. Continue the weekly planning
        session using only the frozen inputs.
    """)

    memo = _block("""
        PLANNING MEMO - ONE WEEK

        All production quantities are integer units. The minimum units for each
        product are contractual commitments and cannot be reduced. The maximum
        units are demand caps and cannot be exceeded. Production may be any
        integer quantity between those bounds, not only the endpoints.

        Machining and finishing capacity are separate. A unit consumes the
        hours shown in the product table from each resource. Capacity may be
        left unused; unused hours have no value and cannot be transferred to
        another week. The base plan buys no overtime. Exactly one of the
        permitted overtime options is chosen for the week. An option adds its
        stated hours to both resources and incurs its fixed cost whether or not
        the added hours are used. Overtime options are not additive and cannot
        be combined; the table already gives the cumulative hours and cost for
        zero, one, or two blocks.

        Material is common across the three products: one finished unit
        consumes one material unit. Material is bought before production in
        whole lots of the two fixed sizes. A purchase may cover more units than
        are produced; unused material has no value. The number of lots of each
        size must be a nonnegative integer.

        Net contribution means gross contribution from the three products
        minus the fixed overtime cost and minus the material-lot cost. There
        are no other costs, no inventory, no subcontracting, and no salvage
        value. The recommended plan must satisfy every commitment, demand cap,
        and capacity limit.

        The useful check is exhaustive and small: for each permitted overtime
        option, enumerate every integer quantity of A, B, and C inside its
        bounds, discard allocations that exceed either capacity, find the
        cheapest covering combination of material lots, and compute net
        contribution. Compare the best plan under each overtime option, then
        compare the overall winner with the second-best plan. Report the
        arithmetic for the recommended plan and for each option's best plan.
    """)

    notes = _block("""
        REPORTING NOTES

        Use a compact table for the recommendation: product quantities,
        machining hours, finishing hours, material lots, gross contribution,
        overtime cost, material cost, and net contribution. State which
        capacity limits are binding. State whether the commitments are
        feasible without overtime before comparing the options. If the
        enumeration proves a unique best plan, say so; otherwise state the tie


        and show the tied plans.
    """)
    compact_note = _block("""
        COMPACT INPUT REMINDER
        Use the table as frozen input. Quantities are integers inside the
        commitment and demand-cap bounds. Capacity additions and their fixed
        costs come from the overtime option chosen; material purchases are
        separate whole lots.
    """)

    worksheet = _block("""
        PLANNING WORKSHEET AND AUDIT CHECKS

        Begin with commitments, not with the best-looking product. For each
        resource, multiply the committed units by that product's per-unit
        hours and add the three products. Compare each resource total with the
        base capacity. If either total exceeds base capacity, the commitments
        are not feasible without overtime, so an overtime option is required.
        If both totals fit, the base option remains permitted, but a different
        option may still produce a better net contribution. Do not assume that
        the cheapest overtime option is automatically the best one; its added
        capacity may be unusable or may enable a better product mix.

        For any candidate quantities (a, b, c), machining usage is the sum of
        a times A's machining rate, b times B's machining rate, and c times C's
        machining rate. Finishing usage is the analogous sum with finishing
        rates. A candidate is feasible only if both usages fit within the
        chosen option's base-plus-overtime capacity. Unused capacity is
        allowed. Keep the usage values in the same units as the input table;
        do not add a machining hour to a finishing hour.

        The option comparison has three layers. First, for each overtime
        option independently, find the best feasible production triple and
        its best material purchase. Second, compute each option's best net
        contribution after that option's fixed overtime cost. Third, compare
        the three option-level best values. A plan that is best for its option
        need not be the overall best plan. Report enough per-option arithmetic
        that a reader can see why the losing options lose; a bare final number
        is not an audit trail.

        Material purchasing is a small covering problem, not a proportional
        estimate. For a candidate total production T, let x be the number of
        five-unit lots and y the number of ten-unit lots. Both are
        nonnegative integers, and 5x + 10y must be at least T. The material
        cost is 12x + 17y. Because the larger lot has a lower cost per unit,
        the cheapest purchase for a fixed coverage is not necessarily all
        large lots: a small lot can avoid paying for ten units when only a few
        are needed. Enumerate a modest range of x and y for each candidate
        total, or enumerate all lot combinations once and keep the cheapest
        cost for every coverage level. The leftover material is 5x + 10y - T
        and has no salvage value; it affects cost only through the lot
        combination chosen.

        Gross contribution is the sum of each product's unit margin times its
        quantity. Margins are already in dollars per unit; do not multiply
        them by hours. Overtime cost is the fixed cost attached to the chosen
        option, not a per-hour rate. Material cost is the lot cost, not a
        per-unit rate. Net contribution is gross contribution minus overtime
        cost minus material cost. Keep the three components separable so the
        report can show the subtraction rather than only the net.

        A useful enumeration order is: choose an overtime option; loop a from
        its commitment to its demand cap; loop b over its range; loop c over
        its range; check machining and finishing; for each feasible triple,
        find the cheapest material cover; compute net. Keep the best triple
        for the option, including its usage, lots, costs, and leftover
        material. After all options, sort or compare the option winners and
        identify whether the overall winner is unique. A second-best plan can
        come from a different option or from the same option with a different
        triple; compare all candidates, not only the option winners, if the
        report asks for the runner-up gap.

        Reporting worksheet fields: the recommended option name and its added
        hours/cost; quantities of A, B, and C; machining usage and capacity;
        finishing usage and capacity; which capacities are binding; gross
        contribution; material lot counts; material units purchased and
        unused; material cost; overtime cost; net contribution. For each
        option, report its best feasible production and net so the comparison
        is visible. For the unique-best claim, report the best plan and the
        next-best plan with their net values and the gap.

        Audit checks before answering: every quantity is an integer within its
        bounds; both resource usages are within the selected capacity; the
        material purchase covers total production; lot counts are
        nonnegative integers; gross minus overtime minus material equals net;
        the chosen overtime option is one of the permitted rows and is not a
        sum of rows; no cost is counted twice; the reported binding
        constraints match the computed usages; and the final recommendation
        is the maximum net plan, not merely the maximum gross plan. If two
        plans tie on net, state the tie rather than inventing a tie-break.

        Keep the arithmetic in a form that can be checked line by line. If a
        capacity is exactly full, say so explicitly; if it has slack, give the
        slack rather than calling it binding. Do not report a material
        purchase in units when the question asks for lots, and do not report a
        lot count without its cost. Distinguish the overtime option's added
        capacity from the total capacity available under that option. Finally,
        verify that the recommended plan is feasible under the option that
        supplies its hours and that the same quantities would not be feasible
        under a cheaper option with less capacity.
    """)

    proof = reasoning_oracle()
    tail = _reasoning_tail()

    return {
        "fixture_id": "reasoning-production-allocation",
        "context_type": "reasoning",
        "title": "Weekly production allocation with overtime and material lots",
        "context_paragraph": context_paragraph,
        "prior_user_prompts": [],
        "head": head,
        "body_compact": _compose(
            context_paragraph,
            _reasoning_inputs_table(),
            compact_note),
        "body_long": _compose(
            context_paragraph,
            memo,
            _reasoning_inputs_table(),
            notes,
            worksheet),
        "required_prompt_markers": [
            "product | contribution $/unit | machining h/unit | "
            "finishing h/unit | minimum units | maximum units",
            "A | 15 | 2 | 1 | 3 | 6",
            "B | 9 | 1 | 2 | 2 | 8",
            "C | 6 | 1 | 1 | 3 | 12",
            "regular capacity: 23 machining hours, 26 finishing hours.",
            "overtime options (choose exactly one): blocks | added "
            "machining h | added finishing h | fixed cost $",
            "0 | 0 | 0 | 0",
            "1 | 6 | 5 | 20",
            "2 | 12 | 10 | 40",
            "material lots: 5 units for $12; 10 units for $17.",
        ],
        "tail": tail,
        "task_requirements": {
            "deliverable": (
                "ordinary prose with a compact results table; no JSON or code"),
            "commitments_first": True,
            "compare": "base plan and each permitted overtime option",
            "report": [
                "production by product",
                "machining and finishing hours",
                "material lots and unused material",
                "gross contribution",
                "overtime cost",
                "material cost",
                "net contribution",
                "binding constraints",
                "second-best comparison",
            ],
            "constraints": [
                "integer units",
                "minimum commitments and demand caps",
                "machining and finishing capacities",
                "whole material lots",
                "exactly one overtime option",
            ],
        },
        "answer_key": {
            "optimal_plan": proof["optimal_plan"],
            "option_summaries": proof["option_summaries"],
            "commitments": proof["commitments"],
            "second_best_plan": proof["second_best_plan"],
            "second_best_gap_usd": proof["second_best_gap_usd"],
            "unique_optimum": proof["unique_optimum"],
            "optimal_plan_count": proof["optimal_plan_count"],
            "enumeration": proof["enumeration"],
            "method": proof["method"],
        },
    }


FIXTURES = [
    _prose_fixture(),
    _structured_fixture(),
    _code_fixture(),
    _reasoning_fixture(),
]

REASONING_ORACLE = reasoning_oracle(verify=False)

_FIXTURE_BY_ID = {fixture["fixture_id"]: fixture for fixture in FIXTURES}
_FIXTURE_BY_TYPE = {
    fixture["context_type"]: fixture for fixture in FIXTURES
}


# ---------------------------------------------------------------------------
# fixture lookup and exact-token prompt construction
# ---------------------------------------------------------------------------

def fixture_for(fixture_id: str) -> dict:
    try:
        return _FIXTURE_BY_ID[fixture_id]
    except KeyError:
        raise KeyError(
            f"unknown fixture_id {fixture_id!r}; "
            f"known: {', '.join(FIXTURE_IDS)}") from None


def fixture_for_type(context_type: str) -> dict:
    try:
        return _FIXTURE_BY_TYPE[context_type]
    except KeyError:
        raise KeyError(
            f"unknown context_type {context_type!r}; "
            f"known: {', '.join(CONTEXT_TYPES)}") from None


def _body_for_target(fixture: dict, target: int) -> tuple[str, str]:
    if target == SHAPES["decode"]:
        return "compact", fixture["body_compact"]
    if target == SHAPES["prefill"]:
        return "long", fixture["body_long"]
    raise ValueError(
        f"target prompt tokens must be one of {sorted(TARGET_TO_SHAPE)}; "
        f"got {target}")


def _served_builders():
    """Load the proven exact builder from the sibling harness module."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from dspark_workload_eval import (  # noqa: PLC0415
        CONNECTORS,
        boundary_score,
        build_exact_prompt,
    )
    return build_exact_prompt, CONNECTORS, boundary_score


def _ids_of(tok, text: str) -> list[int]:
    return [int(token) for token in tok.ids(text)]


def _suffix_matches(ids: list[int], tail_ids: list[int]) -> bool:
    return bool(tail_ids) and len(ids) >= len(tail_ids) and (
        ids[-len(tail_ids):] == tail_ids)


def _prefix_matches(ids: list[int], head_ids: list[int]) -> bool:
    return bool(head_ids) and len(ids) >= len(head_ids) and (
        ids[:len(head_ids)] == head_ids)


def _relaxed_prefix_matches(ids: list[int], head_ids: list[int]) -> bool:
    """Head tokens except the last, which may merge with the first body token.

    The head text itself is checked separately with ``startswith``.  Token
    equality of the whole head is not a valid requirement at a concatenation
    boundary: BPE can merge the head's final character with the first body
    character into one token without changing either text.
    """
    checked = max(0, len(head_ids) - 1)
    return checked == 0 or (
        len(ids) >= checked and ids[:checked] == head_ids[:checked])

def _exact_prompt(
    tok,
    head: str,
    body: str,
    tail: str,
    target: int,
) -> tuple[dict, list[int]]:
    """Build an exact prompt while keeping head and full tail intact.

    A blank line is inserted between the head and the body so the head's
    final token cannot merge into the first body word.  The sibling builder
    is tried first; if its body/tail boundary merges the first tail token, a
    bounded search over natural connectors and nearby body cuts finds a build
    whose served IDs end with the complete tail token sequence.  Exactness,
    text-level head/tail preservation, and the tail token suffix are hard
    requirements; body text is the only thing cut.
    """
    build_exact_prompt, connectors, boundary_score = _served_builders()
    head_ids = _ids_of(tok, head)
    build_head = head if head.endswith("\n\n") else head + "\n\n"
    build_head_ids = _ids_of(tok, build_head)
    tail_ids = _ids_of(tok, tail)
    need = target - len(build_head_ids) - len(tail_ids)
    if need < MIN_BODY_TOKENS:
        raise ValueError(
            f"head ({len(build_head_ids)}) + tail ({len(tail_ids)}) leave "
            f"only {need} body tokens for target {target}")

    def accepted(text: str):
        ids = _ids_of(tok, text)
        if len(ids) != target:
            return None
        if not text.startswith(build_head) or not text.endswith(tail):
            return None
        # The last head token may merge with the first body token.  The head
        # text itself is still verbatim; require every earlier head token.
        if not _relaxed_prefix_matches(ids, head_ids):
            return None
        if not _suffix_matches(ids, tail_ids):
            return None
        return ids

    body_ids = _ids_of(tok, body)
    if len(body_ids) < need:
        raise ValueError(
            f"body has {len(body_ids)} tokens, need at least {need} for "
            f"target {target}")

    sibling = None
    try:
        sibling = build_exact_prompt(tok, build_head, body, tail, target)
        ids = accepted(sibling["text"])
        # A bare concatenation puts the tail's "TASK:" directly after the
        # body, usually with no whitespace.  Keep a newline boundary when the
        # sibling already found one; otherwise let the search below place it.
        if ids is not None and sibling.get("connector") in ("\n\n", "\n"):
            result = dict(sibling)
            result.update({
                "head_tokens": len(head_ids),
                "build_head_tokens": len(build_head_ids),
                "head_separator": "\n\n",
            })
            return result, ids
    except Exception:
        sibling = None

    # Prefer a blank line at the body/tail boundary: it prevents the first
    # tail token from merging with the body.  Keep every sibling connector as
    # a fallback, but test the separating connectors first.
    connector_order = []
    for connector in ("\n\n", "\n", " ", "", *connectors):
        if connector not in connector_order:
            connector_order.append(connector)

    centers = [need]
    if sibling is not None:
        sibling_cut = int(sibling.get("body_tokens_used", need))
        if sibling_cut != need:
            centers.append(sibling_cut)
    iterations = 0
    # Start next to the exactness cut the sibling found.  A different
    # connector usually shifts the served count by only a token or two, so
    # a narrow window resolves the boundary cheaply; widen only if needed.
    for radius in (8, 32, 160):
        cuts = []
        seen_cuts = set()
        for center in centers:
            for delta in range(-radius, radius + 1):
                cut = center + delta
                if cut < MIN_BODY_TOKENS or cut > len(body_ids):
                    continue
                if cut in seen_cuts:
                    continue
                seen_cuts.add(cut)
                cuts.append(cut)
        stage_best = None
        stage_ids = None
        for connector in connector_order:
            for cut in cuts:
                iterations += 1
                body_text = tok.detokenize(body_ids[:cut])
                text = build_head + body_text + connector + tail
                if int(tok.count(text)) != target:
                    continue
                ids = accepted(text)
                if ids is None:
                    continue
                candidate = {
                    "text": text,
                    "target": target,
                    "actual": len(ids),
                    "head_tokens": len(head_ids),
                    "build_head_tokens": len(build_head_ids),
                    "head_separator": "\n\n",
                    "tail_tokens": len(tail_ids),
                    "connector": connector,
                    "connector_chars": len(connector),
                    "body_tokens_used": cut,
                    "body_tokens_available": len(body_ids),
                    "boundary_score": boundary_score(body_text),
                    "body_tail_preview": body_text[-80:],
                    "iterations": iterations,
                    "sha256": _digest(text),
                    "chars": len(text),
                }
                if stage_best is None or (
                    candidate["boundary_score"], -candidate["iterations"]
                ) > (
                    stage_best["boundary_score"], -stage_best["iterations"]
                ):
                    stage_best = candidate
                    stage_ids = ids
                if stage_best["boundary_score"] >= 2:
                    return stage_best, stage_ids
        if stage_best is not None:
            return stage_best, stage_ids
    raise RuntimeError(
        f"could not build exact {target}-token prompt with intact head/tail "
        f"after {iterations} candidate builds")


def build_prompt(
    tok,
    fixture_id: str,
    target: int,
    shape: str | None = None,
) -> dict:
    """Build one exact prompt with the served tokenizer.

    ``tok`` must be the served tokenizer client (the existing
    ``dspark_workload_eval.Tokenizer`` is compatible): ``count``, ``ids``,
    and ``detokenize`` are used.  No approximate character-to-token
    arithmetic is accepted.
    """
    fixture = fixture_for(fixture_id)
    target = int(target)
    expected_shape = TARGET_TO_SHAPE.get(target)
    if expected_shape is None:
        raise ValueError(
            f"target prompt tokens must be one of {sorted(TARGET_TO_SHAPE)}; "
            f"got {target}")
    if shape is None:
        shape = expected_shape
    elif shape != expected_shape:
        raise ValueError(
            f"shape {shape!r} maps to {SHAPES.get(shape)}, not {target}")

    body_variant, body = _body_for_target(fixture, target)
    head = fixture["head"]
    tail = fixture["tail"]
    head_tokens = int(tok.count(head))
    tail_tokens = int(tok.count(tail))
    if tail_tokens > TAIL_TOKEN_BUDGET:
        raise ValueError(
            f"{fixture_id}: tail is {tail_tokens} served tokens, above the "
            f"{TAIL_TOKEN_BUDGET}-token budget")
    if head_tokens + tail_tokens + MIN_BODY_TOKENS > target:
        raise ValueError(
            f"{fixture_id}/{shape}: head+tail leave too few body tokens")

    built, ids = _exact_prompt(tok, head, body, tail, target)
    text = built["text"]
    if int(built["actual"]) != target:
        raise RuntimeError(
            f"{fixture_id}/{shape}: builder returned {built['actual']} "
            f"tokens, target {target}")
    if len(ids) != target:
        raise RuntimeError(
            f"{fixture_id}/{shape}: served tokenizer returned {len(ids)} "
            f"tokens, target {target}")
    if not text.startswith(head):
        raise RuntimeError(f"{fixture_id}/{shape}: head was not preserved")
    if not text.endswith(tail):
        raise RuntimeError(f"{fixture_id}/{shape}: tail was not preserved")
    head_ids = _ids_of(tok, head)
    if not _relaxed_prefix_matches(ids, head_ids):
        raise RuntimeError(
            f"{fixture_id}/{shape}: head token prefix changed before the "
            f"boundary token")
    tail_ids = _ids_of(tok, tail)
    if not _suffix_matches(ids, tail_ids):
        raise RuntimeError(
            f"{fixture_id}/{shape}: tail token suffix changed at the "
            f"body boundary")

    return {
        "fixture_id": fixture_id,
        "context_type": fixture["context_type"],
        "title": fixture["title"],
        "shape": shape,
        "target_prompt_tokens": target,
        "prompt_text": text,
        "prompt_sha256": _digest(text),
        "prompt_chars": len(text),
        "prompt_token_ids": ids,
        "head_text": head,
        "head_tokens": head_tokens,
        "build_head_tokens": int(built.get("build_head_tokens", head_tokens)),
        "head_separator": built.get("head_separator", "\n\n"),
        "head_token_ids": head_ids,
        "head_token_ids_prefix": _relaxed_prefix_matches(ids, head_ids),
        "head_token_ids_prefix_strict": _prefix_matches(ids, head_ids),
        "head_text_preserved": text.startswith(head),
        "tail_text": tail,
        "tail_tokens": tail_tokens,
        "tail_token_ids": tail_ids,
        "tail_text_preserved": text.endswith(tail),
        "tail_within_512": tail_tokens <= TAIL_TOKEN_BUDGET,
        "tail_token_ids_contiguous": _suffix_matches(ids, tail_ids),
        "body_variant": body_variant,
        "body_tokens_used": int(built["body_tokens_used"]),
        "body_tokens_available": int(built.get("body_tokens_available", 0)),
        "connector": built.get("connector", ""),
        "boundary_score": int(built.get("boundary_score", 0)),
        "fitting_iterations": int(built.get("iterations", 0)),
        "add_special_tokens": False,
        "tokenizer_source": "/tokenize",
    }


def build_prompt_set(
    tok,
    fixture_ids=None,
    shapes=None,
) -> dict:
    """Build every requested prompt, keyed by ``(fixture_id, shape)``."""
    if isinstance(fixture_ids, str):
        fixture_ids = (fixture_ids,)
    if isinstance(shapes, str):
        shapes = (shapes,)
    ids = tuple(FIXTURE_IDS if fixture_ids is None else fixture_ids)
    shape_names = tuple(SHAPES if shapes is None else shapes)
    for fixture_id in ids:
        fixture_for(fixture_id)
    for shape in shape_names:
        if shape not in SHAPES:
            raise ValueError(
                f"unknown shape {shape!r}; known: {', '.join(SHAPES)}")
    prompts = {}
    for fixture_id in ids:
        for shape in shape_names:
            prompts[(fixture_id, shape)] = build_prompt(
                tok, fixture_id, SHAPES[shape], shape)
    return prompts


def validate_prompt(tok, prompt: dict) -> dict:
    """Re-check one prompt with the served tokenizer; never raises."""
    errors = []
    details = {}
    try:
        fixture_id = prompt["fixture_id"]
        fixture = fixture_for(fixture_id)
        shape = prompt["shape"]
        target = int(prompt["target_prompt_tokens"])
        text = prompt["prompt_text"]
        head = prompt.get("head_text", fixture["head"])
        tail = prompt.get("tail_text", fixture["tail"])

        if shape not in SHAPES:
            errors.append(f"unknown shape {shape!r}")
        elif SHAPES[shape] != target:
            errors.append(
                f"shape {shape!r} maps to {SHAPES[shape]}, prompt says "
                f"{target}")
        required_markers = list(fixture["required_prompt_markers"])
        missing_markers = [
            marker for marker in required_markers if marker not in text]

        if prompt.get("context_type") != fixture["context_type"]:
            errors.append("context_type does not match fixture registry")

        head_ids = _ids_of(tok, head)
        tail_ids = _ids_of(tok, tail)
        ids = _ids_of(tok, text)
        again = _ids_of(tok, text)
        details = {
            "fixture_id": fixture_id,
            "context_type": fixture["context_type"],
            "shape": shape,
            "target_prompt_tokens": target,
            "token_count": len(ids),
            "required_prompt_marker_count": len(required_markers),
            "missing_required_prompt_markers": missing_markers,
            "head_tokens": len(head_ids),
            "tail_tokens": len(tail_ids),
            "prompt_chars": len(text),
            "retokenize_stable": ids == again,
            "head_text_preserved": text.startswith(head),
            "tail_text_preserved": text.endswith(tail),
            "head_token_ids_prefix": _relaxed_prefix_matches(ids, head_ids),
            "head_token_ids_prefix_strict": _prefix_matches(ids, head_ids),
            "tail_token_ids_contiguous": _suffix_matches(ids, tail_ids),
            "tail_within_512": len(tail_ids) <= TAIL_TOKEN_BUDGET,
            "ids_match_prompt_record": ids == [
                int(value) for value in prompt["prompt_token_ids"]],
            "sha256_matches_prompt_text": (
                prompt["prompt_sha256"] == _digest(text)),
        }
        if len(ids) != target:
            errors.append(f"token count {len(ids)} != target {target}")
        if not details["retokenize_stable"]:
            errors.append("re-tokenized IDs are not stable")
        if not details["head_text_preserved"]:
            errors.append("head text is not a prefix")
        if not details["tail_text_preserved"]:
            errors.append("full tail text is not a suffix")
        if not details["head_token_ids_prefix"]:
            errors.append("head token IDs are not a prefix")
        if not details["tail_token_ids_contiguous"]:
            errors.append("tail token IDs are not a contiguous suffix")
        if not details["tail_within_512"]:
            errors.append(
                f"tail {len(tail_ids)} exceeds {TAIL_TOKEN_BUDGET} tokens")
        if not details["ids_match_prompt_record"]:
            errors.append("stored prompt_token_ids differ from served IDs")
        if missing_markers:
            preview = "; ".join(
                marker[:80] for marker in missing_markers[:3])
            errors.append(
                f"{len(missing_markers)} required prompt marker(s) missing: "
                f"{preview}")
        if not details["sha256_matches_prompt_text"]:
            errors.append("prompt_sha256 does not match prompt_text")
    except Exception as exc:  # validation must report, not raise
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"ok": not errors, "errors": errors, "details": details}


# ---------------------------------------------------------------------------
# definitions, digests, artifact writing, and offline checks
# ---------------------------------------------------------------------------

def prompt_definitions() -> list[dict]:
    """Human-readable authored definitions; no tokenizer or token counts."""
    return [
        {
            "fixture_id": fixture["fixture_id"],
            "context_type": fixture["context_type"],
            "title": fixture["title"],
            "context_paragraph": fixture["context_paragraph"],
            "prior_user_prompts": list(fixture["prior_user_prompts"]),
            "head": fixture["head"],
            "body_compact": fixture["body_compact"],
            "body_long": fixture["body_long"],
            "required_prompt_markers": list(
                fixture["required_prompt_markers"]),
            "tail": fixture["tail"],
            "task_requirements": fixture["task_requirements"],
        }
        for fixture in FIXTURES
    ]


def fixtures_digest() -> str:
    canonical = json.dumps(
        prompt_definitions(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _digest(canonical)


def offline_validate() -> dict:
    """Static checks only.  Deliberately makes no tokenizer or network call."""
    errors = []
    warnings = []
    if tuple(fixture["fixture_id"] for fixture in FIXTURES) != FIXTURE_IDS:
        errors.append("FIXTURES order differs from FIXTURE_IDS")
    if tuple(fixture["context_type"] for fixture in FIXTURES) != CONTEXT_TYPES:
        errors.append("FIXTURES order differs from CONTEXT_TYPES")
    if len(set(FIXTURE_IDS)) != len(FIXTURE_IDS):
        errors.append("duplicate fixture_id")
    if len(set(CONTEXT_TYPES)) != len(CONTEXT_TYPES):
        errors.append("duplicate context_type")

    for fixture in FIXTURES:
        missing = [
            key for key in REQUIRED_FIXTURE_KEYS
            if key not in fixture or fixture[key] in (None, "")]
        if missing:
            errors.append(
                f"{fixture.get('fixture_id', '<unknown>')}: missing {missing}")
            continue
        fixture_id = fixture["fixture_id"]
        if len(fixture["body_compact"]) <= len(fixture["tail"]):
            warnings.append(
                f"{fixture_id}: compact body is not longer than its tail")
        if len(fixture["body_long"]) <= len(fixture["body_compact"]):
            warnings.append(
                f"{fixture_id}: long body is not longer than compact body")
        marker_values = list(fixture["required_prompt_markers"])
        if not marker_values:
            errors.append(f"{fixture_id}: no required_prompt_markers")
        for marker in marker_values:
            if marker not in fixture["body_compact"]:
                errors.append(
                    f"{fixture_id}: compact body lost required marker "
                    f"{marker[:80]!r}")
            if marker not in fixture["body_long"]:
                errors.append(
                    f"{fixture_id}: long body lost required marker "
                    f"{marker[:80]!r}")
        if len(fixture["body_long"]) <= 3000:
            warnings.append(
                f"{fixture_id}: long body is only {len(fixture['body_long'])} "
                f"characters; runtime exact-fit may reject it")
        # A character guard, not a token claim.  Runtime /tokenize enforces
        # the real 512-token tail budget.
        if len(fixture["tail"]) > 2400:
            errors.append(
                f"{fixture_id}: tail is {len(fixture['tail'])} characters; "
                f"the runtime 512-token check is likely to reject it")
        if not fixture["tail"].startswith("TASK:"):
            warnings.append(f"{fixture_id}: tail does not start with TASK:")

    proof = reasoning_oracle()
    if not proof["unique_optimum"]:
        errors.append("reasoning fixture optimum is not unique")
    reasoning = fixture_for_type("reasoning")
    optimal = reasoning["answer_key"]["optimal_plan"]
    if (
        optimal["net_contribution_usd"]
        != proof["optimal_plan"]["net_contribution_usd"]
    ):
        errors.append("reasoning answer_key net contribution differs")
    if optimal["overtime_blocks"] != proof["optimal_plan"]["overtime_blocks"]:
        errors.append("reasoning answer_key blocks differ from oracle")
    if optimal["production"] != proof["optimal_plan"]["production"]:
        errors.append("reasoning answer_key production differs from oracle")
    if optimal["material_lots"] != proof["optimal_plan"]["material_lots"]:
        errors.append("reasoning answer_key material lots differ from oracle")
    if (
        reasoning["answer_key"]["option_summaries"]
        != proof["option_summaries"]
    ):
        errors.append("reasoning answer_key option summaries differ from oracle")
    second_best = proof["second_best_plan"]
    if second_best is not None:
        key_second = reasoning["answer_key"]["second_best_plan"]
        if (
            key_second["net_contribution_usd"]
            != second_best["net_contribution_usd"]
        ):
            errors.append("reasoning answer_key second-best differs from oracle")
        if (
            reasoning["answer_key"]["second_best_gap_usd"]
            != proof["second_best_gap_usd"]
        ):
            errors.append("reasoning answer_key second-best gap differs")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "fixtures": len(FIXTURES),
        "fixtures_digest": fixtures_digest(),
        "reasoning_oracle": {
            "unique_optimum": proof["unique_optimum"],
            "optimal_net_contribution_usd": (
                proof["optimal_plan"]["net_contribution_usd"]),
            "second_best_net_contribution_usd": (
                None if second_best is None
                else second_best["net_contribution_usd"]),
            "second_best_gap_usd": proof["second_best_gap_usd"],
            "enumeration": proof["enumeration"],
        },
    }
def build_artifact(
    tok,
    out_dir,
    fixture_ids=None,
    shapes=None,
) -> dict:
    """Build and validate exact prompts, then write runtime JSON artifacts.

    This is the only function that writes files, and it does so only when
    called with the served tokenizer after the campaign is cleared to prep.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = build_prompt_set(tok, fixture_ids=fixture_ids, shapes=shapes)
    validations = {}
    prompt_rows = []
    for (fixture_id, shape), prompt in prompts.items():
        validation = validate_prompt(tok, prompt)
        validations[f"{fixture_id}|{shape}"] = validation
        if not validation["ok"]:
            raise RuntimeError(
                f"{fixture_id}/{shape} failed validation: "
                f"{validation['errors']}")
        prompt_rows.append(prompt)

    definitions = prompt_definitions()
    generated_at = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z")
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "generated_at": generated_at,
        "fixtures_digest": fixtures_digest(),
        "shapes": dict(SHAPES),
        "tail_token_budget": TAIL_TOKEN_BUDGET,
        "prompt_count": len(prompt_rows),
        "prompts": prompt_rows,
        "validations": validations,
    }
    prompts_path = out / "fixtures_distinct_prompts.json"
    definitions_path = out / "fixtures_distinct_definitions.json"
    prompts_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    definitions_path.write_text(
        json.dumps(
            {
                "schema": "fixtures-distinct-definitions-v1",
                "fixtures_digest": manifest["fixtures_digest"],
                "fixtures": definitions,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8")
    return {
        "prompts_path": str(prompts_path),
        "definitions_path": str(definitions_path),
        "prompt_count": len(prompt_rows),
        "fixtures_digest": manifest["fixtures_digest"],
        "manifest": manifest,
    }


def write_prompt_artifacts(tok, out_dir, fixture_ids=None, shapes=None):
    """Alias used by the runtime prep command."""
    return build_artifact(
        tok, out_dir, fixture_ids=fixture_ids, shapes=shapes)


__all__ = [
    "ARTIFACT_SCHEMA",
    "CONTEXT_TYPES",
    "FIXTURES",
    "FIXTURE_IDS",
    "REASONING_INPUTS",
    "REASONING_ORACLE",
    "SHAPES",
    "TAIL_TOKEN_BUDGET",
    "build_artifact",
    "build_prompt",
    "build_prompt_set",
    "fixture_for",
    "fixture_for_type",
    "fixtures_digest",
    "offline_validate",
    "prompt_definitions",
    "reasoning_oracle",
    "validate_prompt",
    "write_prompt_artifacts",
]
