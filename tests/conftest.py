"""Shared fixtures: a small deterministic transport task and plans over it.

The task is deliberately tiny so that every expected number in the suite can be
worked out by hand::

    locations : l0, l1, l2      trucks : tr1, tr2
    fluents   : at(truck, location), delivered(location)
    actions   : move(t, from, to), drop(t, l)
    goals     : delivered(l1), delivered(l2)

Its plans are chosen so that each metric is the one that separates a particular
pair, which is what lets the per-metric tests assert exact values:

===================  =======  =========  =============  ==========
plan                 actions  goal order  trucks used   length
===================  =======  =========  =============  ==========
plan_l1_then_l2      4        l1, l2      tr1           4
plan_l2_then_l1      4        l2, l1      tr1           4
plan_two_trucks      4        l1, l2      tr1, tr2      4
plan_l1_then_l2_long 6        l1, l2      tr1           6 (detour)
===================  =======  =========  =============  ==========
"""

import pytest
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    IntType,
    Object,
    ObjectExp,
    Problem,
    UserType,
)


@pytest.fixture
def domain():
    """The transport task plus the handles tests need to build expressions."""
    location = UserType("Location")
    truck = UserType("Truck")

    at = Fluent("at", BoolType(), t=truck, l=location)
    delivered = Fluent("delivered", BoolType(), l=location)
    fuel = Fluent("fuel", IntType(0, 100))

    move = InstantaneousAction("move", t=truck, f=location, to=location)
    t, f, to = move.parameter("t"), move.parameter("f"), move.parameter("to")
    move.add_precondition(at(t, f))
    move.add_effect(at(t, f), False)
    move.add_effect(at(t, to), True)
    move.add_decrease_effect(fuel, 10)

    drop = InstantaneousAction("drop", t=truck, l=location)
    dt, dl = drop.parameter("t"), drop.parameter("l")
    drop.add_precondition(at(dt, dl))
    drop.add_effect(delivered(dl), True)

    problem = Problem("transport")
    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(delivered, default_initial_value=False)
    problem.add_fluent(fuel, default_initial_value=100)
    problem.add_action(move)
    problem.add_action(drop)

    objects = {name: Object(name, location) for name in ("l0", "l1", "l2")}
    objects.update({name: Object(name, truck) for name in ("tr1", "tr2")})
    problem.add_objects(objects.values())

    problem.set_initial_value(at(objects["tr1"], objects["l0"]), True)
    problem.set_initial_value(at(objects["tr2"], objects["l0"]), True)

    # Goal order matters: GoalPredicateOrdering derives its canonical predicate
    # order from problem.goals.
    problem.add_goal(delivered(objects["l1"]))
    problem.add_goal(delivered(objects["l2"]))

    return {
        "problem": problem,
        "move": move,
        "drop": drop,
        "at": at,
        "delivered": delivered,
        "fuel": fuel,
        **objects,
    }


@pytest.fixture
def task(domain):
    return domain["problem"]


def _plan(*steps):
    return SequentialPlan(
        [
            ActionInstance(action, tuple(ObjectExp(p) for p in parameters))
            for action, parameters in steps
        ]
    )


@pytest.fixture
def make_plan():
    """Build a SequentialPlan from ``(action, (objects, ...))`` pairs."""
    return _plan


@pytest.fixture
def empty_plan():
    """A plan with no actions. Simulating it yields only the initial state."""
    return _plan()


@pytest.fixture
def plan_l1_then_l2(domain):
    """tr1 delivers l1 first, then l2. Four actions."""
    move, drop = domain["move"], domain["drop"]
    tr1, l0, l1, l2 = domain["tr1"], domain["l0"], domain["l1"], domain["l2"]
    return _plan(
        (move, (tr1, l0, l1)),
        (drop, (tr1, l1)),
        (move, (tr1, l1, l2)),
        (drop, (tr1, l2)),
    )


@pytest.fixture
def plan_l2_then_l1(domain):
    """Same length and same resources as plan_l1_then_l2, reversed goal order.

    Shares no action with it, so Stability is 1.0 while the resource metrics
    are 0.0 -- the pair that separates structural from resource diversity.
    """
    move, drop = domain["move"], domain["drop"]
    tr1, l0, l1, l2 = domain["tr1"], domain["l0"], domain["l1"], domain["l2"]
    return _plan(
        (move, (tr1, l0, l2)),
        (drop, (tr1, l2)),
        (move, (tr1, l2, l1)),
        (drop, (tr1, l1)),
    )


@pytest.fixture
def plan_two_trucks(domain):
    """tr1 delivers l1, tr2 delivers l2. Same goal order as plan_l1_then_l2,
    but a different resource set -- the pair that separates the two."""
    move, drop = domain["move"], domain["drop"]
    tr1, tr2 = domain["tr1"], domain["tr2"]
    l0, l1, l2 = domain["l0"], domain["l1"], domain["l2"]
    return _plan(
        (move, (tr1, l0, l1)),
        (drop, (tr1, l1)),
        (move, (tr2, l0, l2)),
        (drop, (tr2, l2)),
    )


@pytest.fixture
def plan_l1_then_l2_long(domain):
    """plan_l1_then_l2 with a pointless detour through l2 first. Six actions.

    Reaches the goals in the same order via a superset of the actions, so it is
    the plan that exercises the unequal-length branch of States.
    """
    move, drop = domain["move"], domain["drop"]
    tr1, l0, l1, l2 = domain["tr1"], domain["l0"], domain["l1"], domain["l2"]
    return _plan(
        (move, (tr1, l0, l2)),
        (move, (tr1, l2, l0)),
        (move, (tr1, l0, l1)),
        (drop, (tr1, l1)),
        (move, (tr1, l1, l2)),
        (drop, (tr1, l2)),
    )


@pytest.fixture
def plan_no_goals(domain):
    """A plan that executes but achieves neither goal."""
    move = domain["move"]
    tr1, l0, l1 = domain["tr1"], domain["l0"], domain["l1"]
    return _plan((move, (tr1, l0, l1)))


@pytest.fixture
def plan_only_l2(domain):
    """A plan that achieves the second goal and never the first.

    The plan that pins down where an unachieved goal ranks: it must fall
    *after* the achieved one, giving the ordering (l2, l1).
    """
    move, drop = domain["move"], domain["drop"]
    tr1, l0, l2 = domain["tr1"], domain["l0"], domain["l2"]
    return _plan((move, (tr1, l0, l2)), (drop, (tr1, l2)))


@pytest.fixture
def inapplicable_plan(domain):
    """drop(tr1, l1) without moving tr1 to l1 first: the precondition fails."""
    return _plan((domain["drop"], (domain["tr1"], domain["l1"])))


@pytest.fixture
def planset(plan_l1_then_l2, plan_l2_then_l1, plan_two_trucks):
    """The three-plan set most model tests score."""
    return [plan_l1_then_l2, plan_l2_then_l1, plan_two_trucks]


@pytest.fixture
def resource_file(tmp_path):
    """A ``(:resource ...)`` file naming both trucks as resources."""
    path = tmp_path / "resources.txt"
    path.write_text("(:resource tr1 0 10 1)\n(:resource tr2 0 10 1)\n")
    return str(path)
