from agent_committee.arms.retrieval import Retriever
from agent_committee.schemas import Case


def _pool():
    return [
        Case(id="b", text="lost my card", label="x"),
        Case(id="a", text="lost my card", label="x"),
        Case(id="c", text="top up my account by transfer", label="y"),
        Case(id="d", text="exchange rate question", label="z"),
    ]


def test_ranks_by_similarity_and_breaks_ties_by_id():
    r = Retriever(_pool())
    out = r.nearest("lost my card", k=3)
    assert [c.id for c, _ in out[:2]] == ["a", "b"]
    assert out[0][1] == out[1][1] > out[2][1]
    assert len(r) == 4


def test_order_independent_and_k_bounds():
    a = Retriever(_pool()).nearest("card top up", k=4)
    b = Retriever(list(reversed(_pool()))).nearest("card top up", k=4)
    assert a == b
    assert Retriever(_pool()).nearest("card", k=0) == []
    assert len(Retriever(_pool()).nearest("card", k=99)) == 4
