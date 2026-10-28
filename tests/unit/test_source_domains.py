import polars as pl

from pit_multisource.build.source_domains import build_source_domains


def _records(rows: list[tuple[str, str | None, str, float | None, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "cluster_id": pl.String,
            "source_domain": pl.String,
            "source_domain_status": pl.String,
            "tone": pl.Float64,
            "n_entities": pl.Int32,
        },
        orient="row",
    )


def test_edges_are_ordered_pairs():
    tables = build_source_domains(
        _records(
            [
                ("c1", "b.com", "observed", 1.0, 1),
                ("c1", "a.com", "observed", 2.0, 2),
                ("c1", "a.com", "observed", 3.0, 3),
                ("c2", "a.com", "observed", 1.0, 1),
                ("c2", "b.com", "observed", 1.0, 1),
                ("c2", "c.com", "observed", 5.0, 1),
            ]
        )
    )

    edges = tables.edges.rows()
    assert edges == [("a.com", "b.com", 2), ("a.com", "c.com", 1), ("b.com", "c.com", 1)]
    assert tables.families["family_id"].to_list() == ["a.com", "b.com", "c.com"]
    assert tables.families["provenance"].unique().to_list() == ["singleton_by_default"]


def test_unknown_domains_excluded_from_counts():
    tables = build_source_domains(
        _records(
            [
                ("c1", "a.com", "observed", 1.0, 1),
                ("c1", "b.com", "reconstructed_from_url", 2.0, 1),
                ("c1", None, "unknown_not_a_domain", 3.0, 1),
                ("c1", "x.com", "unknown_not_attempted", 4.0, 1),
            ]
        )
    )

    assert tables.cluster_counts.to_dicts() == [
        {
            "cluster_id": "c1",
            "raw_article_count": 4,
            "deduplicated_source_family_count": 2,
            "independent_source_units": 2,
            "unknown_domain_rows": 2,
        }
    ]
    assert tables.families["member_domain"].to_list() == ["a.com", "b.com"]
    assert tables.edges.rows() == [("a.com", "b.com", 1)]


def test_syndication_merge_is_transitive_within_cluster():
    tables = build_source_domains(
        _records(
            [
                # a~b share (1.0, 1); b~c share (2.0, 2): one unit {a, b, c}, two merges.
                ("c1", "a.com", "observed", 1.0, 1),
                ("c1", "b.com", "observed", 1.0, 1),
                ("c1", "b.com", "observed", 2.0, 2),
                ("c1", "c.com", "observed", 2.0, 2),
                ("c1", "d.com", "observed", 9.0, 9),
                # The same pair in another cluster is merged there independently.
                ("c2", "a.com", "observed", 1.0, 1),
                ("c2", "c.com", "observed", 5.0, 1),
                # Null tone groups together; a different n_entities does not.
                ("c3", "a.com", "observed", None, 1),
                ("c3", "b.com", "observed", None, 1),
                ("c3", "c.com", "observed", None, 2),
            ]
        )
    )

    counts = {r["cluster_id"]: r for r in tables.cluster_counts.to_dicts()}
    assert counts["c1"]["deduplicated_source_family_count"] == 4
    assert counts["c1"]["independent_source_units"] == 2
    assert counts["c2"]["independent_source_units"] == 2
    assert counts["c3"]["deduplicated_source_family_count"] == 3
    assert counts["c3"]["independent_source_units"] == 2
