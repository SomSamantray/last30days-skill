"""Pipeline YouTube dispatch: thinness-floor ScrapeCreators search backstop (#977).

The SC YouTube search backstop must fire when yt-dlp returns fewer than
``pipeline._YT_SC_MIN_ITEMS`` items (not only zero), and its results merge with
the yt-dlp items — never discarding them (R2).
"""

from unittest import mock

from lib import env, pipeline, schema


def _subquery():
    return schema.SubQuery(
        label="t", search_query="youtube topic", ranking_query="youtube topic",
        sources=["youtube"],
    )


def _runtime():
    return schema.ProviderRuntime(
        reasoning_provider="mock", planner_model="mock", rerank_model="mock",
    )


def _item(vid, title="video"):
    return {
        "video_id": vid,
        "title": title,
        "url": f"https://www.youtube.com/watch?v={vid}",
    }


def _run(config, free_items, sc_items=None, sc_raises=False):
    """Drive the real pipeline youtube branch with the SC backstop mocked."""
    sc_mock = mock.Mock()
    if sc_raises:
        sc_mock.side_effect = Exception("sc down")
    else:
        sc_mock.return_value = {"items": sc_items or []}
    with mock.patch("lib.pipeline.which", return_value="/usr/local/bin/yt-dlp"), \
         mock.patch(
             "lib.pipeline.youtube_yt.search_and_transcribe",
             return_value={"items": free_items},
         ), \
         mock.patch("lib.pipeline.youtube_yt.search_youtube_sc", sc_mock), \
         mock.patch("lib.pipeline.youtube_yt.enrich_with_comments") as enrich:
        items, artifact = pipeline._retrieve_stream(
            topic="youtube topic", subquery=_subquery(), source="youtube",
            config=config, depth="quick",
            date_range=("2026-07-17", "2026-08-16"),
            runtime=_runtime(), mock=False,
        )
    return items, artifact, sc_mock, enrich


class TestThinnessFloorBackstop:
    KEY = {"SCRAPECREATORS_API_KEY": "k"}

    def test_below_floor_fires_backstop_and_merges(self, capsys):
        # 1 free item (< floor) -> backstop fires; SC item that duplicates the
        # free video_id is not double-listed; free item stays first.
        free = [_item("a")]
        sc = [_item("b"), _item("a", title="dupe"), _item("c")]
        items, _, sc_mock, _ = _run(self.KEY, free, sc)
        sc_mock.assert_called_once()
        assert [i["video_id"] for i in items] == ["a", "b", "c"]
        assert "[YouTube]" in capsys.readouterr().err

    def test_at_floor_skips_backstop(self):
        items, _, sc_mock, _ = _run(self.KEY, [_item("a"), _item("b"), _item("c")])
        sc_mock.assert_not_called()
        assert len(items) == 3

    def test_zero_items_fires_backstop_silently(self, capsys):
        items, _, sc_mock, _ = _run(self.KEY, [], [_item("z")])
        sc_mock.assert_called_once()
        assert [i["video_id"] for i in items] == ["z"]
        assert "[YouTube]" not in capsys.readouterr().err

    def test_empty_backstop_preserves_free_items(self):
        free = [_item("a")]
        items, _, sc_mock, _ = _run(self.KEY, free, [])
        sc_mock.assert_called_once()
        assert [i["video_id"] for i in items] == ["a"]

    def test_backstop_throw_preserves_free_items(self):
        free = [_item("a")]
        items, _, sc_mock, _ = _run(self.KEY, free, sc_raises=True)
        sc_mock.assert_called_once()
        assert [i["video_id"] for i in items] == ["a"]

    def test_keyless_never_calls_sc(self):
        items, _, sc_mock, _ = _run({}, [_item("a")])
        sc_mock.assert_not_called()
        assert [i["video_id"] for i in items] == ["a"]

    def test_no_ytdlp_still_falls_back_to_sc(self):
        # result None (yt-dlp branch not run) is the pre-existing trigger.
        with mock.patch("lib.pipeline.which", return_value=None), \
             mock.patch("lib.pipeline.youtube_yt.search_youtube_sc",
                        return_value={"items": [_item("z")]}) as sc_mock, \
             mock.patch("lib.pipeline.youtube_yt.enrich_with_comments"):
            items, _ = pipeline._retrieve_stream(
                topic="youtube topic", subquery=_subquery(), source="youtube",
                config=self.KEY, depth="quick",
                date_range=("2026-07-17", "2026-08-16"),
                runtime=_runtime(), mock=False,
            )
        sc_mock.assert_called_once()
        assert [i["video_id"] for i in items] == ["z"]

    def test_bot_gate_failure_preserved_after_rescue(self):
        free = [_item("a")]
        with mock.patch("lib.pipeline.which", return_value="/usr/local/bin/yt-dlp"), \
             mock.patch(
                 "lib.pipeline.youtube_yt.search_and_transcribe",
                 return_value={
                     "items": free,
                     "error": "Sign in to confirm you're not a bot",
                 },
             ), \
             mock.patch("lib.pipeline.youtube_yt.search_youtube_sc",
                        return_value={"items": [_item("b"), _item("c"), _item("d")]}), \
             mock.patch("lib.pipeline.youtube_yt.enrich_with_comments"):
            items, artifact = pipeline._retrieve_stream(
                topic="youtube topic", subquery=_subquery(), source="youtube",
                config=self.KEY, depth="quick",
                date_range=("2026-07-17", "2026-08-16"),
                runtime=_runtime(), mock=False,
            )
        assert len(items) == 4
        assert artifact["_source_outcome"]["state"] == schema.RATE_LIMITED

    def test_comments_enrichment_runs_on_rescued_items(self):
        free = [_item("a")]
        items, _, _, enrich = _run(self.KEY, free, [_item("b"), _item("c")])
        enrich.assert_called_once_with(
            items, token=self.KEY["SCRAPECREATORS_API_KEY"],
        )
