"""Tests for src/llm/narrative.py — all Ollama calls are mocked."""
from unittest.mock import AsyncMock

import pytest

from src.llm.narrative import (
    ChartSpec,
    NarrativeResult,
    _DEFAULT_CHART,
    build_prompt,
    generate_narrative,
    generate_narrative_stream,
    parse_narrative,
)
from src.llm.ollama_client import OllamaError

# ── Shared fixtures ───────────────────────────────────────────────────────────

_CTX: dict = {
    "prediction_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "point_estimate": 125_000.0,
    "range_low": 110_000.0,
    "range_high": 140_000.0,
    "currency": "USD",
    "model_mae": 31_500.0,
    "features": {
        "experience_level": 2,
        "employment_type": 3,
        "remote_ratio": 100,
        "company_size": 1,
        "job_family": 2,
        "location_region": 3,
        "is_us_company": 1,
    },
}

_WELL_FORMED_RESPONSE = """\
1. SUMMARY
Senior data scientists at US companies command a predicted salary of $125,000,
placing them in the upper tier of the global data science market.

2. UNCERTAINTY
The model predicts a salary of $125,000. Most peers with this profile earn
between $110,000 and $140,000 (peer-group Q25–Q75). The model carries a
typical absolute error of ± $31,500 (MAE).

3. INSIGHTS
• Experience level (Senior) is the strongest driver — senior roles earn ~40% more.
• US company flag adds approximately $25,000 versus non-US equivalents.
• Full remote work correlates with higher pay in this dataset.

4. COMPARISON
This estimate is above the dataset median of $115,000. North America is the
top-paying region, and this profile lands squarely there.

5. CHART
[CHART]
type: bar
title: Salary by Experience Level
x_label: Experience Level
y_label: Average Salary (USD)
data_key: salary_by_experience
[/CHART]

6. RECOMMENDATION
Target senior IC or staff engineer tracks at US-headquartered companies with
established remote policies.
"""


# ── build_prompt ──────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_point_estimate(self) -> None:
        assert "125,000" in build_prompt(_CTX)

    def test_contains_range_bounds(self) -> None:
        prompt = build_prompt(_CTX)
        assert "110,000" in prompt
        assert "140,000" in prompt

    def test_contains_mae(self) -> None:
        assert "31,500" in build_prompt(_CTX)

    def test_contains_human_readable_feature_label(self) -> None:
        # experience_level=2 must render as the human-readable label
        assert "Senior (SE)" in build_prompt(_CTX)

    def test_missing_key_raises(self) -> None:
        ctx = {k: v for k, v in _CTX.items() if k != "point_estimate"}
        with pytest.raises(KeyError):
            build_prompt(ctx)


# ── parse_narrative ───────────────────────────────────────────────────────────


class TestParseNarrative:
    def _parse(self) -> NarrativeResult:
        return parse_narrative(_WELL_FORMED_RESPONSE)

    def test_summary_extracted(self) -> None:
        assert "125,000" in self._parse().summary

    def test_uncertainty_extracted(self) -> None:
        assert self._parse().uncertainty != ""

    def test_insights_is_list(self) -> None:
        insights = self._parse().insights
        assert isinstance(insights, list)
        assert len(insights) >= 1

    def test_insights_bullet_markers_stripped(self) -> None:
        for insight in self._parse().insights:
            assert not insight.startswith("•")

    def test_recommendation_extracted(self) -> None:
        assert "senior" in self._parse().recommendation.lower()

    def test_chart_spec_type(self) -> None:
        assert self._parse().chart_spec.type == "bar"

    def test_chart_spec_title(self) -> None:
        assert self._parse().chart_spec.title == "Salary by Experience Level"

    def test_chart_spec_data_key(self) -> None:
        assert self._parse().chart_spec.data_key == "salary_by_experience"

    def test_missing_chart_block_returns_default(self) -> None:
        raw = _WELL_FORMED_RESPONSE.replace("[CHART]", "").replace("[/CHART]", "")
        result = parse_narrative(raw)
        assert result.chart_spec.type == _DEFAULT_CHART.type
        assert result.chart_spec.data_key == _DEFAULT_CHART.data_key

    def test_pipe_delimited_chart_type_strips_to_first(self) -> None:
        raw = _WELL_FORMED_RESPONSE.replace("type: bar", "type: bar | histogram | scatter")
        assert parse_narrative(raw).chart_spec.type == "bar"


# ── generate_narrative ────────────────────────────────────────────────────────


class TestGenerateNarrative:
    async def test_returns_narrative_result_and_raw_string(self, mocker) -> None:
        mocker.patch("src.llm.narrative.generate", return_value=_WELL_FORMED_RESPONSE)
        narrative, raw = await generate_narrative(_CTX)
        assert isinstance(narrative, NarrativeResult)
        assert raw == _WELL_FORMED_RESPONSE

    async def test_calls_generate_with_prompt_containing_estimate(self, mocker) -> None:
        mock_gen = mocker.patch("src.llm.narrative.generate", return_value=_WELL_FORMED_RESPONSE)
        await generate_narrative(_CTX)
        mock_gen.assert_called_once()
        called_prompt: str = mock_gen.call_args[0][0]
        assert "125,000" in called_prompt

    async def test_propagates_ollama_error(self, mocker) -> None:
        mocker.patch("src.llm.narrative.generate", side_effect=OllamaError("timeout"))
        with pytest.raises(OllamaError):
            await generate_narrative(_CTX)


# ── generate_narrative_stream ─────────────────────────────────────────────────


class TestGenerateNarrativeStream:
    @staticmethod
    async def _collect(gen) -> list[str]:
        tokens: list[str] = []
        async for token in gen:
            tokens.append(token)
        return tokens

    async def test_yields_tokens_from_stream(self, mocker) -> None:
        async def _fake_stream(prompt):
            yield "Hello "
            yield "world"

        mocker.patch("src.llm.narrative.generate_stream", side_effect=_fake_stream)
        mocker.patch("src.database.crud.insert_narrative", new=AsyncMock())

        tokens = await self._collect(generate_narrative_stream(_CTX))
        assert tokens == ["Hello ", "world"]

    async def test_persists_after_stream_exhausted(self, mocker) -> None:
        async def _fake_stream(prompt):
            yield _WELL_FORMED_RESPONSE

        mocker.patch("src.llm.narrative.generate_stream", side_effect=_fake_stream)
        mock_insert = mocker.patch("src.database.crud.insert_narrative", new=AsyncMock())

        await self._collect(generate_narrative_stream(_CTX))

        mock_insert.assert_called_once()
        kwargs = mock_insert.call_args.kwargs
        assert kwargs["prediction_id"] == _CTX["prediction_id"]
        assert isinstance(kwargs["narrative"], NarrativeResult)
        assert isinstance(kwargs["raw_response"], str)

    async def test_ollama_error_yields_error_sentinel(self, mocker) -> None:
        async def _error_stream(prompt):
            raise OllamaError("Connection refused")
            yield  # pragma: no cover — makes this an async generator

        mocker.patch("src.llm.narrative.generate_stream", side_effect=_error_stream)

        tokens = await self._collect(generate_narrative_stream(_CTX))
        assert len(tokens) == 1
        assert tokens[0].startswith("[ERROR]")

    async def test_missing_context_keys_yields_error_sentinel(self) -> None:
        ctx = {k: v for k, v in _CTX.items() if k != "prediction_id"}
        tokens = await self._collect(generate_narrative_stream(ctx))
        assert len(tokens) == 1
        assert tokens[0].startswith("[ERROR]")
