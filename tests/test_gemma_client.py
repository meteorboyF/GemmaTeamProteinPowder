import json

import config
import gemma_client


def test_primary_success_sets_status_and_caches(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        gemma_client,
        "_call_cloud_with_retry",
        lambda model_id, prompt, image, json_mode: '{"ok": true}',
    )

    output = gemma_client.generate("test", image=b"\xff\xd8\xffx", json_mode=True)

    assert json.loads(output)["ok"] is True
    assert gemma_client.get_status().source == gemma_client.SOURCE_CLOUD_PRIMARY
    assert gemma_client.get_cached_success("extraction")["text"] == output


def test_task_caches_do_not_overwrite_each_other():
    gemma_client.cache_success("extraction result", kind="extraction")
    gemma_client.cache_success("explanation result", kind="generic")

    assert gemma_client.get_cached_success("extraction")["text"] == "extraction result"
    assert gemma_client.get_cached_success("generic")["text"] == "explanation result"


def test_primary_failure_uses_cloud_fallback(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    def call(model_id, *_args):
        if model_id == config.GEMMA_PRIMARY_MODEL:
            raise gemma_client.PermanentModelError("primary unavailable")
        return "fallback answer"

    monkeypatch.setattr(gemma_client, "_call_cloud_with_retry", call)

    output = gemma_client.generate("test", image=b"\xff\xd8\xffx")

    assert output == "fallback answer"
    assert gemma_client.get_status().source == gemma_client.SOURCE_CLOUD_FALLBACK


def test_total_failure_is_structured_and_never_raises(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "ENABLE_LOCAL_FALLBACK", False)

    output = gemma_client.generate("test")

    assert gemma_client.is_error(output)
    assert gemma_client.parse_error(output)["attempts"]


def test_multimodal_cloud_call_is_not_retried(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append(True)
        raise gemma_client.TransientModelError("timeout")

    monkeypatch.setattr(gemma_client, "_call_cloud", fail)

    try:
        gemma_client._call_cloud_with_retry(
            "model",
            "prompt",
            [b"view-one", b"view-two"],
            True,
        )
    except gemma_client.TransientModelError:
        pass

    assert len(calls) == 1
