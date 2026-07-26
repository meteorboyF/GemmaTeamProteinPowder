import tts


def test_chunk_text_preserves_content_and_limits_typical_chunks():
    text = "সকালে একটি ওষুধ নিন। রাতে আরেকটি ওষুধ নিন। পানি পান করুন।"

    chunks = tts.chunk_text(text, max_chars=35)

    assert len(chunks) >= 2
    assert " ".join(chunks).replace("  ", " ") == text


def test_empty_speech_returns_error_without_network():
    result = tts.speak("   ")

    assert not result.ok
    assert result.error
