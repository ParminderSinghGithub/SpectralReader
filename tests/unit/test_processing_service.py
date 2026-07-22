from app.services.processing_service import ProcessingService

def test_process_text_with_chapter_boundaries():
    chapter1 = "\nCHAPTER I\n" + ("Word " * 30 + "\n\n") * 5
    chapter2 = "\nCHAPTER II\n" + ("Text " * 30 + "\n\n") * 5
    sample_text = chapter1 + chapter2
    chunks = ProcessingService.process_text(sample_text)
    assert len(chunks) >= 2

def test_process_text_fallback_splitter():
    sample_text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
    chunks = ProcessingService.process_text(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_process_text_hyphen_cleaning():
    sample_text = "Multi- line text with hypo- thesis split across line break."
    chunks = ProcessingService.process_text(sample_text)
    assert "hypothesis" in chunks[0]
