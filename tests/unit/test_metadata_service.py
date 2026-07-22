from app.services.metadata_service import MetadataService

def test_extract_entities_valid_names():
    sample_text = (
        "Arthur Vance. "
        "Dr. Elizabeth Swann. "
        "Arthur Vance. "
        "Elizabeth Swann. "
        "Arthur Vance. "
        "Elizabeth Swann. "
        "Arthur Vance."
    )
    entities = MetadataService.extract_entities(sample_text)
    assert "Arthur Vance" in entities
    assert "Elizabeth Swann" in entities

def test_extract_entities_below_frequency_threshold():
    sample_text = "Unique Person name mentioned only once."
    entities = MetadataService.extract_entities(sample_text)
    assert "Unique Person" not in entities

def test_extract_metadata_structure():
    sample_text = "Arthur Vance. Arthur Vance. Arthur Vance. Arthur Vance signed the document."
    meta = MetadataService.extract_metadata(sample_text)
    assert "entities" in meta
    assert "entity_count" in meta
    assert meta["entities"] == ["Arthur Vance"]
    assert meta["entity_count"] == 1
