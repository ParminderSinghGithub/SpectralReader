from app.generation.token_budget import TokenBudgetManager
from app.generation.prompt_builder import PromptBuilder
from app.generation.context_builder import ContextBuilder

def test_token_budget_manager():
    mgr = TokenBudgetManager(max_context_chars=100)
    passages = ["Short passage 1.", "Short passage 2.", "Very long passage that exceeds the max context chars budget."]
    budgeted, count = mgr.enforce_char_budget(passages)
    assert count <= 100
    assert len(budgeted) >= 1

def test_prompt_builder():
    prompt = PromptBuilder.build_qa_prompt(["Passage A", "Passage B"], "What is A?")
    assert "Passage A" in prompt
    assert "What is A?" in prompt
    assert "SpectralReader" in prompt

def test_context_builder_deduplication():
    builder = ContextBuilder(max_context_chars=2000)
    passages, count = builder.prepare_context(["Duplicate text", "Duplicate text", "Unique text"])
    assert len(passages) == 2
    assert passages == ["Duplicate text", "Unique text"]
