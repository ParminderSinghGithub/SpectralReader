from typing import List
from app.services.metadata_service import MetadataService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class QAService:
    @staticmethod
    def answer_question(question: str, docs: List[str], tokenizer, model) -> str:
        """Filter context passages containing document entity info and generate answer using model."""
        character_passages = []
        for doc in docs:
            if any(entity in doc for entity in MetadataService.extract_entities(doc)):
                character_passages.append(doc)
        if not character_passages:
            return "I couldn't find relevant information in the document."
        context = "\n".join(character_passages[:3])
        prompt = f"""Analyze this document excerpt and answer the question.
    
    Excerpt:
    {context[:settings.MAX_PROMPT_CONTEXT_CHARS]}
    
    Question: {question}
    
    Answer in complete sentences based on the excerpt:"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_length=settings.MAX_GEN_LENGTH,
            temperature=settings.GEN_TEMPERATURE,
            top_p=settings.GEN_TOP_P,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
