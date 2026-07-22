from typing import List
from app.services.metadata_service import MetadataService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class QAService:
    @staticmethod
    def answer_question(question: str, docs: List[str], tokenizer, model) -> str:
        """Filter context documents containing character info and generate answer using model."""
        character_passages = []
        for doc in docs:
            if any(char in doc for char in MetadataService.extract_character_info(doc)):
                character_passages.append(doc)
        if not character_passages:
            return "I couldn't find character information in the document."
        context = "\n".join(character_passages[:3])
        prompt = f"""Analyze this literary excerpt and answer the question about characters.
    
    Excerpt:
    {context[:settings.MAX_PROMPT_CONTEXT_CHARS]}
    
    Question: {question}
    
    Answer in complete sentences, identifying characters by name when possible:"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_length=settings.MAX_GEN_LENGTH,
            temperature=settings.GEN_TEMPERATURE,
            top_p=settings.GEN_TOP_P,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
