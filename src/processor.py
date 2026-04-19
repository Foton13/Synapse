import os
from typing import List, Tuple, Union, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

class Relation(BaseModel):
    source: str = Field(description="The starting entity of the relationship")
    relation: str = Field(description="The type of relationship (e.g., 'related_to', 'part_of', 'implemented_with')")
    target: str = Field(description="The ending entity of the relationship")

class KnowledgeGraph(BaseModel):
    entities: List[str] = Field(description="List of unique entities found in the text")
    relations: List[Relation] = Field(description="List of relationships between entities")

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o")
    else:
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))

def process_note(content: str) -> Union[KnowledgeGraph, Exception]:
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
    
    prompt = PromptTemplate(
        template="Analyze the following text and extract key entities and their relationships.\n{format_instructions}\n\nText:\n{text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        return chain.invoke({"text": content})
    except Exception as e:
        return e
