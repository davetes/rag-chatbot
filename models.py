from pydantic import BaseModel

class DocumentInput(BaseModel):
    document_id: str
    text: str

class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]