from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from typing import List


class State(TypedDict):
    """State for the SQL retrieval process
    question: The question to ask the model.
    query: The generated SQL query.
    result: The result of the SQL query.
    answer: The answer to the question.
    """

    question: str
    documents: List[str]
    generation: str
    chat_history: List[dict]
    
class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")
