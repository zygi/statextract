import pydantic
from typing import Literal
import anthropic

Tailedness = Literal["left", "right", "two"]

class StudentT(pydantic.BaseModel):
    mean: float
    std: float
    df: int
    
    
class Z(pydantic.BaseModel):
    mean: float
    std: float

class PearsonChi2(pydantic.BaseModel):
    counts: list[int]
    
    
    

###

ComparisonType = Literal["goodness_of_fit", "independence", "homogeneity"]