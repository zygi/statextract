import pydantic
from typing import Literal
import anthropic




class PaperMD(pydantic.BaseModel):
    title: str
    author_names: list[str]
    first_author: str
    id: str
    doi: tuple[str, str] | None
    type: str | None
    
    
    # @pydantic.computed_field
    # @property  
    # def first_author_id(self) -> str:
    #     return self.id.split('/')[-1]

    
class FullPaperMD(PaperMD):
    full: dict
    
    def drop_full(self):
        return PaperMD(**self.model_dump(exclude={"full"}))

# Tailedness = Literal["left", "right", "two"]

# class StudentT(pydantic.BaseModel):
#     mean: float
#     std: float
#     df: int
    
    
# class Z(pydantic.BaseModel):
#     mean: float
#     std: float

# class PearsonChi2(pydantic.BaseModel):
#     counts: list[int]
    
    
    

###

ComparisonType = Literal["goodness_of_fit", "independence", "homogeneity"]