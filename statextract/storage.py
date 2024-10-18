import asyncio
import pathlib
import typing
import pydantic
from statextract.agent.stats_extractor_agent import FullExtractionResult
from statextract.prefilter import ClassifyNHST
import aiodbm
# import cloudpickle

# _prefilter_db = dbm.open('data/dbs/prefilter.db', 'c')


class PrefilterReport(pydantic.BaseModel):
    regex_match: bool
    ai_response: ClassifyNHST | None
    
class TypedDB[T: pydantic.BaseModel]:
    def __init__(self, path: str, TT: type[T]):
        self.path = pathlib.Path(path)
        # create dirs if not exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # self.db = aiodbm.open(path, 'c')
        self.TT = TT

    @property
    def db(self) -> aiodbm.Database:
        return aiodbm.open(self.path, 'c')

    async def get(self, k: str) -> T | None:
        async with self.db as db:
            v = await db.get(k)
            if not v:
                return None
            return self.TT.model_validate_json(v)

    async def set(self, k: str, v: T):
        async with self.db as db:
            await db.set(k, v.model_dump_json())

    async def get_keys(self) -> list[bytes]:
        async with self.db as db:
            return await db.keys()

_prefilter_report_db: TypedDB[PrefilterReport] | None = None
def prefilter_report_db() -> TypedDB[PrefilterReport]:
    global _prefilter_report_db
    if _prefilter_report_db is None:
        _prefilter_report_db = TypedDB('data/dbs/prefilter.db', PrefilterReport)
    return _prefilter_report_db

_full_extraction_result_db: TypedDB[FullExtractionResult] | None = None
def full_extraction_result_db() -> TypedDB[FullExtractionResult]:
    global _full_extraction_result_db
    if _full_extraction_result_db is None:
        _full_extraction_result_db = TypedDB('data/dbs/full_extraction_result.db', FullExtractionResult)
    return _full_extraction_result_db

    # @classmethod
    # def save(cls, k: str, v: "PrefilterReport"):
    #     _prefilter_db.set(k, v)

    # @classmethod
    # def load(cls, k: str) -> "PrefilterReport | None":
    #     obj = _prefilter_db.get(k)
    #     if not obj:
    #         return None
    #     assert isinstance(obj, PrefilterReport)
    #     return obj
