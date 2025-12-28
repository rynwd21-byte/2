from datetime import datetime
from typing import Optional, Dict
from sqlalchemy import Table, Column, String, Float, MetaData, select, insert, update
from .db import engine
_METADATA = MetaData()
model_params_table = Table("model_params", _METADATA, Column("name", String, primary_key=True),
    Column("value", Float, nullable=False), Column("updated_at", String, nullable=False))
def ensure_table(): _METADATA.create_all(bind=engine, tables=[model_params_table])
def set_param(name: str, value: float) -> None:
    ensure_table(); now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        row = conn.execute(select(model_params_table).where(model_params_table.c.name==name)).first()
        if row: conn.execute(update(model_params_table).where(model_params_table.c.name==name).values(value=value, updated_at=now))
        else: conn.execute(insert(model_params_table).values(name=name, value=value, updated_at=now))
def get_params() -> Dict[str, float]:
    ensure_table(); 
    with engine.begin() as conn:
        rows = conn.execute(select(model_params_table.c.name, model_params_table.c.value)).all()
        return {r[0]: float(r[1]) for r in rows}
def get_param(name: str, default: Optional[float] = None) -> Optional[float]:
    ensure_table();
    with engine.begin() as conn:
        row = conn.execute(select(model_params_table.c.value).where(model_params_table.c.name==name)).first()
        return (float(row[0]) if row else default)
