from dataclasses import dataclass
from typing import List

@dataclass
class OrderItem:
    name: str
    quantity: int

@dataclass
class Bill:
    bill_id: str
    table_number: int
    items: List[OrderItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'Bill':
        items = [
            OrderItem(name=i.get("name", ""), quantity=i.get("quantity", 1))
            for i in data.get("items", [])
        ]
        return cls(
            bill_id=data.get("billId", "unknown"),
            table_number=data.get("tableNumber", 0),
            items=items
        )
