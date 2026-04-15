import random
from typing import List
from domain import OrderItem, Bill

class CategoryRules:
    """Rules to decide the unit (lon, chai, phần, dĩa) based on item names"""
    DRINKS = ["7up", "aquafina", "bia", "coca", "pepsi", "sting", "tiger", "nước", "trà"]
    BREAD = ["bánh mì", "vắt mì"]
    
    @classmethod
    def get_unit_for(cls, item_name: str, quantity: int) -> str:
        name_lower = item_name.lower()
        
        # Is drink? (No "phần" for drinks)
        if any(drink in name_lower for drink in cls.DRINKS):
            if quantity == 1:
                return random.choice(["", "1 lon ", "một lon ", "1 chai ", "một chai ", "1 "])
            else:
                return random.choice([f"{quantity} ", f"{quantity} lon ", f"{quantity} chai "])
                
        # Is bread/noodles?
        if any(bread in name_lower for bread in cls.BREAD):
            if quantity == 1:
                return random.choice(["", "1 ổ ", "một ổ ", "1 "])
            else:
                return random.choice([f"{quantity} ", f"{quantity} ổ "])
        
        # Default food (can use "phần", "dĩa", etc)
        if quantity == 1:
            return random.choice(["", "1 ", "một ", "một phần ", "1 phần ", "1 dĩa "])
        else:
            return random.choice([f"{quantity} ", f"{quantity} phần ", f"{quantity} dĩa "])


class BillSentenceFormatter:
    def format_quantity(self, item: OrderItem) -> str:
        unit_prefix = CategoryRules.get_unit_for(item.name, item.quantity)
        return f"{unit_prefix}{item.name}"

    def randomize_and_join_items(self, items: List[OrderItem]) -> str:
        if not items:
            return ""
        copied_items = list(items)
        random.shuffle(copied_items)
        phrases = [self.format_quantity(item) for item in copied_items]
        
        if len(phrases) == 1:
            return phrases[0]

        joined_str = ", ".join(phrases[:-1])
        connector = random.choice([" và ", ", với ", ", ", " thêm "])
        return f"{joined_str}{connector}{phrases[-1]}"

    def place_table_number(self, items_phrase: str, table_number: int) -> str:
        place_first = random.choice([True, False])
        if place_first:
            prefixes = [f"Bàn số {table_number}, ", f"Cho bàn {table_number}, ", f"Bàn {table_number} gọi ", f"Ghi cho bàn {table_number} ", f"Bàn {table_number} nà, "]
            return f"{random.choice(prefixes)}{items_phrase}"
        else:
            suffixes = [f" cho bàn số {table_number}", f" bàn {table_number} nhé", f", bàn {table_number}"]
            return f"{items_phrase}{random.choice(suffixes)}"

    def finalize_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence: return sentence
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", ",")):
            sentence += random.choice([".", " nhé.", " nha.", " nha em.", " nha quán."])
        return sentence

    def generate(self, bill: Bill) -> str:
        if not bill.items: return ""
        items_phrase = self.randomize_and_join_items(bill.items)
        sentence = self.place_table_number(items_phrase, bill.table_number)
        return self.finalize_sentence(sentence)
