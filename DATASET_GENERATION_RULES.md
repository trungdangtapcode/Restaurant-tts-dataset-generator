# 🎲 Randomization Rules Overview

In this project, the system applies multiple layers of randomized rules—from lexical phrasing to acoustic variation—to ensure the generated TTS dataset feels perfectly natural, just like how people speak when ordering in a real restaurant.

## 1. 📝 Lexical Classifiers & Quantifiers
The generator automatically monitors keywords in the item names to select the culturally correct Vietnamese classifiers:

*   **🍹 Drinks (Soda, Beer, Tea...):**
    *   *Triggers:* `7up`, `aquafina`, `bia`, `coca`, `pepsi`, `sting`, `tiger`, `nước`, `trà`
    *   *If Quantity = 1:* Randomly selects `["", "1 lon ", "một lon ", "1 chai ", "một chai ", "1 "]`
    *   *If Quantity > 1:* Randomly selects `["{Qty} ", "{Qty} lon ", "{Qty} chai "]`

*   **🥖 Bakery/Noodles:**
    *   *Triggers:* `bánh mì`, `vắt mì`
    *   *If Quantity = 1:* Randomly selects `["", "1 ổ ", "một ổ ", "1 "]`
    *   *If Quantity > 1:* Randomly selects `["{Qty} ", "{Qty} ổ "]`

*   **🍲 Standard Foods (Default):**
    *   *If Quantity = 1:* Randomly selects `["", "1 ", "một ", "một phần ", "1 phần ", "1 dĩa "]`
    *   *If Quantity > 1:* Randomly selects `["{Qty} ", "{Qty} phần ", "{Qty} dĩa "]`

## 2. 🔀 Item Shuffling (Item Order Randomization)
Real humans rarely sequence their orders identically every time. To mirror this:
*   **Position Mutation:** The internal array storing the list of foods/drinks inside a single bill is completely shuffled at every generation loop.
*   **Conjunction Connectors:** For bridging items together (between the second-to-last and the final item), the system draws randomly from 4 different conjunctions: `[" và ", ", với ", ", ", " thêm "]`.

## 3. 📍 Table Number Placement 
The "Table XX" phrasing has a 50% probability of being spoken at the beginning, and a 50% probability of being placed at the tail end of the sentence:
*   **Prefix Cases:** Randomly draws from `["Bàn số {X}, ", "Cho bàn {X}, ", "Bàn {X} gọi ", "Ghi cho bàn {X} ", "Bàn {X} nà, "]`
*   **Suffix Cases:** Randomly draws from `[" cho bàn số {X}", " bàn {X} nhé", ", bàn {X}"]`

## 4. 🗣️ Sentence Ending Particles
To convey the realistic casual vibe of street food stalls or pubs, the text engine picks a random particle right at the termination of the string:
*   **Ending particles:** `[".", " nhé.", " nha.", " nha em.", " nha quán."]`

## 5. 🎙️ Random AI Regional Speakers (Acoustic Variation)
Located inside the `TTSEngineWorker` (`tts_worker.py`), each independent variation snippet is passed across different AI character voices:

*   **List of Speakers available under ValtecTTS:** 
    *   `NF`: Northern Female
    *   `SF`: Southern Female
    *   `NM1`: Northern Male 1
    *   `NM2`: Northern Male 2
    *   `SM`: Southern Male
*   **Algorithm Rule:** Driven by `speaker = random.choice(self.speakers)`, this enforces a highly scattered variance mixing both accents (Northern vs Southern) and distinct gender traits seamlessly over thousands of iterations.
