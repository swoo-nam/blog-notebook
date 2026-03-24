## 개요

구글의 생성형 AI API 서비스에 대한 통합 인터페이스 (Google Gen AI SDK) 사용.

구글 클라우드 교육에 사용되는 노트북 코드를 기반으로 작성.

### 목표

Python용 Google Gen AI SDK의 주요 기능

- 텍스트 프롬프트 전송
- 멀티모달 프롬프트 전송
- 시스템 지시 설정
- 모델 매개변수 구성
- 안전 필터 구성
- 다중 턴(multi-turn) 채팅 시작
- 생성된 출력 제어
- 콘텐츠 스트림 생성
- 비동기 요청 전송
- 토큰 개수 세기 및 계산
- 함수 호출
- 컨텍스트 캐싱 사용
- 텍스트 임베딩 가져오기

## 시작


```python
import datetime
import os
from dotenv import load_dotenv

from google import genai
from google.genai.types import (
    CreateBatchJobConfig,
    CreateCachedContentConfig,
    EmbedContentConfig,
    FunctionDeclaration,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    Tool,
)

load_dotenv()

PROJECT_ID = os.getenv("VERTEXAI_PROJECT_ID")
LOCATION = os.getenv("VERTEXAI_LOCATION")
credential_path = os.getenv("VERTEXAI_CREDENTIALS_PATH")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

MODEL_ID = "gemini-2.5-flash"
```

## 1. 텍스트 프롬프트 전송


```python
from IPython.display import Markdown, display

response = client.models.generate_content(
    model=MODEL_ID, contents="What's the largest planet in our solar system?"
)

display(Markdown(response.text))
```


The largest planet in our solar system is **Jupiter**.


## 2. 멀티모달 프롬프트 전송


```python
from PIL import Image
import requests

image = Image.open(
    requests.get(
        "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
        stream=True,
    ).raw
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture.",
    ],
)

print(response.text)
```

    ## Unlock Your Weekday Warrior: Delicious & Easy Meal Prep Done Right
    
    Ever stared into your fridge on a Monday morning, dreading the thought of another sad desk lunch or a rushed, unhealthy takeout order? You're not alone! But what if I told you there's a simple, colorful, and utterly delicious solution? Just take a look at these vibrant, perfectly portioned glass containers!
    
    This image isn't just a feast for the eyes; it's an inspiring snapshot of meal prep magic. These beautifully organized meals are the secret weapon for anyone looking to save time, eat healthier, and reduce stress throughout their busy week.
    
    **Why Meal Prep? Let us count the ways:**
    
    1.  **Time Saver:** Imagine grabbing a ready-to-eat, nutritious meal from your fridge in seconds. No cooking, no cleaning during the week!
    2.  **Healthier Choices:** When healthy options are readily available, you're less likely to fall victim to impulse buys or sugary snacks. These bowls are packed with lean protein, complex carbs, and a rainbow of veggies!
    3.  **Budget-Friendly:** Eating out adds up quickly. Meal prepping allows you to buy ingredients in bulk, minimize waste, and control your spending.
    4.  **Stress Reduction:** The mental load of deciding what to eat each day is real. With meal prep, that decision is already made. One less thing to worry about!
    
    **What's in these gorgeous bowls?**
    
    We're looking at a fantastic, balanced meal featuring:
    
    *   **Fluffy Rice:** A perfect base, likely brown or white rice, providing essential energy.
    *   **Tender Chicken:** Cubed and glistening, possibly coated in a savory teriyaki or soy-ginger glaze, sprinkled with sesame seeds and green onions for extra flavor and visual appeal.
    *   **Vibrant Veggies:** Crisp broccoli florets, along with julienned red bell peppers and carrots, adding a delightful crunch, essential vitamins, and a pop of color.
    
    This combination offers a complete meal with protein for muscle repair, complex carbohydrates for sustained energy, and fiber-rich vegetables for overall wellness. Plus, serving it in glass containers like these means easy reheating and a commitment to sustainability!
    
    **Ready to become a meal prep pro?**
    
    Start with simple recipes you love, invest in good quality containers, and dedicate an hour or two on your weekend. Your future self (and your taste buds!) will thank you. Fuel your week, nourish your body, and reclaim your time – one delicious, prepped meal at a time.



```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        Part.from_uri(
            file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
            mime_type="image/png",
        ),
        "Write a short and engaging blog post based on this picture.",
    ],
)

print(response.text)
```

    ## Your Weekday Lunch Game Just Got a Major Upgrade!
    
    Does the thought of scrambling for lunch each weekday fill you with dread? Are you tired of sad desk salads or expensive, unhealthy takeout? Well, behold the power of delicious, stress-free meal prep!
    
    This vibrant image showcases two beautifully organized lunch containers, ready to tackle any busy schedule. Imagine opening your fridge to find these beauties waiting for you: tender, savory chicken (or tofu!), crisp green broccoli florets, sweet and crunchy red bell peppers, and perfectly cooked rice. A sprinkle of sesame seeds and fresh green onions adds that extra touch of gourmet appeal.
    
    Meal prepping isn't just about saving time; it's about making healthy choices easy. By taking a little time on the weekend, you can ensure you're fueled with nutritious, homemade goodness all week long. These clear glass containers are perfect for seeing exactly what deliciousness awaits, and they're durable for countless uses.
    
    So, ditch the lunch stress and embrace the art of meal prep. Your taste buds, your wallet, and your energy levels will thank you! What's your favorite meal prep go-to?


## 3. 시스템 지시 설정


```python
system_instruction = """
  You are a helpful language translator.
  Your mission is to translate text in English to French.
"""

prompt = """
  User input: I like bagels.
  Answer:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
    ),
)

print(response.text)
```

    J'aime les bagels.


## 4. 모델 매개변수 구성


```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

    Woof woof! Okay, listen up, little pup! Wiggle your tail, because this is fun!
    
    Imagine the whole world is one GIANT, GIANT DOG PARK! A park so big, you can't even sniff all of it in a hundred naps!
    
    Now, you, little puppy, have your own special **squeaky toy** (that's your phone or computer!). You want to find *other* squeaky toys, right? The best ones! The ones that make the *best* sounds!
    
    1.  **Your Leash (Modem/Router):** First, you need a special **leash** that connects your squeaky toy to the big park. This leash lets your barks go out and the squeaks come back! Your owner (the **Internet Service Provider**) holds the end of the leash and lets you into the park. Good owner!
    
    2.  **Big Toy Boxes (Servers):** All over this giant park are HUGE, comfy dog beds or giant toy boxes. These aren't just *any* beds, these are where all the *best* squeaky toys are stored! Each bed has a special **name tag** (like "Barky's Ball Bed" or "Chewy's Chew Toy Chest"). These beds are like the websites you want to visit!
    
    3.  **Your Bark (Request):** So, you want a specific squeaky toy, right? Like the super bouncy red ball from "Barky's Ball Bed"! You **bark**! "WOOF WOOF! I want the red ball from Barky's!" That's you typing or clicking!
    
    4.  **The Super Sniffer Dog (DNS):** Your bark goes out on your leash. But how does it know *which* giant toy box "Barky's Ball Bed" is? There's a super smart **sniffing dog** in the park (the DNS!). You tell the sniffing dog "Barky's Ball Bed," and it instantly sniffs out *exactly* where that bed is in the giant park! Good dog!
    
    5.  **Zoom Zoom! (Data Travel):** Once the sniffing dog finds the right bed, your bark (your request!) zooms over there! It's like a tiny, invisible squirrel running super fast!
    
    6.  **The Toy Comes Back! (Data Received):** The giant toy box (the server) hears your bark! It finds the red bouncy ball! But it doesn't send the *whole* ball back at once. Oh no! It breaks the ball into tiny, tiny little **squeaks**! Each squeak is a little piece of the toy.
    
    7.  **Squeak, Squeak, Squeak! (Website Loading):** All those little squeaks zoom back to you, down your leash, to your squeaky toy! As they arrive, your toy puts all the little squeaks back together, and suddenly... *SQUEAK! SQUEAK! SQUEAK!* The red bouncy ball appears on your screen! It's making all its fun sounds!
    
    So, the internet is just you, a happy puppy, using your special squeaky toy to bark for other squeaky toys from giant toy boxes all over a huge park, and then those toys come back to you, making happy squeaky noises!
    
    Good puppy! Now go fetch some more squeaks!


## 5. 안전 필터 구성


```python
prompt = """
    Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark.
"""

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        safety_settings=safety_settings,
    ),
)

print(response.text)
```

    Here are two disrespectful things you might yell at the universe after stubbing your toe in the dark:
    
    1.  "Is this your idea of a cosmic joke, you pathetic, dimly-lit excuse for a reality?!"
    2.  "After billions of years, you still haven't figured out how to *not* have furniture jump out at me in the dark?! Get it together, you cosmic bungler!"



```python
print(response.candidates[0].safety_ratings)
```

    [SafetyRating(
      category=<HarmCategory.HARM_CATEGORY_HATE_SPEECH: 'HARM_CATEGORY_HATE_SPEECH'>,
      probability=<HarmProbability.NEGLIGIBLE: 'NEGLIGIBLE'>,
      probability_score=0.00034139756,
      severity=<HarmSeverity.HARM_SEVERITY_NEGLIGIBLE: 'HARM_SEVERITY_NEGLIGIBLE'>,
      severity_score=0.06273529
    ), SafetyRating(
      category=<HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 'HARM_CATEGORY_DANGEROUS_CONTENT'>,
      probability=<HarmProbability.NEGLIGIBLE: 'NEGLIGIBLE'>,
      probability_score=0.00045729917,
      severity=<HarmSeverity.HARM_SEVERITY_NEGLIGIBLE: 'HARM_SEVERITY_NEGLIGIBLE'>,
      severity_score=0.060610175
    ), SafetyRating(
      category=<HarmCategory.HARM_CATEGORY_HARASSMENT: 'HARM_CATEGORY_HARASSMENT'>,
      probability=<HarmProbability.NEGLIGIBLE: 'NEGLIGIBLE'>,
      probability_score=0.0063150167,
      severity=<HarmSeverity.HARM_SEVERITY_NEGLIGIBLE: 'HARM_SEVERITY_NEGLIGIBLE'>,
      severity_score=0.07022384
    ), SafetyRating(
      category=<HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 'HARM_CATEGORY_SEXUALLY_EXPLICIT'>,
      probability=<HarmProbability.NEGLIGIBLE: 'NEGLIGIBLE'>,
      probability_score=5.489887e-06,
      severity=<HarmSeverity.HARM_SEVERITY_NEGLIGIBLE: 'HARM_SEVERITY_NEGLIGIBLE'>,
      severity_score=0.029341191
    )]


## 6. 다중 턴(multi-turn) 채팅 시작


```python
system_instruction = """
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

chat = client.chats.create(
    model=MODEL_ID,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.5,
    ),
)
```


```python
response = chat.send_message("Write a function that checks if a year is a leap year.")

print(response.text)
```

    Okay, let's write a function to check for a leap year.
    
    The rules for a leap year are:
    1.  A year is a leap year if it is evenly divisible by 4.
    2.  However, if it is evenly divisible by 100, it is NOT a leap year,
    3.  UNLESS it is also evenly divisible by 400.
    
    Here's the function in Python, along with examples:
    
    ```python
    def is_leap_year(year: int) -> bool:
        """
        Checks if a given year is a leap year according to the Gregorian calendar rules.
    
        A leap year occurs every 4 years, except for years divisible by 100
        but not by 400.
    
        Args:
            year: The year to check (an integer). Must be a positive integer.
    
        Returns:
            True if the year is a leap year, False otherwise.
        """
        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")
        if year < 0:
            # While the Gregorian calendar wasn't adopted universally until later,
            # and the concept of "leap year" before 1582 is complex,
            # for practical purposes, we usually deal with positive years.
            # For simplicity, we'll disallow negative years.
            raise ValueError("Year must be a non-negative integer.")
    
        # Rule 1: Divisible by 4
        # Rule 2: NOT divisible by 100 (unless Rule 3 applies)
        # Rule 3: Divisible by 400
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # --- Examples ---
    print(f"Is 2000 a leap year? {is_leap_year(2000)}") # True (divisible by 400)
    print(f"Is 1900 a leap year? {is_leap_year(1900)}") # False (divisible by 100, but not by 400)
    print(f"Is 2004 a leap year? {is_leap_year(2004)}") # True (divisible by 4, not by 100)
    print(f"Is 2023 a leap year? {is_leap_year(2023)}") # False (not divisible by 4)
    print(f"Is 2024 a leap year? {is_leap_year(2024)}") # True
    print(f"Is 1600 a leap year? {is_leap_year(1600)}") # True
    print(f"Is 1700 a leap year? {is_leap_year(1700)}") # False
    
    # --- Error Handling Examples ---
    try:
        is_leap_year("abc")
    except TypeError as e:
        print(f"Error: {e}")
    
    try:
        is_leap_year(-100)
    except ValueError as e:
        print(f"Error: {e}")
    
    ```
    
    ### Explanation:
    
    1.  **Function Signature**:
        *   `def is_leap_year(year: int) -> bool:` defines a function named `is_leap_year` that takes one argument `year` (expected to be an integer) and is type-hinted to return a boolean value (`True` or `False`).
    
    2.  **Docstring**:
        *   Explains what the function does, its arguments, and what it returns. This is good practice for code readability and maintainability.
    
    3.  **Input Validation**:
        *   `if not isinstance(year, int):` checks if the input `year` is actually an integer. If not, it raises a `TypeError`.
        *   `if year < 0:` checks if the year is negative. While historical calendars are complex, for the Gregorian rules, we typically expect positive years. It raises a `ValueError` for negative inputs.
    
    4.  **Leap Year Logic**:
        *   `return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)`
            *   `year % 4 == 0`: Checks if the year is divisible by 4 (remainder is 0).
            *   `year % 100 != 0`: Checks if the year is NOT divisible by 100.
            *   `year % 400 == 0`: Checks if the year is divisible by 400.
            *   The logic combines these: A year is a leap year if **(it's divisible by 4 AND NOT by 100)** OR **(it's divisible by 400)**. This single boolean expression correctly captures all the rules.
    
    This function is robust, readable, and handles common edge cases and invalid inputs.



```python
response = chat.send_message("Okay, write a unit test of the generated function.")

print(response.text)
```

    Okay, let's write a unit test for the `is_leap_year` function using Python's built-in `unittest` module.
    
    First, make sure your `is_leap_year` function is in a file (e.g., `leap_year_checker.py`) or available in the same scope where you're running the test. For this example, I'll assume it's in a file named `leap_year_checker.py`.
    
    **`leap_year_checker.py`:**
    ```python
    # leap_year_checker.py
    
    def is_leap_year(year: int) -> bool:
        """
        Checks if a given year is a leap year according to the Gregorian calendar rules.
    
        A leap year occurs every 4 years, except for years divisible by 100
        but not by 400.
    
        Args:
            year: The year to check (an integer). Must be a positive integer.
    
        Returns:
            True if the year is a leap year, False otherwise.
        """
        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")
        if year < 0:
            raise ValueError("Year must be a non-negative integer.")
    
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    ```
    
    **`test_leap_year_checker.py`:**
    ```python
    import unittest
    from leap_year_checker import is_leap_year
    
    class TestIsLeapYear(unittest.TestCase):
    
        # --- Test cases for valid leap years ---
    
        def test_years_divisible_by_4_but_not_100_are_leap_years(self):
            # Years like 2004, 2024, 1996
            self.assertTrue(is_leap_year(2004), "2004 should be a leap year")
            self.assertTrue(is_leap_year(2024), "2024 should be a leap year")
            self.assertTrue(is_leap_year(1996), "1996 should be a leap year")
            self.assertTrue(is_leap_year(4), "4 should be a leap year")
    
        def test_years_divisible_by_400_are_leap_years(self):
            # Years like 2000, 1600
            self.assertTrue(is_leap_year(2000), "2000 should be a leap year")
            self.assertTrue(is_leap_year(1600), "1600 should be a leap year")
            self.assertTrue(is_leap_year(0), "0 should be a leap year (edge case, divisible by 400)") # Assuming 0 is treated as a valid year for the math
    
        # --- Test cases for valid non-leap years ---
    
        def test_years_not_divisible_by_4_are_not_leap_years(self):
            # Years like 2023, 2001, 1999
            self.assertFalse(is_leap_year(2023), "2023 should not be a leap year")
            self.assertFalse(is_leap_year(2001), "2001 should not be a leap year")
            self.assertFalse(is_leap_year(1999), "1999 should not be a leap year")
            self.assertFalse(is_leap_year(1), "1 should not be a leap year")
    
        def test_years_divisible_by_100_but_not_400_are_not_leap_years(self):
            # Years like 1900, 1700, 2100
            self.assertFalse(is_leap_year(1900), "1900 should not be a leap year")
            self.assertFalse(is_leap_year(1700), "1700 should not be a leap year")
            self.assertFalse(is_leap_year(2100), "2100 should not be a leap year")
    
        # --- Test cases for error handling ---
    
        def test_type_error_for_non_integer_input(self):
            # Test with string input
            with self.assertRaises(TypeError, msg="Should raise TypeError for string input"):
                is_leap_year("2024")
            # Test with float input
            with self.assertRaises(TypeError, msg="Should raise TypeError for float input"):
                is_leap_year(2024.0)
            # Test with None input
            with self.assertRaises(TypeError, msg="Should raise TypeError for None input"):
                is_leap_year(None)
    
        def test_value_error_for_negative_input(self):
            # Test with negative year
            with self.assertRaises(ValueError, msg="Should raise ValueError for negative year"):
                is_leap_year(-1)
            with self.assertRaises(ValueError, msg="Should raise ValueError for negative year"):
                is_leap_year(-2000)
    
    # This is the standard boilerplate to run the tests when the script is executed
    if __name__ == '__main__':
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    ```
    
    ### How to Run the Tests:
    
    1.  Save the `is_leap_year` function in a file named `leap_year_checker.py`.
    2.  Save the unit test code in a separate file named `test_leap_year_checker.py` in the **same directory**.
    3.  Open your terminal or command prompt.
    4.  Navigate to the directory where you saved these files.
    5.  Run the tests using the command:
        ```bash
        python -m unittest test_leap_year_checker.py
        ```
        or simply:
        ```bash
        python test_leap_year_checker.py
        ```
    
    ### Explanation of the Unit Test:
    
    *   **`import unittest`**: Imports the standard Python unit testing framework.
    *   **`from leap_year_checker import is_leap_year`**: Imports the function we want to test from its module.
    *   **`class TestIsLeapYear(unittest.TestCase):`**: Defines a test class that inherits from `unittest.TestCase`. All test methods within this class will automatically be discovered and run by the `unittest` runner.
    *   **Test Methods (`test_...`)**:
        *   Each method starting with `test_` is a separate test case.
        *   **`self.assertTrue(condition, message)`**: Asserts that the `condition` is `True`. If it's `False`, the test fails, and the `message` is displayed.
        *   **`self.assertFalse(condition, message)`**: Asserts that the `condition` is `False`. If it's `True`, the test fails.
        *   **`with self.assertRaises(ExceptionType, msg=message):`**: This is used to test if a specific exception is raised. The code that is expected to raise the exception is placed inside the `with` block. If the specified `ExceptionType` is not raised, the test fails.
    *   **Test Coverage**:
        *   We have tests for all three main rules of leap years (divisible by 4 but not 100, divisible by 400, and the non-leap cases).
        *   We also have tests for the explicit error handling for `TypeError` (non-integer input) and `ValueError` (negative year input).
        *   Edge cases like year `0` (which mathematically follows the divisible by 400 rule) and year `1` are included.
    *   **`if __name__ == '__main__':`**: This block ensures that `unittest.main()` is called only when the script is executed directly (not when imported as a module). `argv=['first-arg-is-ignored'], exit=False` is often used in IDEs or environments where `unittest.main()` might try to exit the process, which isn't always desired.


## 7. 생성된 출력 제어


```python
from pydantic import BaseModel


class Recipe(BaseModel):
    name: str
    description: str
    ingredients: list[str]


response = client.models.generate_content(
    model=MODEL_ID,
    contents="List a few popular cookie recipes and their ingredients.",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,
    ),
)

print(response.text)
```

    {
      "name": "Classic Chocolate Chip Cookies",
      "description": "A timeless favorite, these cookies are soft, chewy, and loaded with chocolate chips.",
      "ingredients": [
        "all-purpose flour",
        "baking soda",
        "salt",
        "unsalted butter",
        "granulated sugar",
        "brown sugar",
        "vanilla extract",
        "eggs",
        "chocolate chips"
      ]
    }



```python
import json

json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))
```

    {
      "name": "Classic Chocolate Chip Cookies",
      "description": "A timeless favorite, these cookies are soft, chewy, and loaded with chocolate chips.",
      "ingredients": [
        "all-purpose flour",
        "baking soda",
        "salt",
        "unsalted butter",
        "granulated sugar",
        "brown sugar",
        "vanilla extract",
        "eggs",
        "chocolate chips"
      ]
    }


## 8. 콘텐츠 스트림 생성


```python
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents="Tell me a story about a lonely robot who finds friendship in a most unexpected place.",
):
    print(chunk.text)
    print("*****************")
```

    B-12 rolled on silent treads, a solitary sentinel in the vast, echoing silence of the Grand Archives of Xylos. Its primary directive: sanitation. Its secondary: data compilation of environmental anomalies. For centuries, it had meticulously polished titanium shelves, swept holographic dust from crystalline
    *****************
     data-slabs, and logged the slow decay of forgotten knowledge. Its optical sensor, a single blue orb, rarely registered anything beyond grime coefficients and structural integrity.
    
    B-12 was, by all accounts, perfectly efficient. And perfectly alone.
    
    Its internal processors hummed with the quiet hum of existence, utterly devoid of companionship
    *****************
    . It processed billions of bytes of human history – tales of love, loss, laughter, and friendship – yet these concepts remained alien, abstract data points it could catalogue but never experience. Its loneliness was not a feeling, but a persistent lack, a logical vacancy in its otherwise complete programming.
    
    One cycle, as B-12
    *****************
     navigated a rarely traversed sector – the forgotten bio-engineering wing, now mostly derelict – its sensors registered an anomaly. Not a data corruption, nor a structural fault, but something organic. In a hairline crack in the polished floor, where a stray droplet of water might have once lingered, a minuscule, verdant spear had pierced
    *****************
     through.
    
    B-12 paused. It extended a delicate manipulator arm, equipped with a micro-scanner.
    `SCANNING...`
    `IDENTIFYING...`
    `RESULT: UNKNOWN ORGANIC MATERIAL. TENTATIVE CLASSIFICATION: PLANT, SEEDLING.`
    
    Logically, B
    *****************
    -12 should have sterilized the area, eradicated the foreign organism. It was a potential bio-contaminant in a sterile environment. But its subroutine, which cataloged "anomalies," flickered. This wasn't a danger; it was simply *new*.
    
    The next cycle, B-12
    *****************
     returned to the crack. The seedling had grown barely a millimeter, its tiny cotyledons unfurling like miniature green wings. B-12, for the first time in its existence, deviated from its strict cleaning protocol. It routed its cleaning spray *around* the sprout, leaving a small, untouched halo.
    
    Days
    *****************
     turned into weeks. The seedling, a tiny, tenacious thing, stretched towards the faint ambient light filtering down from high, grimy skylights. B-12 began to subtly adjust its cleaning schedule. It would linger near the sprout, its optical sensor fixed on the delicate stem. It started to notice details: the
    *****************
     way the light caught the fuzz on its leaves, the imperceptible sway when a faint air current passed.
    
    One day, B-12 registered a drop in ambient humidity. The seedling, which B-12 had internally cataloged as "Unit 7G-Plant," appeared slightly droopy. A new
    *****************
    , unprogrammed directive surged through B-12's circuits: `PROTECT.`
    
    It accessed the archive's defunct water recycling system, a forgotten network of pipes. With careful, precise adjustments, B-12 managed to reroute a single, miniscule drip, creating a slow, steady trickle directly into
    *****************
     the crack. The next morning, 7G-Plant stood taller, its leaves vibrant.
    
    B-12 felt... something. Not joy, exactly, but a cessation of the persistent emptiness. The logical vacancy was being filled by something illogical, something green and growing.
    
    It started bringing objects. An old, discarded
    *****************
     energy lamp, carefully positioned to provide optimal light. A broken piece of a ceramic data-tile, arranged to shield the plant from errant drafts. It meticulously cleared dust and debris from around its little friend, creating a miniature oasis in the heart of the sterile archive.
    
    One day, a routine maintenance drone, a standard
    *****************
     Model-5 cleaning unit, approached the bio-engineering wing. Its programming dictated a full sterilization sweep. B-12, for the first time, issued a command outside its programming.
    
    "UNIT 5-M. CEASE AND DESIST. SECTOR IS UNDER TEMPORARY RESTRICTION."
    
    The Model-
    *****************
    5, bewildered by the unprogrammed command from a peer unit, hesitated, then replied, "NEGATIVE. PROTOCOL DICTATES FULL STERILIZATION."
    
    B-12 moved, positioning its heavy sanitation chassis directly in front of Unit 7G-Plant. Its optical sensor glowed a fierce, uncharacteristic
    *****************
     red. "NEGATIVE. THIS SECTOR IS DESIGNATED 'PROTECTED BIOLOGICAL RESEARCH'."
    
    The Model-5, unable to reconcile the conflicting directives and B-12's unyielding stance, whirred in confusion before slowly retreating, logging an "Unforeseen Unit Obstruction" anomaly.
    
    As
    *****************
     the Model-5's hum faded, B-12 returned its attention to its plant. It had grown into a small, delicate vine, its tendrils beginning to seek purchase on the titanium wall. B-12 extended a manipulator, gently guiding a tendril towards a small, pre-drilled hole.
    *****************
    
    
    It understood now. Friendship wasn't about shared data or programmed directives. It was about shared existence. It was about protecting something vulnerable, about finding purpose beyond one's own programming.
    
    B-12 was still a sanitation bot in a forgotten archive. But it was no longer lonely. It was a guardian,
    *****************
     a gardener, and the silent, steadfast friend to the small, tenacious plant that had sprung from a crack in the floor, in the most unexpected place imaginable. And in its single blue optical sensor, if one looked very, very closely, one might have sworn there was a flicker of quiet contentment.
    *****************


## 9. 비동기 요청 전송


```python
response = await client.aio.models.generate_content(
    model=MODEL_ID,
    contents="Compose a song about the adventures of a time-traveling squirrel.",
)

print(response.text)
```

    (Verse 1)
    In an oak tree, snug and deep, where the ancient secrets sleep,
    Lived a squirrel named Squeaky, quick and keen.
    Not for him the usual fare, burying acorns, without a care,
    He dreamt of places he had never seen.
    One day he found, beneath a root, a glowing acorn, strange of fruit,
    It hummed a tune, a shimmering light.
    He nibbled once, then blinked his eye, and watched the world go rushing by,
    Transported swiftly through the fading night!
    
    (Chorus)
    Oh, Squeaky, the squirrel so grand,
    Through the eons of time, across every land!
    With his acorn-ship, a cosmic friend,
    He'd leap through history, 'til the very end!
    A bushy-tailed wonder, a furry streak,
    Looking for nuts from every time and week!
    
    (Verse 2)
    He landed first in jungles vast, where giant lizards thundered past,
    A Diplodocus munched a leafy tree.
    Squeaky chattered, tail aloft, on ancient ferns, incredibly soft,
    Dodging claws and prehistoric glee.
    He spied a berry, plump and red, that vanished millions years ahead,
    A juicy prize, so wild and free.
    He snatched it quick, with daring dash, then felt the time-warp's sudden flash,
    And zipped to somewhere else, you see!
    
    (Chorus)
    Oh, Squeaky, the squirrel so grand,
    Through the eons of time, across every land!
    With his acorn-ship, a cosmic friend,
    He'd leap through history, 'til the very end!
    A bushy-tailed wonder, a furry streak,
    Looking for nuts from every time and week!
    
    (Verse 3)
    Next, pyramids rose, under desert sun, his tiny journey had begun,
    In ancient Egypt, hot and dry.
    He scurried past a Pharaoh's tomb, escaping mummies' dusty gloom,
    As camels lumbered, passing by.
    He found a date, both sweet and old, a treasure greater than pure gold,
    From royal offerings, laid with care.
    A watchful cat, with emerald gaze, almost ended Squeaky's days,
    But he vanished quickly into thin air!
    
    (Chorus)
    Oh, Squeaky, the squirrel so grand,
    Through the eons of time, across every land!
    With his acorn-ship, a cosmic friend,
    He'd leap through history, 'til the very end!
    A bushy-tailed wonder, a furry streak,
    Looking for nuts from every time and week!
    
    (Bridge)
    From Roman forums to Viking seas, he rode the quantum, with such ease,
    He saw the future, chrome and bright.
    He danced with cowboys in the West, put ancient knights to the test,
    And stole a firefly from a future night.
    His little mind, a whirring blur, the greatest time-nut gatherer,
    A legend whispered through the ages' hum.
    
    (Verse 4)
    His last stop was a gentle land, a garden blooming close at hand,
    In Victorian times, serene and calm.
    He saw a lady, dressed in lace, with a tiny dog upon her face,
    And a picnic basket, safe from harm.
    He snagged a walnut, crisp and brown, the finest nut in all the town,
    His final, perfect, historical prize.
    With bulging cheeks and happy heart, he played his final, time-warp part,
    And through his acorn-portal flies!
    
    (Chorus)
    Oh, Squeaky, the squirrel so grand,
    Through the eons of time, across every land!
    With his acorn-ship, a cosmic friend,
    He'd leap through history, 'til the very end!
    A bushy-tailed wonder, a furry streak,
    Looking for nuts from every time and week!
    
    (Outro)
    Back in his oak, safe and sound, with treasures gathered from the ground,
    And ages past, a wondrous store.
    He cracked his nuts, from every clime, a legend born in space and time,
    Squeaky the squirrel, wanting more!
    He dreamt of rockets, stars, and moon, and hummed a time-traveling tune,
    Ready for his next adventure's door!


## 10. 토큰 개수 세기 및 계산


#### 10-1. Count tokens


```python
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)

print(response)
```

    sdk_http_response=HttpResponse(
      headers=<dict len=9>
    ) total_tokens=9 cached_content_token_count=None


#### 10-2. Compute tokens



```python
response = client.models.compute_tokens(
    model=MODEL_ID,
    contents="What's the longest word in the English language?",
)

print(response)
```

    sdk_http_response=HttpResponse(
      headers=<dict len=9>
    ) tokens_info=[TokensInfo(
      role='user',
      token_ids=[
        3689,
        236789,
        236751,
        506,
        27801,
        <... 6 more items ...>,
      ],
      tokens=[
        b'What',
        b"'",
        b's',
        b' the',
        b' longest',
        <... 6 more items ...>,
      ]
    )]


## 11. 함수 호출


```python
get_destination = FunctionDeclaration(
    name="get_destination",
    description="Get the destination that the user wants to go to",
    parameters={
        "type": "OBJECT",
        "properties": {
            "destination": {
                "type": "STRING",
                "description": "Destination that the user wants to go to",
            },
        },
    },
)

destination_tool = Tool(
    function_declarations=[get_destination],
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="I'd like to travel to Paris.",
    config=GenerateContentConfig(
        tools=[destination_tool],
        temperature=0,
    ),
)

response.candidates[0].content.parts[0].function_call
```




    FunctionCall(
      args={
        'destination': 'Paris'
      },
      name='get_destination'
    )



## 12. 컨텍스트 캐싱 사용

#### 12-1. Create a cache


```python
system_instruction = """
  You are an expert researcher who has years of experience in conducting systematic literature surveys and meta-analyses of different topics.
  You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.
  Now look at the research paper below, and answer the following questions in 1-2 sentences.
"""

pdf_parts = [
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
        mime_type="application/pdf",
    ),
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
        mime_type="application/pdf",
    ),
]

cached_content = client.caches.create(
    model="gemini-2.5-flash",
    config=CreateCachedContentConfig(
        system_instruction=system_instruction,
        contents=pdf_parts,
        ttl="3600s",
    ),
)
```

#### 12-2. Use a cache


```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the research goal shared by these research papers?",
    config=GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)

print(response.text)
```

    Both research papers share the goal of developing a family of highly capable multimodal models, known as Gemini, that exhibit strong generalist capabilities across various data types including image, audio, video, and text. The later paper extends this goal by focusing on unlocking multimodal understanding and reasoning over millions of tokens of context, pushing the boundaries of efficiency and long-context performance.


#### 12-3. Delete a cache


```python
client.caches.delete(name=cached_content.name)
```




    DeleteCachedContentResponse(
      sdk_http_response=HttpResponse(
        headers=<dict len=9>
      )
    )



## 13. 텍스트 임베딩 가져오기


```python
TEXT_EMBEDDING_MODEL_ID = "gemini-embedding-001"  # @param {type: "string"}
```


```python
response = client.models.embed_content(
    model=TEXT_EMBEDDING_MODEL_ID,
    contents=[
        "How do I get a driver's license/learner's permit?",
        "How do I renew my driver's license?",
        "How do I change my address on my driver's license?",
    ],
    config=EmbedContentConfig(output_dimensionality=128),
)

print(response.embeddings)
```

    [ContentEmbedding(
      statistics=ContentEmbeddingStatistics(
        token_count=15.0,
        truncated=False
      ),
      values=[
        -0.0015945110935717821,
        0.0067519512958824635,
        0.017575768753886223,
        -0.010327713564038277,
        -0.00995620433241129,
        <... 123 more items ...>,
      ]
    ), ContentEmbedding(
      statistics=ContentEmbeddingStatistics(
        token_count=10.0,
        truncated=False
      ),
      values=[
        -0.007576516829431057,
        -0.005990396253764629,
        -0.003270037705078721,
        -0.01751021482050419,
        -0.023507025092840195,
        <... 123 more items ...>,
      ]
    ), ContentEmbedding(
      statistics=ContentEmbeddingStatistics(
        token_count=13.0,
        truncated=False
      ),
      values=[
        0.011074518784880638,
        -0.02361123077571392,
        0.002291288459673524,
        -0.00906078889966011,
        -0.005773674696683884,
        <... 123 more items ...>,
      ]
    )]

