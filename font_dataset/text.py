import os
import random
import requests
from .font import DSFont
from .helper import char_in_font

__all__ = [
    "random_char",
    "UnqualifiedFontException",
    "CorpusGenerationConfig",
    "CorpusGeneratorManager",
]

# https://zh.wikipedia.org/zh-hans/%E5%B9%B3%E5%81%87%E5%90%8D
hiragana = (
    "ぁあ"
)

# https://zh.wikipedia.org/zh-hans/%E7%89%87%E5%81%87%E5%90%8D
katakana = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヲンヵヶ"

# https://ja.wiktionary.org/wiki/%E4%BB%98%E9%8C%B2:%E5%B8%B8%E7%94%A8%E6%BC%A2%E5%AD%97%E3%81%AE%E4%B8%80%E8%A6%A7
common_kanji = "亜哀"

# https://gist.github.com/simongfxu/13accd501f6c91e7a423ddc43e674c0f
common_simplified_chinese = "一乙"

# https://gist.github.com/simongfxu/13accd501f6c91e7a423ddc43e674c0f
common_traditional_chinese = "一乙"

korean_alphabet = "가각"


class UnqualifiedFontException(Exception):
    def __init__(self, font: DSFont):
        super().__init__(f"Unqualified font: {font.path}")
        self.font = font


def random_char(length: int, font: DSFont, char_set: str) -> str:
    assert length > 0
    assert len(char_set) > 0

    ret = ""
    fail_cnt = 0
    while len(ret) < length:
        char = char_set[random.randint(0, len(char_set) - 1)]
        if char_in_font(char, font.path):
            ret += char
        else:
            fail_cnt += 1
            print(f"FAILING {fail_cnt} for {font.path}")
            if fail_cnt > 2000:
                raise UnqualifiedFontException(font)

    return ret


class CorpusGenerationConfig(object):
    def __init__(
        self,
        min_num_line: int,
        max_num_line: int,
        min_num_char_per_line: int,
        max_num_char_per_line: int,
    ):
        self.min_num_line = min_num_line
        self.max_num_line = max_num_line
        self.min_num_char_per_line = min_num_char_per_line
        self.max_num_char_per_line = max_num_char_per_line


class CommonCorpusGenerator(object):
    def generate_line(self, length: int, font: DSFont) -> str:
        _, _ = length, font
        pass

    def generate(self, config: CorpusGenerationConfig, font: DSFont) -> str:
        num_lines = random.randint(config.min_num_line, config.max_num_line)
        lines = []

        for _ in range(num_lines):
            num_chars = random.randint(
                config.min_num_char_per_line, config.max_num_char_per_line
            )
            lines.append(self.generate_line(num_chars, font))

        return "\n".join(lines)


class JapaneseUtaNetCorpusGenerator(CommonCorpusGenerator):
    def _corpus_generator(self):
        import sqlite3

        self.conn = sqlite3.connect("lyrics_corpus/cache/uta-net.db")
        self.cur = self.conn.cursor()

        while True:
            self.cur.execute(
                "SELECT lyrics FROM lyrics WHERE song_id IN (SELECT song_id FROM lyrics ORDER BY RANDOM() LIMIT 1)"
            )
            row = self.cur.fetchone()
            if row is not None:
                row = str(row[0])
                for line in row.splitlines():
                    if len(line) > 0:
                        yield line
                    continue
            else:
                return

    def _random_place_holder(self, font: DSFont) -> str:
        r = random.randint(1, 3)
        if r == 1:
            ret = random_char(1, font, katakana)
        elif r == 2:
            ret = random_char(1, font, hiragana)
        else:
            ret = random_char(1, font, common_kanji)
        return ret

    def __init__(self):
        self.corpus_iterator = self._corpus_generator()

    def generate_line(self, length: int, font: DSFont) -> str:
        while True:
            try:
                # get new line
                line = next(self.corpus_iterator)

                # filter for font
                ret_line = ""
                for char in line:
                    if char_in_font(char, font.path):
                        ret_line += char
                    else:
                        ret_line += self._random_place_holder(font)

                # truncate or pad
                if len(ret_line) >= length:
                    ret_line = ret_line[:length]
                else:
                    for _ in range(length - len(ret_line)):
                        ret_line += self._random_place_holder(font)

                return ret_line

            except StopIteration:
                self.corpus_iterator = self._corpus_generator()


class RandomCorpusGeneratorWithEnglish(CommonCorpusGenerator):
    def __init__(
        self, char_set: str, prob: float = 0.3, when_length_greater_than: int = 10
    ):
        if os.path.exists("wordlist.txt"):
            with open("wordlist.txt", "r", encoding="utf-8") as f:
                self.english_words = f.read().splitlines()
        else:
            word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
            response = requests.get(word_site)
            self.english_words = response.text.splitlines()
        self.char_set = char_set
        self.prob = prob
        self.when_length_greater_than = when_length_greater_than

    def generate_line(self, length: int, font: DSFont) -> str:
        generate_corpus = random_char(length, font, self.char_set)
        if length > self.when_length_greater_than:
            if random.random() < self.prob:
                random_english_word = random.choice(self.english_words)
                if len(random_english_word) > length:
                    return random_english_word[:length]
                start_place = random.randint(0, length - len(random_english_word))
                ret = (
                    generate_corpus[:start_place]
                    + random_english_word
                    + generate_corpus[start_place + len(random_english_word) :]
                )
                assert len(ret) == length
                return ret
        return generate_corpus


class SimplifiedChineseRandomCorpusGeneratorWithEnglish(
    RandomCorpusGeneratorWithEnglish
):
    def __init__(self, prob: float = 0.3, when_length_greater_than: int = 10):
        super().__init__(common_simplified_chinese, prob, when_length_greater_than)


class TraditionalChineseRandomCorpusGeneratorWithEnglish(
    RandomCorpusGeneratorWithEnglish
):
    def __init__(self, prob: float = 0.3, when_length_greater_than: int = 10):
        super().__init__(common_traditional_chinese, prob, when_length_greater_than)


class KoreanRandomCorpusGeneratorWithEnglish(RandomCorpusGeneratorWithEnglish):
    def __init__(self, prob: float = 0.3, when_length_greater_than: int = 10):
        super().__init__(korean_alphabet, prob, when_length_greater_than)


class EnglishCorpusGenerator(CommonCorpusGenerator):
    def __init__(self):
        if os.path.exists("wordlist.txt"):
            with open("wordlist.txt", "r", encoding="utf-8") as f:
                self.english_words = f.read().splitlines()
        else:
            # Fetch a list of English words from MIT's website
            word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
            response = requests.get(word_site)
            self.english_words = response.text.splitlines()
            # Save for future use
            with open("wordlist.txt", "w", encoding="utf-8") as f:
                f.write(response.text)

    def generate_line(self, length: int, font: DSFont) -> str:
        words = []
        current_length = 0
        max_attempts = 100  # Prevent endless loops
        attempts = 0
        
        # Keep adding words until we reach or exceed the desired length
        while current_length < length and attempts < max_attempts:
            attempts += 1
            word = random.choice(self.english_words)
            
            # Use only characters that exist in the font
            valid_chars = []
            for char in word:
                if char_in_font(char, font.path):
                    valid_chars.append(char)
                    
            # If we found at least one valid character, use it
            if valid_chars:
                valid_word = ''.join(valid_chars)
                if len(valid_word) > 0:
                    words.append(valid_word)
                    current_length += len(valid_word) + 1  # +1 for space
                    attempts = 0  # Reset attempts after success
            
            # If we've tried too many times and still don't have content, fall back to basic ASCII
            if attempts >= max_attempts and not words:
                print(f"Font {font.path} has limited English support. Using fallback ASCII.")
                basic_ascii = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                for char in basic_ascii:
                    if char_in_font(char, font.path):
                        words.append(char)
                        current_length += 2  # Character plus space
                        if current_length >= length:
                            break
                
                # If still no characters are supported, raise exception
                if not words:
                    raise UnqualifiedFontException(font)
        
        # Join words with spaces and truncate/pad to exact length
        line = " ".join(words)
        if len(line) > length:
            line = line[:length]
        elif len(line) < length:
            # Pad with spaces if needed
            line = line + " " * (length - len(line))
        
        return line


class CorpusGeneratorManager:
    def __init__(self):
        self.generators = {
            "ja": JapaneseUtaNetCorpusGenerator(),
            "zh-Hans": SimplifiedChineseRandomCorpusGeneratorWithEnglish(),
            "zh-Hant": TraditionalChineseRandomCorpusGeneratorWithEnglish(),
            "ko": KoreanRandomCorpusGeneratorWithEnglish(),
            "en": EnglishCorpusGenerator(),
        }

    def _get_generator(
        self, font: DSFont, CJK_language: str = None
    ) -> CommonCorpusGenerator:
        langauge = CJK_language if CJK_language is not None else font.language

        for k, v in self.generators.items():
            if langauge.startswith(k):
                return v

        raise Exception(f"no generator for {font.language}")

    def generate(
        self, config: CorpusGenerationConfig, font: DSFont, CJK_language: str = None
    ) -> str:
        return self._get_generator(font, CJK_language).generate(config, font)
