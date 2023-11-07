# Chat
GPT_4 = "gpt-4"
GPT_4_0314 = "gpt-4-0314"
GPT_4_0613 = "gpt-4-0613"
GPT_4_32K = "gpt-4-32k"
GPT_4_32K_0314 = "gpt-4-32k-0314"
GPT_4_32K_0613 = "gpt-4-32k-0613"
GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
GPT_4_VISION_1106_PREVIEW = "gpt-4-1106-preview"
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
GPT_35_TURBO_0301 = "gpt-3.5-turbo-0301"
GPT_35_TURBO_0613 = "gpt-3.5-turbo-0613"
GPT_35_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"
GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"

GPT_MODEL_LIST = [
    GPT_4,
    GPT_4_0314,
    GPT_4_0613,
    GPT_4_32K,
    GPT_4_32K_0314,
    GPT_4_32K_0613,
    GPT_4_1106_PREVIEW,
    GPT_4_VISION_1106_PREVIEW,
    GPT_35_TURBO,
    GPT_35_TURBO_16K,
    GPT_35_TURBO_0301,
    GPT_35_TURBO_0613,
    GPT_35_TURBO_16K_0613,
    GPT_35_TURBO_1106,
]


# Constants for token counts and model encodings
TOKENS_PER_MESSAGE = {GPT_35_TURBO_0301: 4, GPT_4_0314: 3}
TOKENS_PER_NAME = {GPT_35_TURBO_0301: -1, GPT_4_0314: 1}
DEFAULT_MODEL = "cl100k_base"
WARNINGS = {
    GPT_35_TURBO: "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.",
    GPT_4: "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.",
}


# Speech to Text
WHISPER_1 = "whisper-1"

# Whisper Response Format
class WhisperResponseType:
    JSON = 'json'
    TEXT = 'text'
    SRT = 'srt'
    VERBOSE_JSON = 'verbose_json'
    VTT = 'vtt'

# ISO-639-1
class LanguageCodes:
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    CHINESE = "zh"
    HINDI = "hi"
    BENGALI = "bn"
    GREEK = "el"
    HEBREW = "he"
    TURKISH = "tr"
    DUTCH = "nl"
    SWEDISH = "sv"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    INDONESIAN = "id"
    UKRAINIAN = "uk"
    THAI = "th"
    VIETNAMESE = "vi"


# TTS
TTS_1 = "tts-1"
TTS_1_HD = "tts-1-hd"

# Voice
class AIVoice:
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"

# Audio Type
class TTSAudioType:
    MP3 = 'mp3'
    OPUS = 'opus'
    ACC = 'acc'
    FLAC = 'flac'


# Embedding
ADA_002 = "text-embedding-ada-002"
