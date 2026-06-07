"""Tests for translators/utils/common.parse_response — parsing LLM output
(plain JSON, fenced code blocks, YAML) into Python objects."""
from manga_translator.translators.utils.common import parse_response


def test_plain_json_array():
    out = parse_response('[{"text_no": 0, "translated_text": "a"}]')
    assert out == [{"text_no": 0, "translated_text": "a"}]


def test_json_fenced_block():
    content = 'here you go:\n```json\n[{"text_no": 1, "translated_text": "b"}]\n```\nthanks'
    assert parse_response(content) == [{"text_no": 1, "translated_text": "b"}]


def test_generic_fenced_block():
    content = '```\n[{"text_no": 2, "translated_text": "c"}]\n```'
    assert parse_response(content) == [{"text_no": 2, "translated_text": "c"}]


def test_yaml_list():
    content = "- text_no: 0\n  translated_text: hello\n"
    out = parse_response(content)
    assert out == [{"text_no": 0, "translated_text": "hello"}]


def test_garbage_returns_empty_list():
    assert parse_response("this is not json or yaml: {{{ ][") == []


def test_scalar_not_dict_or_list_returns_empty_list():
    # "5" parses as int 5 -> not dict/list -> []
    assert parse_response("5") == []


def test_json_object_passthrough():
    out = parse_response('{"text_no": 0, "translated_text": "x"}')
    assert out == {"text_no": 0, "translated_text": "x"}
