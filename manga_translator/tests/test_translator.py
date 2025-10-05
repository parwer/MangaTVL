from manga_translator.translators.groq import AsyncGroqTranslator
from manga_translator.schemas.interface import OCRResult, DetectionResult
from groq import AsyncGroq
import os
import asyncio

aclient = AsyncGroq(api_key="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")
concurrent_requests = int(os.getenv("CONCURRENT_REQUESTS", 1))

ocr_results = [OCRResult(text='IT WOULD BE ABOUT THAT, YES.', boxes=[[229, 311, 270, 348], [164, 342, 324, 390], [216, 390, 275, 426], [167, 430, 315, 474], [184, 476, 299, 520], [189, 521, 291, 563]], detection_result=DetectionResult(bbox=[155, 268, 355, 592], form_type='bubble')),
 OCRResult(text="C'MON, IORI. WE'RE FAMILY. NO NEED TO BE SO FORMAL.", boxes=[[1425, 815, 1711, 861], [1407, 864, 1733, 908], [1464, 910, 1673, 954], [1452, 956, 1683, 1002], [1464, 1002, 1664, 1048]], detection_result=DetectionResult(bbox=[1373, 627, 1741, 1221], form_type='bubble')),
 OCRResult(text="YOU'VE SURE GROWN. IT'S BEEN, WHAT, TEN YEARS?", boxes=[[1046, 237, 1174, 270], [1061, 271, 1159, 304], [1040, 305, 1179, 341], [1076, 338, 1146, 374], [1060, 370, 1161, 410], [1061, 405, 1159, 443], [1077, 437, 1148, 474], [1046, 469, 1181, 509]], detection_result=DetectionResult(bbox=[1005, 178, 1201, 585], form_type='bubble')),
 OCRResult(text='DO YA, THOUGH ?', boxes=[[513, 720, 694, 781], [489, 775, 723, 832], [581, 828, 632, 879]], detection_result=DetectionResult(bbox=[478, 629, 731, 1018], form_type='bubble')),
 OCRResult(text='AH, CERTAINLY. I UNDER- STAND.', boxes=[[842, 1467, 925, 1520], [756, 1513, 1001, 1564], [769, 1561, 991, 1612], [797, 1611, 966, 1662]], detection_result=DetectionResult(bbox=[736, 1359, 1075, 1729], form_type='bubble'))]

async def test_groq_translator():
    translator = AsyncGroqTranslator(
        client=aclient,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        concurrent_limit=concurrent_requests,
    )
    response = await translator.translate(ocr_results)
    return response

if __name__ == "__main__":
    output = asyncio.run(test_groq_translator())
    print(output)