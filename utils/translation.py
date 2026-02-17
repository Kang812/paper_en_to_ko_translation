from ollama import chat

try:
    from utils.lang_config import LangConfig
except:
    try:
        from lang_config import LangConfig
    except:
        from .lang_config import LangConfig

def translate_text(text, source_lang, target_lang):
    lang_config = LangConfig()
    source_lang = lang_config.get_ko_to_en(source_lang)
    target_lang = lang_config.get_ko_to_en(target_lang)

    source_lang_code = lang_config.get_lang_code(source_lang)
    target_lang_code = lang_config.get_lang_code(target_lang)
    
    prompt = f"""
            You are a professional {source_lang} ({source_lang_code}) to {target_lang} ({target_lang_code}) translator. 
            Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while 
            adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
            Produce only the {target_lang} translation, without any additional explanations or commentary. 
            Please translate the following {source_lang} text into {target_lang}:
            
            Input Text:
            {text}
            """
    
    response = chat(
                    model='translategemma:12b',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'repeat_penalty': 1.5, 'top_p': 0.9} 
                )
    
    text_content = response.message.content
    return text_content