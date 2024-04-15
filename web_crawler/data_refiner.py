import os
import re

def clean_text(text):
    # Normalize space: convert multiple spaces into one, trim leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove navigation and utility words or phrases
    remove_phrases = [
        'Версія для друку',
        'Twitter',
        'Facebook',
        'Telegram',
        '(Текст для друку)',
        'Друкувати',
        'Допомога',
        'Шрифт',
        '+збільшити',
        '−зменшити',
        'або Ctrl + mouse wheel'
    ]
    for phrase in remove_phrases:
        text = text.replace(phrase, '')
    
    # Compact multiple newlines to a single newline
    text = re.sub(r'\n\s*\n', '\n', text)

    return text

def process_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {input_file_path}, {e}")
        return

    cleaned_content = clean_text(content)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

    print(f"Processed and cleaned the file: {input_file_path}, result saved as: {output_file_path}")

def clean_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'): 
            file_path = os.path.join(directory_path, filename)
            output_file_path = os.path.join(directory_path, f"cleaned_{filename}")
            process_file(file_path, output_file_path)

# Directory containing the text files
current_directory = os.path.dirname(os.path.abspath(__file__))
directory_path = 'textzakon.rada.gov.ua copy'

clean_directory(os.path.join(current_directory, directory_path))