import os
import re
import pandas

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

def toCSV(directory_path_with_files, csv_file_path):
    processed_csv_directory_path = csv_file_path
    if not os.path.exists(processed_csv_directory_path):
        os.mkdir(processed_csv_directory_path)
    
    texts = []
    for filename in os.listdir(directory_path_with_files):
        if filename.endswith('.txt'): 
            file_path = os.path.join(directory_path_with_files, filename)
            try:    
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    
                    # Omit the first 11 lines and the last 4 lines, then replace special characters with spaces
                    texts.append((text[11:-4].replace('-',' ').replace('_', ' ').replace(':',' '), text))
            except Exception as e:
                    print(f"Error reading file: {file_path}, {e}. Ignoring this file.")
                    continue
                
    df = pandas.DataFrame(texts, columns = ['fname', 'text'])
    df.to_csv(f"{processed_csv_directory_path}/scraped.csv")
    df.head()

def clean_directory(directory_path):
    processed_directory_path = os.path.join(directory_path, 'processed')
    if not os.path.exists(processed_directory_path):
        os.mkdir(processed_directory_path)
        
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'): 
            file_path = os.path.join(directory_path, filename)
            output_file_path = os.path.join(processed_directory_path, filename)
            process_file(file_path, output_file_path)

# Directory containing the text files
current_directory = os.path.dirname(os.path.abspath(__file__))
directory_path = 'textzakon.rada.gov.ua copy'

clean_directory(os.path.join(current_directory, directory_path))
toCSV(os.path.join(current_directory, directory_path, 'processed'), os.path.join(current_directory, 'processed'))