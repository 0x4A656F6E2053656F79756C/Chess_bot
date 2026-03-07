import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def remove_hash_comments(code):
    # 문자열 내부의 #은 보호하고, 진짜 # 주석만 지우는 정규표현식
    pattern = re.compile(
        r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")|"
        r"(#[^\n]*)",
    )

    def replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    cleaned_code = pattern.sub(replacer, code)
    cleaned_code = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_code)
    return cleaned_code

def main():
    root = tk.Tk()
    root.withdraw() 

    # 1. 원본 파일 선택
    input_filepath = filedialog.askopenfilename(
        title="주석을 제거할 파이썬 파일 선택",
        filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
    )

    if not input_filepath:
        return

    # 2. 원본 파일 이름에서 기본 저장 파일명 만들기 (예: test.py -> test_removed.py)
    original_filename = os.path.basename(input_filepath) # 파일명만 추출 (예: test.py)
    name, ext = os.path.splitext(original_filename)      # 이름과 확장자 분리 (예: test, .py)
    default_save_name = f"{name}_removed{ext}"

    # 3. 저장할 경로 및 파일명 선택
    output_filepath = filedialog.asksaveasfilename(
        title="새로 저장할 파이썬 파일 경로 선택",
        initialfile=default_save_name, # 기본 파일명 지정
        defaultextension=".py",
        filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
    )

    if not output_filepath:
        return

    # 4. 파일 변환 및 저장
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        cleaned_code = remove_hash_comments(code)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_code)
            
        messagebox.showinfo("완료", "주석 제거가 성공적으로 완료되었습니다!")

    except Exception as e:
        messagebox.showerror("오류", f"작업 중 다음 오류가 발생했습니다:\n{e}")

if __name__ == "__main__":
    main()