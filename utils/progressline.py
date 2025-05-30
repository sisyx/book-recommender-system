def progress_bar(percentage=0, filled_char='#', remaining_char="-"):
    length = 20
    percentage = max(0, min(percentage, 100))
    filled_chars_count = int(length * percentage / 100)
    remaining_chars_count = length - filled_chars_count
    bar = filled_chars_count * filled_char + remaining_chars_count * remaining_char
    return f"{bar} {percentage:.1f}%"

